"""Real HTTP/native job acceptance with deliberately unread first responses.

Use a private staging Gateway and its separate job database. No credentials or
command payloads are included in the report. This does not claim a ChatGPT
control-plane or VPS-reboot test.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid

import httpx
from e2e_native import PROTOCOL, rpc, text


def request_body(name, arguments, session=None):
    bridge = {"calls": [{"name": name, "arguments": arguments}]}
    if session is not None:
        bridge["session_token"] = session
    return {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": "tools/call", "params": {
        "name": "tool_call", "arguments": bridge,
        "_meta": {"io.modelcontextprotocol/protocolVersion": PROTOCOL,
                  "io.modelcontextprotocol/clientInfo": {"name": "job-recovery-smoke", "version": "1"},
                  "io.modelcontextprotocol/clientCapabilities": {}},
    }}


async def invoke(client, name, arguments, session=None):
    bridge = {"calls": [{"name": name, "arguments": arguments}]}
    if session is not None:
        bridge["session_token"] = session
    return await rpc(client, "tools/call", {"name": "tool_call", "arguments": bridge})


def payload(result):
    if result.get("isError"):
        raise AssertionError("MCP operation failed; inspect the redacted Gateway logs")
    return json.loads(text(result))


def prepared_count(database):
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        return db.execute("SELECT COUNT(*) FROM jobs WHERE state='prepared'").fetchone()[0]


async def until(predicate, seconds=15):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Acceptance condition did not become true before its deadline")
        await asyncio.sleep(0.025)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3062")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--temp-root", default="/home/hermes/worktrees")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.url.rstrip("/") == "http://127.0.0.1:3060":
        parser.error("Use a private staging listener, not production")
    database = (args.state_dir / "gateway-jobs.sqlite3").absolute()
    checks = {}
    headers = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": PROTOCOL,
               "Mcp-Method": "tools/call", "Mcp-Name": "tool_call"}
    async with httpx.AsyncClient(base_url=args.url, timeout=25, trust_env=False) as first:
        session = payload(await invoke(first, "session_open", {}))["session_token"]
        with tempfile.TemporaryDirectory(prefix="gateway-recovery-", dir=args.temp_root) as root:
            workdir = Path(root)
            lost_prepare = workdir / "must-not-run"
            before = prepared_count(database)
            prepare_args = {"command": "printf forbidden > " + shlex.quote(str(lost_prepare)),
                            "workdir": str(workdir), "max_runtime": 30,
                            "idempotency_key": "lost-prepare-" + uuid.uuid4().hex}
            async with first.stream("POST", "/mcp", headers=headers,
                                    json=request_body("job_prepare", prepare_args, session)) as response:
                response.raise_for_status()
                await until(lambda: prepared_count(database) > before)
                assert not response.is_stream_consumed
                # Close without delivering the credential-bearing MCP response.
            assert not lost_prepare.exists()
            checks["lost_prepare_response_has_no_execution"] = True

            marker = workdir / "launch-count"
            command = "printf 'once\\n' >> " + shlex.quote(str(marker)) + "; sleep 1; printf recovered-native-ok"
            receipt = payload(await invoke(first, "job_prepare", {
                "command": command, "workdir": str(workdir), "max_runtime": 30,
                "idempotency_key": "lost-start-" + uuid.uuid4().hex,
            }, session))
            credential = {"job_id": receipt["job_id"], "recovery_token": receipt["recovery_token"]}
            try:
                async with first.stream("POST", "/mcp", headers=headers,
                                        json=request_body("job_start", credential, session)) as response:
                    response.raise_for_status()
                    await until(marker.exists)
                    assert not response.is_stream_consumed
                checks["start_acknowledgment_deliberately_unread"] = True
                # A distinct HTTP client and Gateway session simulate reconnection.
                async with httpx.AsyncClient(base_url=args.url, timeout=25, trust_env=False) as second:
                    next_session = payload(await invoke(second, "session_open", {}))["session_token"]
                    results = await asyncio.gather(*(
                        invoke(second, "job_start", credential, next_session) for _ in range(2)
                    ))
                    assert all(not item.get("isError") for item in results)
                    checks["authorized_duplicate_starts_do_not_relaunch"] = True
                    state = {"state": "running"}
                    deadline = time.monotonic() + 20
                    while time.monotonic() < deadline:
                        state = payload(await invoke(second, "job_status", credential, "z" * 43))
                        if state["state"] not in {"running", "starting", "unknown"}:
                            break
                        await asyncio.sleep(0.1)
                    assert state["state"] == "succeeded" and state["exit_code"] == 0
                    assert "recovered-native-ok" in state["output"]
                    checks["recovery_ignores_unrelated_invalid_session_token"] = True
                    denied = await invoke(second, "job_status", {**credential, "recovery_token": "x" * 43})
                    assert denied.get("isError")
                    checks["wrong_recovery_credential_denied"] = True
                assert marker.read_text().splitlines() == ["once"]
                checks["one_real_native_execution"] = True
                operator = Path(__file__).resolve().parents[1] / "scripts" / "job_admission.py"

                def control(action):
                    result = subprocess.run(
                        [sys.executable, str(operator), action, "--database", str(database), "--timeout", "0"],
                        capture_output=True, text=True, check=True, timeout=5,
                    )
                    return json.loads(result.stdout)

                assert not control("status")["draining"]
                try:
                    control("drain")
                    forbidden = workdir / "must-not-launch-during-drain"
                    blocked = payload(await invoke(first, "job_prepare", {
                        "command": "printf forbidden > " + shlex.quote(str(forbidden)),
                        "workdir": str(workdir), "max_runtime": 30,
                    }, session))
                    blocked_credential = {"job_id": blocked["job_id"], "recovery_token": blocked["recovery_token"]}
                    result = await invoke(first, "job_start", blocked_credential, session)
                    assert result.get("isError") and "draining" in text(result)
                    recovered = payload(await invoke(first, "job_status", credential))
                    assert recovered["state"] == "succeeded"
                    cancelled = payload(await invoke(first, "job_cancel", blocked_credential))
                    assert cancelled["state"] == "cancelled" and not forbidden.exists()
                    checks["operator_drain_blocks_native_launch_not_recovery"] = True
                finally:
                    assert not control("resume")["draining"]
                checks["operator_abort_resume_restores_admission"] = True
            finally:
                await invoke(first, "job_cancel", credential)
            await invoke(first, "session_close", {}, session)
    report = {"status": "passed", "transport": "private HTTP MCP + real native terminal",
              "checks": checks}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": checks, "report": str(args.report)}))


if __name__ == "__main__":
    asyncio.run(main())
