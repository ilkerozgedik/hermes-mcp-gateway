"""Real HTTP MCP migration acceptance probes (no simulated upstream responses)."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import uuid

import httpx
import psutil

PROTOCOL = "2026-07-28"


def unpack(response):
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        payload = json.loads(next(line[6:] for line in response.text.splitlines() if line.startswith("data: ")))
    else:
        payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["result"]


def text(result):
    return "\n".join(x.get("text", "") for x in result.get("content", []) if x.get("type") == "text")


async def rpc(client, method, params, *, legacy_headers=None):
    headers = {"Accept": "application/json, text/event-stream"}
    if legacy_headers is None:
        headers.update({"MCP-Protocol-Version": PROTOCOL, "Mcp-Method": method})
        if method == "tools/call":
            headers["Mcp-Name"] = params["name"]
        params = {**params, "_meta": {
            "io.modelcontextprotocol/protocolVersion": PROTOCOL,
            "io.modelcontextprotocol/clientInfo": {"name": "migration-acceptance", "version": "1"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }}
    else:
        headers.update(legacy_headers)
    return unpack(await client.post("/mcp", headers=headers, json={
        "jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params,
    }))


async def call(client, name, arguments, **kwargs):
    return await rpc(client, "tools/call", {"name": "tool_call", "arguments": {
        "calls": [{"name": name, "arguments": arguments}],
    }}, **kwargs)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3060")
    parser.add_argument("--report", required=True)
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args()
    run = Path(os.environ["TMPDIR"]) / ("migration-e2e-" + uuid.uuid4().hex)
    run.mkdir(mode=0o700)
    fixture = run / "fixture.txt"
    fixture.write_text("native-migration-evidence\n", encoding="utf-8")
    report = {"url": args.url, "fixture_dir": str(run), "checks": {}}
    async with httpx.AsyncClient(base_url=args.url, timeout=360, trust_env=False) as client:
        for path in ["/healthz", "/readyz"]:
            response = await client.get(path)
            report["checks"][path] = {"status": response.status_code, "body": response.json()}
            assert response.status_code == 200, report["checks"][path]
        listing = await rpc(client, "tools/list", {})
        public = {x["name"] for x in listing["tools"]}
        assert public == {"tool_search", "tool_describe", "tool_call", "startup_context", "memory_context", "skills_list"}, public
        report["checks"]["public_tools"] = sorted(public)
        described = await rpc(client, "tools/call", {"name": "tool_describe", "arguments": {"names": ["execute_code"]}})
        assert not described.get("isError"), text(described)
        assert "execute_code" in text(described), text(described)
        report["checks"]["execute_code_schema"] = text(described)
        denied = await call(client, "execute_code", {"code": "raise RuntimeError('must not run')"})
        assert denied.get("isError"), denied
        report["checks"]["missing_confirmation_denied"] = True
        hidden = await rpc(client, "tools/call", {"name": "execute_code", "arguments": {"code": "print('forbidden')", "confirmed": True}})
        assert hidden.get("isError"), hidden
        report["checks"]["hidden_direct_call_denied"] = True
        code = (
            "from hermes_tools import read_file, search_files, terminal\n"
            "import json\n"
            f"r = read_file({str(fixture)!r})\n"
            f"s = search_files('native-migration-evidence', path={str(run)!r})\n"
            "t = terminal('printf gateway-native-rpc-ok')\n"
            "print(json.dumps({'read':r,'search':s,'terminal':t}))\n"
        )
        executed = await call(client, "execute_code", {"code": code, "confirmed": True})
        assert not executed.get("isError"), text(executed)
        payload = json.loads(text(executed))
        assert payload.get("status") == "success", payload
        assert payload.get("tool_calls_made") == 3, payload
        data = json.loads(payload["output"])
        assert "native-migration-evidence" in data["read"]["content"], data
        assert data["terminal"].get("exit_code") == 0, data
        assert "gateway-native-rpc-ok" in data["terminal"].get("output", ""), data
        report["checks"]["native_rpc"] = payload
        marker = "secret_" + uuid.uuid4().hex
        first = await call(client, "execute_code", {"code": f"{marker} = 'private'; print('created')", "confirmed": True})
        assert not first.get("isError"), text(first)
        second = await call(client, "execute_code", {"code": f"print({marker!r} in globals())", "confirmed": True})
        assert not second.get("isError"), text(second)
        assert json.loads(text(second))["output"].strip() == "False", text(second)
        report["checks"]["request_state_isolation"] = True
        retired = await call(client, "ctx_execute", {"language": "python", "code": "print('must not run')"})
        assert retired.get("isError"), retired
        report["checks"]["retired_tool_denied"] = True
        read = await call(client, "read_file", {"path": str(fixture)})
        assert not read.get("isError"), text(read)
        assert "native-migration-evidence" in text(read), read
        report["checks"]["native_file_preserved"] = True
        for name in ["memory_context", "session_search", "cronjob"]:
            params = {"action": "list"} if name == "cronjob" else ({"query": "MCP Gateway", "limit": 1} if name == "session_search" else {})
            if name == "memory_context":
                result = await rpc(client, "tools/call", {"name": name, "arguments": params})
            else:
                result = await call(client, name, params)
            assert not result.get("isError"), (name, text(result))
            report["checks"][name] = {"is_error": False, "response_chars": len(text(result))}
        if args.cancel:
            pidfile = run / "cancel-pid.json"
            code = (
                "import os, subprocess, time, json\nfrom pathlib import Path\n"
                "p=subprocess.Popen(['sleep','120'], start_new_session=True)\n"
                f"Path({str(pidfile)!r}).write_text(json.dumps({{'kernel':os.getpid(),'descendant':p.pid}}))\n"
                "time.sleep(120)\n"
            )
            request = asyncio.create_task(call(client, "execute_code", {"code": code, "confirmed": True}))
            try:
                deadline = time.monotonic() + 30
                while not pidfile.exists() and time.monotonic() < deadline:
                    if request.done():
                        raise AssertionError(text(await request))
                    await asyncio.sleep(0.1)
                assert pidfile.exists(), "execution never reached cancellation fixture"
                pids = json.loads(pidfile.read_text())
                request.cancel()
                try:
                    await request
                except asyncio.CancelledError:
                    pass
                deadline = time.monotonic() + 15
                def running(pid):
                    try:
                        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
                    except psutil.NoSuchProcess:
                        return False
                while any(running(pid) for pid in pids.values()) and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                assert not any(running(pid) for pid in pids.values()), pids
                report["checks"]["disconnect_kills_native_tree"] = pids
            finally:
                request.cancel()
                if pidfile.exists():
                    for pid in json.loads(pidfile.read_text()).values():
                        try:
                            psutil.Process(pid).kill()
                        except psutil.NoSuchProcess:
                            pass
    report["status"] = "passed"
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": list(report["checks"]), "report": args.report}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
