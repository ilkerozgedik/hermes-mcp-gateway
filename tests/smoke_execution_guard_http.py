"""Live MCP Execution Guard + resumable job smoke on an isolated or deployed Gateway."""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid

import httpx
from e2e_native import rpc, text


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3062")
    parser.add_argument("--workdir", default="/home/hermes/work/hermes-mcp-gateway")
    args = parser.parse_args()
    async with httpx.AsyncClient(base_url=args.url, timeout=35, trust_env=False) as client:
        async def invoke(name, arguments, token=None):
            bridge = {"calls": [{"name": name, "arguments": arguments}]}
            if token:
                bridge["session_token"] = token
            return await rpc(client, "tools/call", {"name": "tool_call", "arguments": bridge})

        lease = await invoke("session_open", {})
        session = json.loads(text(lease))["session_token"]
        denied = await invoke("terminal", {"command": "printf forbidden", "background": True}, session)
        assert denied.get("isError") and "LONG_RUNNING_OPERATION_REQUIRES_JOB" in text(denied), repr(denied)
        denied = await invoke("terminal", {"command": "printf forbidden", "timeout": 120}, session)
        assert denied.get("isError") and "maximum of 30" in text(denied), repr(denied)
        short = await invoke("terminal", {"command": "printf guarded-short-ok"}, session)
        assert not short.get("isError") and "guarded-short-ok" in text(short), text(short)
        timed = await invoke("terminal", {"command": "python3 -c 'import time; time.sleep(4)'", "timeout": 1}, session)
        timed_data = json.loads(text(timed))
        assert timed_data.get("exit_code") not in (None, 0), timed_data.get("status")
        job = await invoke("job_start", {
            "command": "printf recoverable-job-ok",
            "workdir": args.workdir,
            "idempotency_key": "execution-guard-http-" + uuid.uuid4().hex,
            "max_runtime": 60,
        }, session)
        assert not job.get("isError"), text(job)
        receipt = json.loads(text(job))
        assert receipt["state"] == "running" and receipt["recovery_token"]
        lease2 = await invoke("session_open", {})
        session2 = json.loads(text(lease2))["session_token"]
        status = {"state": "running", "exit_code": None}
        for _ in range(20):
            outcome = await invoke("job_status", {
                "job_id": receipt["job_id"], "recovery_token": receipt["recovery_token"],
            }, session2)
            assert not outcome.get("isError"), text(outcome)
            status = json.loads(text(outcome))
            if status["state"] != "running":
                break
            await asyncio.sleep(0.5)
        assert status["state"] == "succeeded" and status["exit_code"] == 0, status["state"]
        print("MCP execution guard smoke passed: background/long denied, short allowed, job recovered.")


if __name__ == "__main__":
    asyncio.run(main())
