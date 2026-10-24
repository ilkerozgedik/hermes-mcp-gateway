"""Non-destructive four-agent MCP concurrency smoke; requires a private gateway."""
import argparse
import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path

import httpx
from e2e_native import rpc, text


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3062")
    parser.add_argument("--temp-root", default="/home/hermes/worktrees")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=35, trust_env=False) as client:
        async def tool(name, arguments, session=None):
            payload = {"calls": [{"name": name, "arguments": arguments}]}
            if session is not None:
                payload["session_token"] = session
            return await rpc(client, "tools/call", {"name": "tool_call", "arguments": payload})

        sessions = [json.loads(text(await tool("session_open", {})))["session_token"]
                    for _ in range(4)]
        receipts = []
        with tempfile.TemporaryDirectory(prefix="gateway-agents-", dir=args.temp_root) as root:
            dirs = [Path(root) / f"agent-{i}" for i in range(6)]
            for d in dirs:
                d.mkdir()

            async def start(i, session):
                prepared = await tool("job_prepare", {
                    "command": "python3 -c 'import time; time.sleep(15); print(42)'",
                    "workdir": str(dirs[i]),
                    "max_runtime": 90,
                    "idempotency_key": "multi-agent-smoke-" + uuid.uuid4().hex,
                }, session)
                if prepared.get("isError"):
                    return prepared
                receipt = json.loads(text(prepared))
                result = await tool("job_start", {
                    "job_id": receipt["job_id"], "recovery_token": receipt["recovery_token"],
                }, session)
                if result.get("isError"):
                    return result
                return {**receipt, **json.loads(text(result))}

            try:
                first = await asyncio.gather(start(0, sessions[0]), start(1, sessions[0]))
                assert all(x.get("state") == "running" for x in first), first
                receipts.extend(first)
                exceeded = await start(2, sessions[0])
                assert exceeded.get("isError") and "session capacity" in text(exceeded), exceeded

                later = await asyncio.gather(start(2, sessions[1]), start(3, sessions[2]))
                assert all(x.get("state") == "running" for x in later), later
                receipts.extend(later)
                exceeded = await start(4, sessions[3])
                assert exceeded.get("isError") and "capacity" in text(exceeded), exceeded

                # All accepted jobs must be independently recoverable with their token.
                async def recover(receipt):
                    args = {"job_id": receipt["job_id"], "recovery_token": receipt["recovery_token"]}
                    for _ in range(50):
                        result = await tool("job_status", args, sessions[3])
                        assert not result.get("isError"), result
                        state = json.loads(text(result))
                        if state["state"] != "running":
                            return state
                        await asyncio.sleep(0.5)
                    raise AssertionError("job never reached a terminal state")

                results = await asyncio.gather(*(recover(item) for item in receipts))
                assert all(x["state"] == "succeeded" and x["exit_code"] == 0
                           for x in results), results
                print("Four-agent smoke passed: session fairness, global cap, cross-session recovery.")
            finally:
                for item in receipts:
                    try:
                        await tool("job_cancel", {
                            "job_id": item["job_id"],
                            "recovery_token": item["recovery_token"],
                        }, sessions[3])
                    except Exception:  # noqa: BLE001 - preserve the original smoke failure during cleanup
                        print("Warning: smoke job cleanup failed; reconcile its receipt.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
