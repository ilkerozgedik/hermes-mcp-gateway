"""Manual, non-destructive smoke for real Hermes native background jobs.

Run only on the VPS with HERMES_HOME and the pinned Hermes virtualenv.
"""
import asyncio
import tempfile
from pathlib import Path

from hermes_mcp_gateway.config import GatewayConfig
from hermes_mcp_gateway.jobs import ResumableJobs
from hermes_mcp_gateway.upstreams import HermesToolsClient


async def main():
    root = str(Path.cwd())
    with tempfile.TemporaryDirectory() as temp:
        native = HermesToolsClient(GatewayConfig())
        first = ResumableJobs(native, Path(temp), allowed_roots=(root,))
        prepared = await first.call("job_prepare", {
            "command": "python3 -m compileall -q src/hermes_mcp_gateway",
            "workdir": root,
            "idempotency_key": "manual-smoke-20261024-" + "a" * 25,
        }, caller_id="gateway-native-smoke")
        credential = {"job_id": prepared["job_id"], "recovery_token": prepared["recovery_token"]}
        started = await first.call("job_start", credential, caller_id="gateway-native-smoke")
        assert started["state"] == "running", started["state"]
        # A second independent adapter simulates a new MCP client/session.
        recovered = ResumableJobs(native, Path(temp), allowed_roots=(root,))
        result = {"state": "running", "exit_code": None}
        for _ in range(12):
            result = await recovered.call("job_status", credential)
            if result["state"] != "running":
                break
            await asyncio.sleep(1)
        assert result["state"] == "succeeded", result["state"]
        assert result["exit_code"] == 0, result["exit_code"]
        print("Native recovery smoke passed; no secrets printed.")


if __name__ == "__main__":
    asyncio.run(main())
