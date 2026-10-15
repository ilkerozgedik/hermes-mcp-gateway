"""Exercise the official Tunnel Client against a real Gateway, without hosted keys.

This uses the released client's local control plane, not simulated MCP responses.
It opens an isolated legacy browser tab and runs a harmless foreground command.
It does not prove ChatGPT workspace/connector permissions; test those in ChatGPT.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

import httpx

from e2e_native import PROTOCOL, text, unpack


async def exercise(url: str, report: dict) -> None:
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        async def rpc(method, params, *, headers=None):
            request_headers = {"Accept": "application/json, text/event-stream"}
            if headers is None:
                request_headers.update({"MCP-Protocol-Version": PROTOCOL,
                                        "Mcp-Method": method})
                if method == "tools/call":
                    request_headers["Mcp-Name"] = params["name"]
                params = {**params, "_meta": {
                    "io.modelcontextprotocol/protocolVersion": PROTOCOL,
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "gateway-tunnel-acceptance", "version": "1"},
                    "io.modelcontextprotocol/clientCapabilities": {},
                }}
            else:
                request_headers.update(headers)
            response = await client.post(url, headers=request_headers, json={
                "jsonrpc": "2.0", "id": uuid.uuid4().hex,
                "method": method, "params": params,
            })
            return unpack(response), response

        async def call(name, arguments, *, headers=None, session_token=None):
            bridge_arguments = {"calls": [{"name": name, "arguments": arguments}]}
            if session_token is not None:
                bridge_arguments["session_token"] = session_token
            result, _ = await rpc("tools/call", {"name": "tool_call", "arguments": bridge_arguments},
                                  headers=headers)
            assert not result.get("isError"), (name, text(result))
            return result

        listing, _ = await rpc("tools/list", {})
        expected = {"tool_search", "tool_describe", "tool_call", "startup_context",
                    "memory_context", "skills_list"}
        assert {t["name"] for t in listing["tools"]} == expected
        report["checks"]["modern_tools_list"] = sorted(expected)
        search, _ = await rpc("tools/call", {"name": "tool_search", "arguments": {
            "queries": ["execute_code", "browser_navigate", "terminal"], "limit": 3,
        }})
        assert not search.get("isError"), text(search)
        assert "execute_code" in text(search)
        report["checks"]["modern_discovery"] = True
        described, _ = await rpc("tools/call", {"name": "tool_describe", "arguments": {
            "names": ["execute_code", "terminal", "browser_navigate"],
        }})
        assert not described.get("isError"), text(described)
        report["checks"]["modern_describe"] = True
        read = await call("read_file", {"path": str(Path(__file__).resolve()), "limit": 2})
        assert "official Tunnel Client" in text(read)
        report["checks"]["modern_native_file"] = True
        executed = await call("execute_code", {
            "code": "from hermes_tools import terminal\nprint(terminal('printf tunnel-native-ok'))",
        })
        assert json.loads(text(executed)).get("status") == "success", text(executed)
        assert "tunnel-native-ok" in text(executed)
        report["checks"]["modern_native_execution"] = True
        opened = await call("session_open", {})
        token = json.loads(text(opened))["session_token"]
        assert isinstance(token, str) and len(token) >= 32
        try:
            await call("terminal", {"command": "export MCP_LEASE_MARKER=tunnel-lease-ok"},
                       session_token=token)
            terminal = await call("terminal", {"command": "printf '%s' \"$MCP_LEASE_MARKER\""},
                                  session_token=token)
            assert "tunnel-lease-ok" in text(terminal), text(terminal)
            report["checks"]["modern_terminal_continuity"] = True
            browser = await call("browser_navigate", {"url": "https://example.com"},
                                 session_token=token)
            browser_payload = json.loads(text(browser))
            assert browser_payload.get("success") is True, text(browser)
            assert browser_payload.get("url", "").rstrip("/") == "https://example.com", text(browser)
            snapshot = await call("browser_snapshot", {}, session_token=token)
            snapshot_payload = json.loads(text(snapshot))
            assert snapshot_payload.get("success") is True and snapshot_payload.get("snapshot"), text(snapshot)
            report["checks"]["modern_browser_continuity"] = True
        finally:
            await call("session_close", {}, session_token=token)
        denied, _ = await rpc("tools/call", {"name": "tool_call", "arguments": {
            "session_token": token,
            "calls": [{"name": "terminal", "arguments": {"command": "printf must-not-run"}}],
        }})
        assert denied.get("isError"), "closed capability was not revoked"
        report["checks"]["modern_session_revocation"] = True
        initialized, response = await rpc("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "gateway-tunnel-legacy-acceptance", "version": "1"},
        }, headers={})
        session = response.headers.get("mcp-session-id")
        assert session, ("legacy session header missing", dict(response.headers))
        protocol = initialized["protocolVersion"]
        assert protocol != PROTOCOL, protocol
        legacy = {"Mcp-Session-Id": session, "MCP-Protocol-Version": protocol}
        try:
            notification = await client.post(url, headers={
                "Accept": "application/json, text/event-stream", **legacy,
            }, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
            notification.raise_for_status()
            terminal = await call("terminal", {"command": "printf tunnel-terminal-ok"},
                                  headers=legacy)
            assert "tunnel-terminal-ok" in text(terminal), text(terminal)
            report["checks"]["legacy_native_terminal"] = True
            browser = await call("browser_navigate", {"url": "https://example.com"},
                                 headers=legacy)
            browser_payload = json.loads(text(browser))
            assert browser_payload.get("success") is True, text(browser)
            assert browser_payload.get("url", "").rstrip("/") == "https://example.com", text(browser)
            assert browser_payload.get("snapshot"), text(browser)
            snapshot = await call("browser_snapshot", {}, headers=legacy)
            snapshot_payload = json.loads(text(snapshot))
            assert snapshot_payload.get("success") is True, text(snapshot)
            assert snapshot_payload.get("snapshot"), text(snapshot)
            report["checks"]["legacy_browser_navigate_snapshot"] = True
            report["checks"]["legacy_initialize"] = {"protocol": protocol}
        finally:
            # The local tunnel ingress is POST-only. Closing its listener below
            # ends this isolated test; do not DELETE the real Gateway endpoint.
            pass


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3060/mcp")
    parser.add_argument("--tunnel-client", default="/usr/local/bin/tunnel-client")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = {"target": args.url, "checks": {}, "status": "failed"}
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="gateway-tunnel-", dir=os.environ["TMPDIR"]) as tmp:
            url_file = Path(tmp) / "proxy.json"
            log_path = Path(tmp) / "proxy.log"
            with log_path.open("w+") as log:
                process = subprocess.Popen([
                    args.tunnel_client, "dev", "proxy", "--mcp-server-url", args.url,
                    "--url-file", str(url_file), "--duration", "240s",
                    "--readiness-timeout", "45s", "--response-timeout", "120s",
                ], stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 55
                while not url_file.exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        log.seek(0)
                        raise RuntimeError("Local tunnel did not start: " + log.read()[-4000:])
                    await asyncio.sleep(0.1)
                connection = json.loads(url_file.read_text())
                report["local_proxy"] = connection
                await exercise(connection["mcp_url"], report)
                report["status"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "checks": report["checks"],
                          "report": args.report}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
