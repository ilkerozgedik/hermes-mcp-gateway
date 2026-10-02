from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hermes_mcp_gateway.lsp import LSP_TOOLS, LSPAdapter


class FakeClient:
    def __init__(self):
        self.requests = []
        self.opened = []
        self._docs = {}

    async def open_file(self, path: str, *, language_id: str = "plaintext") -> int:
        self.opened.append((path, language_id))
        self._docs[path] = SimpleNamespace(seed_seen=True)
        return len(self.opened) - 1

    async def save_file(self, path: str) -> None:
        return None

    async def wait_for_diagnostics(self, path, version, *, mode="document", timeout=None):
        return True

    def diagnostics_for(self, path, *, fresh_only=False):
        return [{"message": "bad", "severity": 1, "range": {"start": {"line": 2, "character": 3}, "end": {"line": 2, "character": 4}}}]

    async def _send_request_with_retry(self, method, params, *, timeout):
        self.requests.append((method, params))
        if method == "textDocument/hover":
            return {"contents": {"kind": "markdown", "value": "number"}}
        if method == "textDocument/definition":
            return [{"uri": Path(params["textDocument"]["uri"][7:]).as_uri(), "range": {"start": {"line": 1, "character": 2}, "end": {"line": 1, "character": 5}}}]
        if method == "textDocument/references":
            return []
        if method == "textDocument/documentSymbol":
            return [{"name": "thing", "kind": 12, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}}, "selectionRange": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}}}]
        raise AssertionError(method)


class FakeLoop:
    def run(self, coro, *, timeout=None):
        import asyncio
        return asyncio.run(coro)


class FakeService:
    def __init__(self, client):
        self.client = client
        self._loop = FakeLoop()
        self._install_strategy = "manual"
        self._init_overrides = {}
        self.seed_first_push = False
        self.shutdown_called = False

    async def _get_or_spawn(self, file_path):
        return self.client

    def _server_for(self, file_path):
        return SimpleNamespace(seed_first_push=self.seed_first_push, language_id="")

    def get_status(self):
        return {"enabled": True, "clients": []}

    def shutdown(self):
        self.shutdown_called = True


class LSPAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.file = self.root / "a.py"
        self.file.write_text("x = 1\n")
        self.client = FakeClient()
        self.service = FakeService(self.client)
        self.adapter = LSPAdapter(allowed_roots=(str(self.root),), service=self.service)

    def tearDown(self):
        self.adapter.close()
        self.tmp.cleanup()

    def test_exact_read_only_surface(self):
        self.assertEqual(
            set(LSP_TOOLS),
            {
                "lsp_diagnostics",
                "lsp_hover",
                "lsp_definition",
                "lsp_references",
                "lsp_document_symbols",
                "lsp_status",
            },
        )
        self.assertFalse(any("rename" in name or "format" in name or "action" in name for name in LSP_TOOLS))

    def test_rejects_path_outside_allowed_root_and_symlink_escape(self):
        other = Path(self.tmp.name).parent / "outside-lsp.py"
        other.write_text("x=1\n")
        self.addCleanup(lambda: other.unlink(missing_ok=True))
        with self.assertRaises(PermissionError):
            self.adapter.call("lsp_diagnostics", {"file": str(other)})
        link = self.root / "escape.py"
        link.symlink_to(other)
        with self.assertRaises(PermissionError):
            self.adapter.call("lsp_diagnostics", {"file": str(link)})

    def test_public_coordinates_are_one_based(self):
        result = self.adapter.call("lsp_hover", {"file": str(self.file), "line": 4, "column": 7})
        self.assertEqual(result["result"]["contents"]["value"], "number")
        method, params = self.client.requests[-1]
        self.assertEqual(method, "textDocument/hover")
        self.assertEqual(params["position"], {"line": 3, "character": 6})

    def test_routes_standard_read_only_methods(self):
        self.adapter.call("lsp_definition", {"file": str(self.file), "line": 1, "column": 1})
        self.adapter.call("lsp_references", {"file": str(self.file), "line": 1, "column": 1})
        self.adapter.call("lsp_document_symbols", {"file": str(self.file)})
        self.assertEqual(
            [m for m, _ in self.client.requests],
            ["textDocument/definition", "textDocument/references", "textDocument/documentSymbol"],
        )

    def test_diagnostics_are_full_and_converted_to_one_based(self):
        result = self.adapter.call("lsp_diagnostics", {"file": str(self.file)})
        self.assertEqual(result["diagnostics"][0]["range"]["start"], {"line": 3, "column": 4})

    def test_service_never_auto_installs(self):
        self.assertEqual(self.service._install_strategy, "manual")

    def test_diagnostics_timeout_is_not_reported_as_clean(self):
        async def no_verdict(*_args, **_kwargs):
            return False
        self.client.wait_for_diagnostics = no_verdict
        with self.assertRaisesRegex(RuntimeError, "fresh diagnostics verdict"):
            self.adapter.call("lsp_diagnostics", {"file": str(self.file)})

    def test_seeded_server_gets_second_document_version_for_fresh_diagnostics(self):
        self.service.seed_first_push = True
        self.adapter.call("lsp_diagnostics", {"file": str(self.file)})
        self.assertEqual(len(self.client.opened), 2)

    def test_close_shuts_down_service(self):
        self.adapter.close()
        self.assertTrue(self.service.shutdown_called)


if __name__ == "__main__":
    unittest.main()
