import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_mcp_gateway.startup import (
    STARTUP_PATHS,
    StartupContext,
    startup_tool_schema,
)


class StartupContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.paths = tuple(root / name for name in ("a.md", "b.md"))
        self.paths[0].write_text("alpha\n", encoding="utf-8")
        self.paths[1].write_text("beta\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_startup_paths_exclude_retired_ponytail_skill(self):
        self.assertNotIn(Path("/home/hermes/.hermes/skills/ponytail/SKILL.md"), STARTUP_PATHS)

    def test_schema_accepts_no_path_or_other_arguments(self):
        tool = startup_tool_schema()
        self.assertEqual(tool.name, "startup_context")
        self.assertEqual(tool.input_schema.get("properties"), {})
        self.assertEqual(tool.input_schema.get("additionalProperties"), False)

    def test_reads_only_fixed_files_with_metadata(self):
        startup = StartupContext(paths=self.paths, max_total_bytes=1024)
        payload = startup.read()
        self.assertEqual(
            [item["path"] for item in payload["files"]], [str(p) for p in self.paths]
        )
        self.assertEqual(
            [item["content"] for item in payload["files"]], ["alpha\n", "beta\n"]
        )
        self.assertTrue(all(len(item["sha256"]) == 64 for item in payload["files"]))
        self.assertTrue(
            all(isinstance(item["mtime_ns"], int) for item in payload["files"])
        )

    def test_rejects_symlink_even_when_target_is_regular_file(self):
        link = self.paths[0].with_name("link.md")
        link.symlink_to(self.paths[0])
        startup = StartupContext(paths=(link,), max_total_bytes=1024)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            startup.read()

    def test_missing_required_file_is_an_error(self):
        missing = self.paths[0].with_name("missing.md")
        startup = StartupContext(paths=(missing,), max_total_bytes=1024)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            startup.read()

    def test_total_output_is_bounded_before_reading(self):
        self.paths[0].write_text("x" * 20, encoding="utf-8")
        startup = StartupContext(paths=self.paths, max_total_bytes=8)
        with self.assertRaisesRegex(RuntimeError, "size limit"):
            startup.read()

    def test_readiness_checks_real_readability(self):
        startup = StartupContext(paths=self.paths)
        with patch("hermes_mcp_gateway.startup.os.open", side_effect=PermissionError("denied")):
            self.assertFalse(startup.check())
            with self.assertRaises(PermissionError):
                startup.read()

    def test_readiness_rejects_invalid_utf8(self):
        self.paths[0].write_bytes(b"\xff")
        self.assertFalse(StartupContext(paths=self.paths).check())

    def test_serialized_limit_includes_metadata_and_escaping(self):
        self.paths[0].write_text("x" * 65_536, encoding="utf-8")
        startup = StartupContext(paths=(self.paths[0],))
        self.assertFalse(startup.check())
        with self.assertRaisesRegex(RuntimeError, "serialized output"):
            startup.read()
        self.paths[0].write_text('"' * 40_000, encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "serialized output"):
            startup.read()

    def test_open_rejects_replacement_symlink_after_stat(self):
        startup = StartupContext(paths=(self.paths[0],))
        original = startup._validated_stat
        def replace_after_stat(path):
            info = original(path)
            path.unlink()
            path.symlink_to(self.paths[1])
            return info
        with (
            patch.object(startup, "_validated_stat", side_effect=replace_after_stat),
            self.assertRaises(OSError),
        ):
            startup.read()


class StartupTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_envelope_overflow_is_an_explicit_error_not_broken_json(self):
        from hermes_mcp_gateway.config import GatewayConfig
        from hermes_mcp_gateway.server import Gateway, build_catalog

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "startup.md"
            path.write_text("x" * 65_536, encoding="utf-8")
            gateway = Gateway(GatewayConfig())
            gateway.startup = StartupContext(paths=(path,))
            gateway.catalog = build_catalog([], [], [startup_tool_schema()])
            result = await gateway.call("startup_context", {})
            self.assertTrue(result.is_error)
            self.assertIn("serialized output", str(result.content))


if __name__ == "__main__":
    unittest.main()
