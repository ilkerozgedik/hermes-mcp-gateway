"""Operator-only admission control; never execute jobs or expose credentials."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "job_admission.py"


class AdmissionControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gateway-admission-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "gateway-jobs.sqlite3"
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
        self.path.chmod(0o600)

    def call(self, action, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), action, "--database", str(self.path), *extra],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def test_status_is_read_only_and_missing_setting_means_accepting(self):
        before = self.path.read_bytes()
        result = self.call("status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["draining"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_drain_and_resume_preserve_job_rows(self):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO jobs VALUES('done','succeeded')")
        drained = self.call("drain", "--timeout", "0")
        self.assertEqual(drained.returncode, 0, drained.stderr)
        self.assertTrue(json.loads(drained.stdout)["draining"])
        self.assertEqual(json.loads(drained.stdout)["active_count"], 0)
        resumed = self.call("resume")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertFalse(json.loads(resumed.stdout)["draining"])
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT * FROM jobs").fetchall(), [("done", "succeeded")])
            self.assertEqual(db.execute("SELECT value FROM gateway_settings WHERE key='draining'").fetchone()[0], "0")

    def test_unknown_is_a_blocker_even_when_other_states_are_final(self):
        with sqlite3.connect(self.path) as db:
            db.executemany("INSERT INTO jobs VALUES(?,?)", [
                ("a", "unknown"), ("b", "starting"), ("c", "running"),
                ("d", "prepared"), ("e", "failed"),
            ])
        result = self.call("drain", "--timeout", "0")
        self.assertEqual(result.returncode, 3, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["draining"])
        self.assertEqual(payload["active_count"], 3)
        self.assertEqual(payload["states"]["unknown"], 1)
        self.assertNotIn("job_id", payload)

    def test_missing_database_is_not_created(self):
        self.path.unlink()
        result = self.call("drain", "--timeout", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.path.exists())

    def test_symlinked_database_is_rejected_without_changes(self):
        target = self.path.with_name("real.sqlite3")
        self.path.rename(target)
        self.path.symlink_to(target)
        before = target.read_bytes()
        result = self.call("drain", "--timeout", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_bytes(), before)

    def test_snapshot_is_consistent_private_and_never_overwrites(self):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO jobs VALUES('done','succeeded')")
        destination = self.path.with_name("before-deploy.sqlite3")
        result = self.call("snapshot", "--destination", str(destination))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
        with sqlite3.connect(destination) as db:
            self.assertEqual(db.execute("SELECT * FROM jobs").fetchall(), [("done", "succeeded")])
        self.assertNotEqual(self.call("snapshot", "--destination", str(destination)).returncode, 0)

    def test_invalid_timeout_cannot_change_admission(self):
        before = self.path.read_bytes()
        result = self.call("drain", "--timeout", "-1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
