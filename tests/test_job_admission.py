"""Operator-only admission control; never execute jobs or expose credentials."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "job_admission.py"
OWNER_A = "a" * 32
OWNER_B = "b" * 32


class AdmissionControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="gateway-admission-", dir=os.environ.get("TMPDIR")
        )
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "gateway-jobs.sqlite3"
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
        self.path.chmod(0o600)

    def call(self, action, *extra, owner: str | None = OWNER_A):
        command = [sys.executable, str(SCRIPT), action, "--database", str(self.path)]
        if action in ("drain", "resume") and owner is not None:
            command.extend(("--owner-token", owner))
        command.extend(extra)
        return subprocess.run(
            command, capture_output=True, text=True, timeout=5,
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
            self.assertIsNone(db.execute("SELECT value FROM gateway_settings WHERE key='drain_owner'").fetchone())

    def test_foreign_drain_and_resume_cannot_change_an_owned_pause(self):
        drained = self.call("drain", "--timeout", "0", owner=OWNER_A)
        self.assertEqual(drained.returncode, 0, drained.stderr)
        with sqlite3.connect(self.path) as db:
            owner_hash = hashlib.sha256(OWNER_A.encode("ascii")).hexdigest()
            stored_owner = db.execute(
                "SELECT value FROM gateway_settings WHERE key='drain_owner'"
            ).fetchone()[0]
            self.assertEqual(stored_owner, owner_hash)
        self.assertEqual(self.call("drain", "--timeout", "0", owner=OWNER_A).returncode, 0)

        foreign_drain = self.call("drain", "--timeout", "0", owner=OWNER_B)
        self.assertEqual(foreign_drain.returncode, 2, foreign_drain.stderr)
        foreign_resume = self.call("resume", owner=OWNER_B)
        self.assertEqual(foreign_resume.returncode, 2, foreign_resume.stderr)
        status = self.call("status")
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertTrue(json.loads(status.stdout)["draining"])
        owner_hash = hashlib.sha256(OWNER_A.encode("ascii")).hexdigest()
        self.assertNotIn(OWNER_A, status.stdout)
        self.assertNotIn(owner_hash, status.stdout)

        resumed = self.call("resume", owner=OWNER_A)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertFalse(json.loads(resumed.stdout)["draining"])
        self.assertEqual(self.call("resume", owner=OWNER_B).returncode, 0)

    def test_legacy_pause_without_owner_is_never_cleared(self):
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE gateway_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("INSERT INTO gateway_settings VALUES('draining','1')")
        result = self.call("resume")
        self.assertEqual(result.returncode, 2, result.stderr)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT value FROM gateway_settings WHERE key='draining'").fetchone()[0], "1")

    def test_owner_token_is_required_and_validated(self):
        missing = self.call("drain", "--timeout", "0", owner=None)
        self.assertNotEqual(missing.returncode, 0)
        for token in ("short", "x" * 129, "a" * 31 + " "):
            result = self.call("drain", "--timeout", "0", owner=token)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse(json.loads(self.call("status").stdout)["draining"])

    def test_concurrent_owners_have_exactly_one_claim_winner(self):
        commands = [
            [sys.executable, str(SCRIPT), "drain", "--database", str(self.path),
             "--owner-token", owner, "--timeout", "0"]
            for owner in (OWNER_A, OWNER_B)
        ]
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                     for command in commands]
        results = [process.communicate(timeout=5) + (process.returncode,) for process in processes]
        self.assertEqual(sorted(result[2] for result in results), [0, 2], results)
        status = self.call("status")
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertTrue(json.loads(status.stdout)["draining"])

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

    def test_symlinked_ancestor_is_rejected(self):
        real_dir = Path(self.temp.name) / "real"
        real_dir.mkdir()
        target = real_dir / "source.sqlite3"
        target.write_bytes(self.path.read_bytes())
        target.chmod(0o600)
        alias = Path(self.temp.name) / "alias"
        alias.symlink_to(real_dir, target_is_directory=True)
        original = self.path
        self.path = alias / target.name
        result = self.call("drain", "--timeout", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_bytes(), original.read_bytes())

    def test_snapshot_rejects_writable_source_and_destination_ancestors_without_changes(self):
        unsafe_source_parent = Path(self.temp.name) / "unsafe-source"
        unsafe_source_parent.mkdir()
        unsafe_source_parent.chmod(0o777)
        source = unsafe_source_parent / "source.sqlite3"
        source.write_bytes(self.path.read_bytes())
        source.chmod(0o600)
        before = source.read_bytes()
        safe_copy = Path(self.temp.name) / "source-copy.sqlite3"
        rejected_source = subprocess.run(
            [sys.executable, str(SCRIPT), "snapshot", "--database", str(source),
             "--destination", str(safe_copy)],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(rejected_source.returncode, 2)
        self.assertEqual(source.read_bytes(), before)
        self.assertFalse(safe_copy.exists())

        unsafe_destination_parent = Path(self.temp.name) / "unsafe-destination"
        unsafe_destination_parent.mkdir()
        unsafe_destination_parent.chmod(0o777)
        unsafe_copy = unsafe_destination_parent / "destination-copy.sqlite3"
        rejected_destination = self.call("snapshot", "--destination", str(unsafe_copy))
        self.assertEqual(rejected_destination.returncode, 2)
        self.assertFalse(unsafe_copy.exists())
        unsafe_source_parent.chmod(0o700)
        unsafe_destination_parent.chmod(0o700)

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

    def test_snapshot_includes_committed_wal_data(self):
        source = sqlite3.connect(self.path)
        try:
            source.execute("PRAGMA journal_mode=WAL")
            source.execute("INSERT INTO jobs VALUES('wal-row','succeeded')")
            source.commit()
            self.assertTrue(Path(str(self.path) + "-wal").exists())
            destination = self.path.with_name("wal-snapshot.sqlite3")
            result = self.call("snapshot", "--destination", str(destination))
            self.assertEqual(result.returncode, 0, result.stderr)
            with sqlite3.connect(destination) as db:
                self.assertEqual(db.execute("SELECT * FROM jobs").fetchall(), [("wal-row", "succeeded")])
        finally:
            source.close()

    def test_invalid_timeout_cannot_change_admission(self):
        before = self.path.read_bytes()
        result = self.call("drain", "--timeout", "-1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
