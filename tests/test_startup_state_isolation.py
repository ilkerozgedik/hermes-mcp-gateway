"""Startup tests must never initialize the operator's real job database."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_mcp_gateway.jobs import ResumableJobs


class StartupStateIsolationTests(unittest.TestCase):
    def test_startup_cases_use_temporary_job_state(self):
        scratch = Path(tempfile.gettempdir()).resolve()
        directories = []

        def guarded_constructor(hermes, directory, **kwargs):
            directory = Path(directory).resolve()
            self.assertIn(scratch, directory.parents)
            self.assertNotEqual(directory, Path("/home/hermes/.hermes"))
            directories.append(directory)
            return ResumableJobs(hermes, directory, **kwargs)

        suite = unittest.defaultTestLoader.loadTestsFromNames([
            "test_gateway.WebReadinessProbeTests.test_start_does_not_duplicate_failed_web_probe",
            "test_image_fallback.DeferredImageTests.test_cooldown_does_not_break_startup_or_hide_image_tool",
        ])
        result = unittest.TestResult()
        with patch("hermes_mcp_gateway.server.ResumableJobs", side_effect=guarded_constructor):
            suite.run(result)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertEqual(len(directories), 2)
        self.assertTrue(all(not directory.exists() for directory in directories))
