"""Phase 1 smoke tests."""

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from unittest import TestCase


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScaffoldImportTests(TestCase):
    """Validate that the initial packages are import-safe."""

    def test_project_packages_import(self) -> None:
        """Local packages should import without optional runtime setup."""

        self.assertIsNotNone(import_module("agent"))
        self.assertIsNotNone(import_module("tools"))
        self.assertIsNotNone(import_module("app"))

    def test_tools_import_first_from_an_external_working_directory(self) -> None:
        """Package imports must not depend on import order or current directory."""

        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from tools import load_case_metadata; "
                    "item = load_case_metadata('case_001'); "
                    "assert item.workload_name == 'payment-api'"
                ),
            ],
            cwd=PROJECT_ROOT.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
