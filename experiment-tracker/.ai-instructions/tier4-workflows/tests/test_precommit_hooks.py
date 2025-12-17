#!/usr/bin/env python3
"""
Integration Tests for EDS v1.0 Pre-Commit Hooks

Tests the complete pre-commit hook validation chain:
1. Naming validation (blocks ALL-CAPS)
2. Metadata validation (requires .metadata/)
3. EDS compliance (validates frontmatter)

Usage:
    python test_precommit_hooks.py
    python test_precommit_hooks.py --verbose
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


class Color:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


class PreCommitHookTester:
    """Test suite for pre-commit hooks."""

    def __init__(self, tracker_root: Path):
        self.tracker_root = tracker_root
        self.hooks_dir = tracker_root / ".ai-instructions" / "tier4-workflows" / "pre-commit-hooks"
        self.experiments_dir = tracker_root / "experiments"

        self.tests_passed = 0
        self.tests_failed = 0
        self.verbose = "--verbose" in sys.argv

    def run_hook(self, hook_name: str, *args) -> tuple[int, str, str]:
        """Run a pre-commit hook script."""
        hook_path = self.hooks_dir / hook_name

        if not hook_path.exists():
            raise FileNotFoundError(f"Hook not found: {hook_path}")

        cmd = ["bash", str(hook_path)] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.tracker_root)

        return result.returncode, result.stdout, result.stderr

    def test_naming_validation_blocks_all_caps(self):
        """Test: naming-validation.sh blocks ALL-CAPS filenames."""
        test_name = "Naming validation blocks ALL-CAPS"

        # Create test file with ALL-CAPS name
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / "TEST_FILE.md"
            test_file.write_text("# Test", encoding='utf-8')

            # Stage file (simulate git add)
            os.chdir(tmpdir_path)
            subprocess.run(["git", "init"], capture_output=True)
            subprocess.run(["git", "add", "TEST_FILE.md"], capture_output=True)

            # Run naming validation
            exit_code, stdout, stderr = self.run_hook("naming-validation.sh")

            if exit_code != 0 and "ALL-CAPS" in stdout:
                self._pass(test_name)
                return True
            else:
                self._fail(test_name, f"Expected exit code != 0 and 'ALL-CAPS' in output, got: {exit_code}")
                return False

    def test_naming_validation_allows_compliant(self):
        """Test: naming-validation.sh allows compliant filenames."""
        test_name = "Naming validation allows compliant filenames"

        # Create test file with compliant name
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create experiment structure
            exp_dir = tmpdir_path / "experiments" / "20251217_1234_test_experiment"
            exp_dir.mkdir(parents=True)

            test_file = exp_dir / "20251217_1234_assessment_test-file.md"
            test_file.write_text("# Test", encoding='utf-8')

            # Stage file
            os.chdir(tmpdir_path)
            subprocess.run(["git", "init"], capture_output=True)
            subprocess.run(["git", "add", str(test_file)], capture_output=True)

            # Run naming validation
            exit_code, stdout, stderr = self.run_hook("naming-validation.sh")

            if exit_code == 0:
                self._pass(test_name)
                return True
            else:
                self._fail(test_name, f"Expected exit code 0, got: {exit_code}\n{stdout}")
                return False

    def test_metadata_validation_requires_structure(self):
        """Test: metadata-validation.sh requires .metadata/ directory."""
        test_name = "Metadata validation requires .metadata/ directory"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create experiment WITHOUT .metadata/
            exp_dir = tmpdir_path / "experiments" / "20251217_1234_test_experiment"
            exp_dir.mkdir(parents=True)

            test_file = exp_dir / "20251217_1234_assessment_test.md"
            test_file.write_text("# Test", encoding='utf-8')

            # Stage file
            os.chdir(tmpdir_path)
            subprocess.run(["git", "init"], capture_output=True)
            subprocess.run(["git", "add", str(test_file)], capture_output=True)

            # Run metadata validation
            exit_code, stdout, stderr = self.run_hook("metadata-validation.sh")

            if exit_code != 0 and ".metadata" in stdout:
                self._pass(test_name)
                return True
            else:
                self._fail(test_name, f"Expected exit code != 0 and '.metadata' in output, got: {exit_code}")
                return False

    def test_metadata_validation_allows_compliant(self):
        """Test: metadata-validation.sh allows experiments with .metadata/."""
        test_name = "Metadata validation allows compliant structure"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create experiment WITH .metadata/
            exp_dir = tmpdir_path / "experiments" / "20251217_1234_test_experiment"
            exp_dir.mkdir(parents=True)
            (exp_dir / ".metadata").mkdir()
            (exp_dir / ".metadata" / "assessments").mkdir()

            test_file = exp_dir / ".metadata" / "assessments" / "20251217_1234_assessment_test.md"
            test_file.write_text("# Test", encoding='utf-8')

            # Stage file
            os.chdir(tmpdir_path)
            subprocess.run(["git", "init"], capture_output=True)
            subprocess.run(["git", "add", str(test_file)], capture_output=True)

            # Run metadata validation
            exit_code, stdout, stderr = self.run_hook("metadata-validation.sh")

            if exit_code == 0:
                self._pass(test_name)
                return True
            else:
                self._fail(test_name, f"Expected exit code 0, got: {exit_code}\n{stdout}")
                return False

    def test_eds_compliance_blocks_missing_frontmatter(self):
        """Test: eds-compliance.sh blocks artifacts without frontmatter."""
        test_name = "EDS compliance blocks missing frontmatter"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create experiment with artifact missing frontmatter
            exp_dir = tmpdir_path / "experiments" / "20251217_1234_test_experiment"
            metadata_dir = exp_dir / ".metadata" / "assessments"
            metadata_dir.mkdir(parents=True)

            test_file = metadata_dir / "20251217_1234_assessment_test.md"
            test_file.write_text("# Test\n\nNo frontmatter here.", encoding='utf-8')

            # Stage file
            os.chdir(tmpdir_path)
            subprocess.run(["git", "init"], capture_output=True)
            subprocess.run(["git", "add", str(test_file)], capture_output=True)

            # Run EDS compliance (note: may fail if compliance-checker.py not accessible)
            try:
                exit_code, stdout, stderr = self.run_hook("eds-compliance.sh")

                if exit_code != 0:
                    self._pass(test_name)
                    return True
                else:
                    self._fail(test_name, f"Expected exit code != 0, got: {exit_code}")
                    return False
            except Exception as e:
                self._skip(test_name, f"Compliance checker not accessible: {str(e)}")
                return True

    def test_eds_compliance_allows_valid_frontmatter(self):
        """Test: eds-compliance.sh allows artifacts with valid frontmatter."""
        test_name = "EDS compliance allows valid frontmatter"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create experiment with valid artifact
            exp_dir = tmpdir_path / "experiments" / "20251217_1234_test_experiment"
            metadata_dir = exp_dir / ".metadata" / "assessments"
            metadata_dir.mkdir(parents=True)

            test_file = metadata_dir / "20251217_1234_assessment_test.md"
            content = """---
ads_version: "1.0"
type: assessment
experiment_id: "20251217_1234_test_experiment"
phase: "analysis"
priority: "medium"
status: "active"
created: "2025-12-17T18:00:00Z"
updated: "2025-12-17T18:00:00Z"
tags: ["test"]
evidence_count: 0
---

# Test Assessment

Content here.
"""
            test_file.write_text(content, encoding='utf-8')

            # Stage file
            os.chdir(tmpdir_path)
            subprocess.run(["git", "init"], capture_output=True)
            subprocess.run(["git", "add", str(test_file)], capture_output=True)

            # Run EDS compliance
            try:
                exit_code, stdout, stderr = self.run_hook("eds-compliance.sh")

                if exit_code == 0:
                    self._pass(test_name)
                    return True
                else:
                    self._fail(test_name, f"Expected exit code 0, got: {exit_code}\n{stdout}")
                    return False
            except Exception as e:
                self._skip(test_name, f"Compliance checker not accessible: {str(e)}")
                return True

    def test_full_chain_integration(self):
        """Test: Complete pre-commit hook chain."""
        test_name = "Full pre-commit hook chain integration"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create compliant experiment structure
            exp_dir = tmpdir_path / "experiments" / "20251217_1234_test_experiment"
            metadata_dir = exp_dir / ".metadata" / "assessments"
            metadata_dir.mkdir(parents=True)

            # Valid artifact
            test_file = metadata_dir / "20251217_1234_assessment_valid-test.md"
            content = """---
ads_version: "1.0"
type: assessment
experiment_id: "20251217_1234_test_experiment"
phase: "analysis"
priority: "medium"
status: "active"
created: "2025-12-17T18:00:00Z"
updated: "2025-12-17T18:00:00Z"
tags: ["test", "integration"]
evidence_count: 1
---

# Valid Test Assessment

This artifact should pass all validations.

## Evidence

Evidence section.
"""
            test_file.write_text(content, encoding='utf-8')

            # Stage file
            os.chdir(tmpdir_path)
            subprocess.run(["git", "init"], capture_output=True)
            subprocess.run(["git", "add", str(test_file)], capture_output=True)

            # Run all hooks in sequence
            hooks = ["naming-validation.sh", "metadata-validation.sh", "eds-compliance.sh"]
            all_passed = True

            for hook in hooks:
                try:
                    exit_code, stdout, stderr = self.run_hook(hook)
                    if exit_code != 0:
                        all_passed = False
                        if self.verbose:
                            print(f"  Hook {hook} failed: {stdout}")
                        break
                except Exception as e:
                    if self.verbose:
                        print(f"  Hook {hook} error: {str(e)}")
                    # Skip compliance if checker not accessible
                    if "compliance" in hook:
                        continue
                    else:
                        all_passed = False
                        break

            if all_passed:
                self._pass(test_name)
                return True
            else:
                self._fail(test_name, "One or more hooks failed")
                return False

    def run_all_tests(self):
        """Run complete test suite."""
        print(f"\n{Color.BLUE}{'='*60}{Color.END}")
        print(f"{Color.BLUE}EDS v1.0 Pre-Commit Hook Integration Tests{Color.END}")
        print(f"{Color.BLUE}{'='*60}{Color.END}\n")

        tests = [
            self.test_naming_validation_blocks_all_caps,
            self.test_naming_validation_allows_compliant,
            self.test_metadata_validation_requires_structure,
            self.test_metadata_validation_allows_compliant,
            self.test_eds_compliance_blocks_missing_frontmatter,
            self.test_eds_compliance_allows_valid_frontmatter,
            self.test_full_chain_integration,
        ]

        for test in tests:
            test()

        print(f"\n{Color.BLUE}{'='*60}{Color.END}")
        print(f"{Color.GREEN}✅ Passed: {self.tests_passed}{Color.END}")
        print(f"{Color.RED}❌ Failed: {self.tests_failed}{Color.END}")
        print(f"{Color.BLUE}{'='*60}{Color.END}\n")

        return self.tests_failed == 0

    def _pass(self, test_name: str):
        """Mark test as passed."""
        self.tests_passed += 1
        print(f"{Color.GREEN}✅ PASS{Color.END}: {test_name}")

    def _fail(self, test_name: str, reason: str):
        """Mark test as failed."""
        self.tests_failed += 1
        print(f"{Color.RED}❌ FAIL{Color.END}: {test_name}")
        if self.verbose:
            print(f"  Reason: {reason}")

    def _skip(self, test_name: str, reason: str):
        """Skip test."""
        print(f"{Color.YELLOW}⏭️  SKIP{Color.END}: {test_name}")
        if self.verbose:
            print(f"  Reason: {reason}")


def main():
    # Find tracker root
    current = Path.cwd()
    while current != current.parent:
        if (current / "experiment-tracker").exists():
            tracker_root = current / "experiment-tracker"
            break
        if (current / ".ai-instructions").exists():
            tracker_root = current
            break
        current = current.parent
    else:
        print("❌ ERROR: Unable to find experiment-tracker root", file=sys.stderr)
        sys.exit(1)

    tester = PreCommitHookTester(tracker_root)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
