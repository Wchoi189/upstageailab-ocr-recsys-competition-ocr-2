
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from AgentQMS.tools.compliance.validate_artifacts import ArtifactValidator

class TestDualVersionValidation(unittest.TestCase):
    def setUp(self):
        # Create temp dir INSIDE project structure to satisfy ensure_within_project
        self.base_temp = Path("AgentQMS/tests/compliance/temp_artifacts")
        self.base_temp.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = TemporaryDirectory(dir=self.base_temp)
        self.root = Path(self.tmp_dir.name).resolve()

        self.validator = ArtifactValidator(artifacts_root=self.root)

        # Mock rules to ensure stability for tests
        self.validator.rules = {
            "frontmatter": {
                "required_fields": ["title", "date", "type", "category", "status", "version", "ads_version"],
                "valid_statuses": ["active"],
                "valid_categories": ["development"]
            },
            "artifact_types": {
                "test": {"prefix": "test_", "directory": "tests/", "frontmatter_type": "test"}
            }
        }
        self.validator._init_frontmatter_rules()
        self.validator._init_type_mappings()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_legacy_artifact(self):
        # V1 artifact (no ads_version) - should NOT require ads_version
        path = self.root / "implementation_plan_legacy_test.md"
        with open(path, "w") as f:
            f.write("---\n")
            f.write("title: Test\n")
            f.write("date: 2023-01-01 00:00 (KST)\n")
            f.write("type: implementation_plan\n")
            f.write("category: development\n")
            f.write("status: active\n")
            f.write("version: '1.0'\n")
            f.write("---\nContent")

        result = {"errors": []}
        self.validator._validate_file_frontmatter(path, strict_mode=True, result=result)
        # Should be valid because ads_version is not required for legacy
        errors = result.get("errors", [])
        self.assertEqual(len(errors), 0, f"Legacy artifact failed validation: {errors}")

    def test_modern_artifact(self):
        # V2 artifact (has ads_version) - SHOULD require ads_version (which it has)
        path = self.root / "implementation_plan_modern_test.md"
        with open(path, "w") as f:
            f.write("---\n")
            f.write("ads_version: '2.0'\n")
            f.write("title: Test\n")
            f.write("date: 2023-01-01 00:00 (KST)\n")
            f.write("type: implementation_plan\n")
            f.write("category: development\n")
            f.write("status: active\n")
            f.write("version: '1.0'\n")
            f.write("---\nContent")

        result = {"errors": []}
        self.validator._validate_file_frontmatter(path, strict_mode=True, result=result)
        errors = result.get("errors", [])
        self.assertEqual(len(errors), 0, f"Modern artifact failed validaton: {errors}")

    def test_modern_artifact_missing_field(self):
        # V2 artifact missing required field
        path = self.root / "implementation_plan_broken_modern.md"
        with open(path, "w") as f:
            f.write("---\n")
            f.write("ads_version: '2.0'\n")
            # Missing 'status'
            f.write("title: Test\n")
            f.write("date: 2023-01-01 00:00 (KST)\n")
            f.write("type: implementation_plan\n")
            f.write("category: development\n")
            f.write("version: '1.0'\n")
            f.write("---\nContent")


        result = {"errors": []}
        self.validator._validate_file_frontmatter(path, strict_mode=True, result=result)
        errors = result.get("errors", [])
        self.assertTrue(any("Missing required frontmatter field" in e for e in errors), "Should fail validation")

if __name__ == "__main__":
    unittest.main()
