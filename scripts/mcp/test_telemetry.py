
import unittest
from AgentQMS.middleware.policies import ComplianceInterceptor, PolicyViolation, StandardsInterceptor, FileOperationInterceptor
from AgentQMS.middleware.telemetry import TelemetryPipeline

class TestTelemetryIntercepts(unittest.TestCase):
    def setUp(self):
        self.compliance = ComplianceInterceptor()
        self.standards = StandardsInterceptor()
        self.files = FileOperationInterceptor()
        self.pipeline = TelemetryPipeline([self.compliance, self.standards, self.files])

    def test_compliance_python_catch(self):
        """Test if 'python' usage is caught."""
        # Should execute without error
        self.compliance.validate("run_command", {"CommandLine": "uv run python script.py"})

        # Should raise PolicyViolation
        with self.assertRaises(PolicyViolation) as cm:
            self.compliance.validate("run_command", {"CommandLine": "python script.py"})
        self.assertIn("Internal Violation: Plain 'python' used", str(cm.exception))

        with self.assertRaises(PolicyViolation) as cm:
             self.compliance.validate("run_command", {"CommandLine": "echo 'hi' && python3 script.py"})
        self.assertIn("Internal Violation: Plain 'python' used", str(cm.exception))

    def test_compliance_sys_path_catch(self):
        """Test if sys.path modification is caught."""
        code = "import sys\nsys.path.append('/tmp')"  # noqa: path-hack
        with self.assertRaises(PolicyViolation) as cm:
            self.compliance.validate("write_to_file", {"CodeContent": code, "TargetFile": "test.py"})
        self.assertIn("sys.path modified", str(cm.exception))

    def test_standards_frontmatter_catch(self):
        """Test if ADS frontmatter is enforced."""
        # Minimal valid content
        valid_yaml = """
ads_version: "1.0"
type: standard
title: "Test"
agent: all
tier: 2
priority: critical
validates_with: "checker.py"
compliance_status: "certified"
memory_footprint: 100
description: "Test"
"""
        self.standards.validate("write_to_file", {
            "TargetFile": "/abs/path/to/AgentQMS/standards/test.yaml",
            "CodeContent": valid_yaml
        })

        # Invalid content (missing keys)
        invalid_yaml = """
type: standard
description: "Missing keys"
"""
        with self.assertRaises(PolicyViolation) as cm:
            self.standards.validate("write_to_file", {
                "TargetFile": "/abs/path/to/AgentQMS/standards/tier2-framework/invalid.yaml",
                "CodeContent": invalid_yaml
            })
        self.assertIn("ADS VIOLATION: Missing required ADS v1.0 frontmatter keys", cm.exception.feedback_to_ai)

    def test_file_ops_config_catch(self):
        """Test if writing to config is blocked."""
        with self.assertRaises(PolicyViolation) as cm:
            self.files.validate("write_to_file", {
                "TargetFile": "/abs/path/to/AgentQMS/config/forbidden.yaml",
                "CodeContent": "foo: bar"
            })
        self.assertIn("AgentQMS/config is read-only", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
