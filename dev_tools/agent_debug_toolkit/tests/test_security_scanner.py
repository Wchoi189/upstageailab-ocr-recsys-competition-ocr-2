"""Tests for SecurityScanner."""

from agent_debug_toolkit.analyzers.security_scanner import SecurityScanner


class TestSecurityScanner:
    """Test cases for SecurityScanner."""

    def test_eval_detection(self):
        """Test detection of eval() calls."""
        source = """
result = eval("1 + 2")
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert len(report.results) >= 1
        assert any(r.metadata.get("pattern_name") == "config_eval" for r in report.results)

    def test_eval_with_config_critical(self):
        """Test that eval with config is flagged as critical."""
        source = """
result = eval(cfg.command)
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        findings = analyzer.get_critical_findings()
        assert len(findings) >= 1

    def test_exec_detection(self):
        """Test detection of exec() calls."""
        source = """
exec(code_string)
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert any(r.metadata.get("pattern_name") == "config_exec" for r in report.results)

    def test_subprocess_shell_true(self):
        """Test detection of subprocess with shell=True."""
        source = """
import subprocess
subprocess.run("ls -la", shell=True)
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert any(r.metadata.get("pattern_name") == "shell_injection" for r in report.results)

    def test_subprocess_shell_false_safe(self):
        """Test that subprocess with shell=False is not flagged."""
        source = """
import subprocess
subprocess.run(["ls", "-la"], shell=False)
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        # Should not flag shell_injection for shell=False
        assert not any(r.metadata.get("pattern_name") == "shell_injection" for r in report.results)

    def test_os_system_detection(self):
        """Test detection of os.system calls."""
        source = """
import os
os.system("rm -rf /tmp/test")
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert any(r.metadata.get("pattern_name") == "os_system" for r in report.results)

    def test_hardcoded_api_key(self):
        """Test detection of hardcoded API keys."""
        source = '''
api_key = "sk-1234567890abcdef1234567890abcdef12345678"
'''
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert any(r.metadata.get("pattern_name") == "hardcoded_secret" for r in report.results)

    def test_hardcoded_password(self):
        """Test detection of hardcoded passwords."""
        source = '''
password = "my_secret_password_123"
'''
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert any(r.metadata.get("pattern_name") == "hardcoded_secret" for r in report.results)

    def test_dynamic_target_subscript(self):
        """Test detection of dynamic _target_ access via subscript."""
        source = """
target = cfg["_target_"]
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert any(r.metadata.get("pattern_name") == "dynamic_target" for r in report.results)

    def test_dynamic_target_attribute(self):
        """Test detection of dynamic _target_ access via attribute."""
        source = """
target = cfg._target_
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert any(r.metadata.get("pattern_name") == "dynamic_target" for r in report.results)

    def test_instantiate_with_dynamic_target(self):
        """Test detection of instantiate with dynamic _target_."""
        source = """
from hydra.utils import instantiate
obj = instantiate(cfg, _target_=cfg.target_class)
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        # Should detect unsafe instantiate pattern
        findings = analyzer.get_findings()
        assert len(findings) >= 1

    def test_severity_filtering(self):
        """Test that severity filter works."""
        source = '''
eval(cfg.command)  # critical
api_key = "sk-1234567890abcdef1234567890abcdef12345678"  # critical
subprocess.run("ls", shell=True)  # high
'''
        # Filter for critical only
        analyzer = SecurityScanner(min_severity="critical")
        report = analyzer.analyze_source(source)

        # All reported should be critical
        for result in report.results:
            assert result.category == "critical"

    def test_clean_code_no_findings(self):
        """Test that clean code produces no findings."""
        source = """
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert len(report.results) == 0

    def test_summary_by_severity(self):
        """Test summary includes counts by severity."""
        source = """
eval(user_input)
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert "by_severity" in report.summary
        assert "by_pattern" in report.summary

    def test_comment_line_not_flagged(self):
        """Test that comments are not flagged for secrets."""
        source = """
# api_key = "sk-1234567890abcdef1234567890abcdef12345678"
# This is just a comment about secrets
real_key = os.environ.get("API_KEY")
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        # Comment should not trigger hardcoded_secret
        assert not any(
            r.metadata.get("pattern_name") == "hardcoded_secret" for r in report.results
        )

    def test_get_critical_findings(self):
        """Test get_critical_findings helper method."""
        source = """
eval("1+1")
"""
        analyzer = SecurityScanner()
        analyzer.analyze_source(source)

        critical = analyzer.get_critical_findings()
        assert len(critical) >= 1
        assert all(f.severity == "critical" for f in critical)

    def test_recommendation_included(self):
        """Test that recommendations are included in results."""
        source = """
eval(code)
"""
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert len(report.results) >= 1
        for result in report.results:
            assert "recommendation" in result.metadata
            assert result.metadata["recommendation"]  # Non-empty

    def test_github_token_detection(self):
        """Test detection of GitHub tokens."""
        source = '''
token = "ghp_1234567890abcdef1234567890abcdef1234"
'''
        analyzer = SecurityScanner()
        report = analyzer.analyze_source(source)

        assert any(r.metadata.get("pattern_name") == "hardcoded_secret" for r in report.results)
