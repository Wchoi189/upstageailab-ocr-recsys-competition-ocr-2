"""
SecurityScanner - Detect unsafe patterns in configuration and code.

Patterns detected:
  Hydra-specific:
    - unsafe_instantiate: instantiate() with user input
    - dynamic_target: _target_ from untrusted source
    - config_exec: cfg values passed to eval/exec

  General:
    - hardcoded_secrets: API keys, passwords in code
    - path_traversal: os.path.join with unchecked input
    - shell_injection: subprocess with shell=True + config

Severity levels: critical, high, medium, low, info
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseAnalyzer, AnalysisReport, AnalysisResult


@dataclass
class SecurityFinding:
    """A security finding with severity and recommendation."""

    pattern_name: str
    severity: str  # critical, high, medium, low, info
    description: str
    recommendation: str
    line: int
    column: int


# Patterns for detecting hardcoded secrets
SECRET_PATTERNS = [
    (r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"][^'\"]{8,}['\"]", "API key"),
    (r"(?i)(secret|password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{4,}['\"]", "Password/Secret"),
    (r"(?i)(token|auth[_-]?token)\s*[=:]\s*['\"][^'\"]{8,}['\"]", "Auth token"),
    (r"(?i)(aws[_-]?access|aws[_-]?secret)\s*[=:]\s*['\"][^'\"]{8,}['\"]", "AWS credential"),
    (r"sk-[a-zA-Z0-9]{32,}", "OpenAI API key"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth token"),
]


class SecurityScanner(BaseAnalyzer):
    """
    Scan Python code for security vulnerabilities in configuration usage.

    Detects both Hydra-specific unsafe patterns and general security issues
    like hardcoded secrets and shell injection.
    """

    name = "SecurityScanner"
    severity_levels = ["critical", "high", "medium", "low", "info"]

    def __init__(self, min_severity: str = "info"):
        """
        Initialize the security scanner.

        Args:
            min_severity: Minimum severity level to report (critical, high, medium, low, info)
        """
        super().__init__()
        self.min_severity = min_severity
        self._findings: list[SecurityFinding] = []
        self._severity_order = {s: i for i, s in enumerate(self.severity_levels)}

    def analyze_file(self, file_path: str | Path) -> AnalysisReport:
        """Analyze a single file for security issues."""
        path = Path(file_path).resolve()
        self._current_file = str(path)
        self._results = []
        self._findings = []

        try:
            source = path.read_text(encoding="utf-8")
            self._source_lines = source.splitlines()
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            return AnalysisReport(
                analyzer_name=self.name,
                target_path=str(path),
                summary={"error": f"Syntax error: {e}"},
            )
        except FileNotFoundError:
            return AnalysisReport(
                analyzer_name=self.name,
                target_path=str(path),
                summary={"error": f"File not found: {path}"},
            )

        # Run AST-based checks
        self.visit(tree)

        # Run regex-based checks for secrets
        self._scan_for_secrets(source)

        # Generate results from findings
        self._generate_results()

        return AnalysisReport(
            analyzer_name=self.name,
            target_path=str(path),
            results=self._results.copy(),
            summary=self._generate_summary(),
        )

    def analyze_source(self, source: str, filename: str = "<string>") -> AnalysisReport:
        """Analyze Python source code directly."""
        self._current_file = filename
        self._source_lines = source.splitlines()
        self._results = []
        self._findings = []

        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            return AnalysisReport(
                analyzer_name=self.name,
                target_path=filename,
                summary={"error": f"Syntax error: {e}"},
            )

        # Run AST-based checks
        self.visit(tree)

        # Run regex-based checks for secrets
        self._scan_for_secrets(source)

        # Generate results from findings
        self._generate_results()

        return AnalysisReport(
            analyzer_name=self.name,
            target_path=filename,
            results=self._results.copy(),
            summary=self._generate_summary(),
        )

    def visit(self, node: ast.AST) -> None:
        """Visit AST nodes to detect security patterns."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        visitor(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect dangerous function calls."""
        # Check for eval/exec calls
        if isinstance(node.func, ast.Name):
            if node.func.id == "eval":
                self._check_eval_exec(node, "eval")
            elif node.func.id == "exec":
                self._check_eval_exec(node, "exec")

        # Check for subprocess calls with shell=True
        if self._is_subprocess_call(node):
            self._check_subprocess(node)

        # Check for hydra.utils.instantiate with dynamic _target_
        if self._is_instantiate_call(node):
            self._check_instantiate(node)

        # Check for os.system calls
        if self._is_os_system_call(node):
            self._findings.append(
                SecurityFinding(
                    pattern_name="os_system",
                    severity="high",
                    description="os.system() call detected",
                    recommendation="Use subprocess.run() with shell=False instead",
                    line=node.lineno,
                    column=node.col_offset,
                )
            )

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Detect dynamic _target_ access patterns."""
        if isinstance(node.slice, ast.Constant) and node.slice.value == "_target_":
            # Check if this is cfg["_target_"] or similar
            if isinstance(node.value, ast.Name) and node.value.id.lower() in (
                "cfg",
                "config",
                "conf",
            ):
                self._findings.append(
                    SecurityFinding(
                        pattern_name="dynamic_target",
                        severity="high",
                        description="Dynamic _target_ access from config",
                        recommendation="Validate _target_ against allowlist before use",
                        line=node.lineno,
                        column=node.col_offset,
                    )
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Detect _target_ attribute access."""
        if node.attr == "_target_":
            if isinstance(node.value, ast.Name) and node.value.id.lower() in (
                "cfg",
                "config",
                "conf",
            ):
                self._findings.append(
                    SecurityFinding(
                        pattern_name="dynamic_target",
                        severity="medium",
                        description="Dynamic _target_ attribute access",
                        recommendation="Ensure _target_ comes from trusted config source",
                        line=node.lineno,
                        column=node.col_offset,
                    )
                )
        self.generic_visit(node)

    def _check_eval_exec(self, node: ast.Call, func_name: str) -> None:
        """Check if eval/exec is called with config values."""
        severity = "critical"
        description = f"{func_name}() call detected"
        recommendation = f"Avoid {func_name}() - use safer alternatives"

        # Check if argument comes from config
        if node.args:
            arg = node.args[0]
            if self._is_config_value(arg):
                description = f"{func_name}() with configuration value - code injection risk"
                recommendation = "Never pass user/config input to eval/exec"

        self._findings.append(
            SecurityFinding(
                pattern_name=f"config_{func_name}",
                severity=severity,
                description=description,
                recommendation=recommendation,
                line=node.lineno,
                column=node.col_offset,
            )
        )

    def _check_subprocess(self, node: ast.Call) -> None:
        """Check subprocess calls for shell injection risks."""
        has_shell_true = False
        has_config_arg = False

        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    has_shell_true = True

        # Check if command argument comes from config
        if node.args:
            if self._is_config_value(node.args[0]):
                has_config_arg = True

        if has_shell_true:
            severity = "critical" if has_config_arg else "high"
            description = (
                "subprocess with shell=True and config input"
                if has_config_arg
                else "subprocess with shell=True"
            )
            self._findings.append(
                SecurityFinding(
                    pattern_name="shell_injection",
                    severity=severity,
                    description=description,
                    recommendation="Use shell=False and pass arguments as list",
                    line=node.lineno,
                    column=node.col_offset,
                )
            )

    def _check_instantiate(self, node: ast.Call) -> None:
        """Check hydra instantiate for unsafe patterns."""
        # Look for _target_ in keyword arguments
        for keyword in node.keywords:
            if keyword.arg == "_target_":
                if not isinstance(keyword.value, ast.Constant):
                    # Dynamic _target_ value
                    self._findings.append(
                        SecurityFinding(
                            pattern_name="unsafe_instantiate",
                            severity="high",
                            description="hydra.utils.instantiate with dynamic _target_",
                            recommendation="Use static _target_ or validate against allowlist",
                            line=node.lineno,
                            column=node.col_offset,
                        )
                    )
                    return

        # If config argument is passed, check if it could have dynamic _target_
        if node.args:
            arg = node.args[0]
            if self._is_config_value(arg):
                self._findings.append(
                    SecurityFinding(
                        pattern_name="unsafe_instantiate",
                        severity="medium",
                        description="instantiate() with config object",
                        recommendation="Ensure _target_ in config is validated/trusted",
                        line=node.lineno,
                        column=node.col_offset,
                    )
                )

    def _is_subprocess_call(self, node: ast.Call) -> bool:
        """Check if this is a subprocess call."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("run", "call", "Popen", "check_output", "check_call"):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                    return True
        elif isinstance(node.func, ast.Name):
            if node.func.id in ("run", "call", "Popen", "check_output", "check_call"):
                return True
        return False

    def _is_instantiate_call(self, node: ast.Call) -> bool:
        """Check if this is a hydra instantiate call."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "instantiate":
                return True
        elif isinstance(node.func, ast.Name):
            if node.func.id == "instantiate":
                return True
        return False

    def _is_os_system_call(self, node: ast.Call) -> bool:
        """Check if this is an os.system call."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "system":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    return True
        return False

    def _is_config_value(self, node: ast.expr) -> bool:
        """Check if an expression represents a config value."""
        if isinstance(node, ast.Attribute):
            # Check for cfg.something pattern
            if isinstance(node.value, ast.Name):
                if node.value.id.lower() in ("cfg", "config", "conf", "self"):
                    return True
            # Recursive check for cfg.something.deeper
            return self._is_config_value(node.value)
        elif isinstance(node, ast.Subscript):
            # Check for cfg["key"] pattern
            if isinstance(node.value, ast.Name):
                if node.value.id.lower() in ("cfg", "config", "conf"):
                    return True
            return self._is_config_value(node.value)
        elif isinstance(node, ast.Name):
            if node.id.lower() in ("cfg", "config", "conf"):
                return True
        elif isinstance(node, ast.Call):
            # Check for getattr(cfg, "key") pattern
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if node.args and self._is_config_value(node.args[0]):
                    return True
        return False

    def _scan_for_secrets(self, source: str) -> None:
        """Scan source code for hardcoded secrets using regex patterns."""
        for line_num, line in enumerate(source.splitlines(), 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern, secret_type in SECRET_PATTERNS:
                if re.search(pattern, line):
                    self._findings.append(
                        SecurityFinding(
                            pattern_name="hardcoded_secret",
                            severity="critical",
                            description=f"Potential hardcoded {secret_type}",
                            recommendation="Use environment variables or secure config management",
                            line=line_num,
                            column=0,
                        )
                    )
                    break  # Only report one finding per line

    def _generate_results(self) -> None:
        """Generate analysis results from findings."""
        min_level = self._severity_order.get(self.min_severity, 4)

        for finding in self._findings:
            finding_level = self._severity_order.get(finding.severity, 4)
            if finding_level <= min_level:
                self._results.append(
                    AnalysisResult(
                        file=self._current_file,
                        line=finding.line,
                        column=finding.column,
                        pattern=f"[{finding.severity.upper()}] {finding.pattern_name}",
                        context=finding.description,
                        category=finding.severity,
                        code_snippet=self._get_code_snippet(finding.line),
                        metadata={
                            "pattern_name": finding.pattern_name,
                            "severity": finding.severity,
                            "recommendation": finding.recommendation,
                        },
                    )
                )

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        by_severity: dict[str, int] = {}
        by_pattern: dict[str, int] = {}

        for finding in self._findings:
            by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
            by_pattern[finding.pattern_name] = by_pattern.get(finding.pattern_name, 0) + 1

        # Count findings at or above min_severity
        min_level = self._severity_order.get(self.min_severity, 4)
        reported = sum(
            1
            for f in self._findings
            if self._severity_order.get(f.severity, 4) <= min_level
        )

        return {
            "total_findings": len(self._findings),
            "reported_findings": reported,
            "min_severity": self.min_severity,
            "by_severity": by_severity,
            "by_pattern": by_pattern,
        }

    def get_findings(self) -> list[SecurityFinding]:
        """Get all security findings."""
        return self._findings

    def get_findings_by_severity(self, severity: str) -> list[SecurityFinding]:
        """Get findings filtered by severity."""
        return [f for f in self._findings if f.severity == severity]

    def get_critical_findings(self) -> list[SecurityFinding]:
        """Get only critical severity findings."""
        return self.get_findings_by_severity("critical")
