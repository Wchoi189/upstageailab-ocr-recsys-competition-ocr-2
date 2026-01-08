#!/usr/bin/env python3
"""
Check for common test isolation issues that cause CI failures.
Run as pre-commit hook to catch mock pollution before push.
"""

import ast
import sys
from pathlib import Path


def check_module_level_mocking(filepath: Path) -> list[str]:
    """Check if file has module-level sys.modules mocking without cleanup."""
    issues = []

    with open(filepath) as f:
        try:
            tree = ast.parse(f.read(), filepath)
        except SyntaxError:
            return []  # Syntax errors caught by other hooks

    has_sys_modules_mock = False
    has_cleanup_fixture = False

    for node in ast.walk(tree):
        # Check for sys.modules assignment at module level
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    if (
                        isinstance(target.value, ast.Attribute)
                        and isinstance(target.value.value, ast.Name)
                        and target.value.value.id == "sys"
                        and target.value.attr == "modules"
                    ):
                        has_sys_modules_mock = True

        # Check for cleanup fixture
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "fixture":
                        # Has a fixture, check if it has cleanup (yield)
                        for stmt in ast.walk(node):
                            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                                has_cleanup_fixture = True

    if has_sys_modules_mock and not has_cleanup_fixture:
        issues.append(
            f"{filepath}: Module-level sys.modules mocking without cleanup fixture. "
            "This causes mock pollution. Use a fixture with yield for cleanup."
        )

    return issues


def main():
    """Check all test files for isolation issues."""
    test_dir = Path("tests")
    issues = []

    for test_file in test_dir.rglob("*.py"):
        if test_file.name.startswith("test_"):
            issues.extend(check_module_level_mocking(test_file))

    if issues:
        print("❌ Test Isolation Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 Fix: Use pytest fixture with cleanup:")
        print("""
@pytest.fixture(autouse=True)
def mock_module():
    original = sys.modules.get('module')
    sys.modules['module'] = MagicMock()
    yield
    if original:
        sys.modules['module'] = original
    else:
        sys.modules.pop('module', None)
""")
        return 1

    print("✅ No test isolation issues found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
