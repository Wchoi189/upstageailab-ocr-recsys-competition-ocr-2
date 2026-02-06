"""
ImportTracker - Categorize and analyze Python imports.

Tracks:
- Standard library imports
- Third-party package imports
- Local/project imports
- Dynamic imports (__import__, importlib)

Outputs:
- Categorized import lists
- Unused import candidates (heuristic)
- Import statistics
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

from .base import BaseAnalyzer, AnalysisReport


# Known third-party packages (common ones)
KNOWN_THIRD_PARTY = frozenset({
    "numpy", "pandas", "torch", "tensorflow", "sklearn", "scipy",
    "matplotlib", "seaborn", "plotly", "requests", "httpx", "aiohttp",
    "flask", "django", "fastapi", "starlette", "pydantic", "sqlalchemy",
    "pytest", "hypothesis", "mypy", "ruff", "black", "isort",
    "typer", "click", "rich", "tqdm", "yaml", "toml", "dotenv",
    "PIL", "cv2", "opencv", "hydra", "omegaconf", "wandb", "mlflow",
    "boto3", "botocore", "s3fs", "fsspec", "mcp", "anthropic", "openai",
})


class ImportTracker(BaseAnalyzer):
    """
    Analyze Python imports and categorize them.

    Categories:
    - stdlib: Python standard library
    - third_party: External packages
    - local: Project-local imports
    - dynamic: Runtime imports (__import__, importlib)

    Use cases:
    - Find unused imports (heuristic)
    - Audit dependencies
    - Detect circular import candidates
    """

    name = "ImportTracker"

    def __init__(self):
        super().__init__()
        self._stdlib_imports: set[str] = set()
        self._third_party_imports: set[str] = set()
        self._local_imports: set[str] = set()
        self._dynamic_imports: list[dict[str, Any]] = []
        self._all_imports: dict[str, dict[str, Any]] = {}
        self._used_names: set[str] = set()
        self._defined_names: set[str] = set()

        # Build stdlib module list from sys.stdlib_module_names (Python 3.10+)
        if hasattr(sys, "stdlib_module_names"):
            self._stdlib_modules = frozenset(sys.stdlib_module_names)
        else:
            # Fallback for older Python
            self._stdlib_modules = frozenset({
                "abc", "aifc", "argparse", "array", "ast", "asyncio",
                "atexit", "base64", "bdb", "binascii", "bisect", "builtins",
                "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd",
                "code", "codecs", "codeop", "collections", "colorsys",
                "compileall", "concurrent", "configparser", "contextlib",
                "contextvars", "copy", "copyreg", "cProfile", "crypt",
                "csv", "ctypes", "curses", "dataclasses", "datetime",
                "dbm", "decimal", "difflib", "dis", "distutils", "doctest",
                "email", "encodings", "enum", "errno", "faulthandler",
                "fcntl", "filecmp", "fileinput", "fnmatch", "fractions",
                "ftplib", "functools", "gc", "getopt", "getpass", "gettext",
                "glob", "graphlib", "grp", "gzip", "hashlib", "heapq",
                "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
                "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
                "json", "keyword", "lib2to3", "linecache", "locale",
                "logging", "lzma", "mailbox", "mailcap", "marshal", "math",
                "mimetypes", "mmap", "modulefinder", "multiprocessing",
                "netrc", "nis", "nntplib", "numbers", "operator", "optparse",
                "os", "ossaudiodev", "pathlib", "pdb", "pickle", "pickletools",
                "pipes", "pkgutil", "platform", "plistlib", "poplib", "posix",
                "posixpath", "pprint", "profile", "pstats", "pty", "pwd",
                "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random",
                "re", "readline", "reprlib", "resource", "rlcompleter",
                "runpy", "sched", "secrets", "select", "selectors", "shelve",
                "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
                "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "ssl",
                "stat", "statistics", "string", "stringprep", "struct",
                "subprocess", "sunau", "symtable", "sys", "sysconfig",
                "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
                "termios", "test", "textwrap", "threading", "time",
                "timeit", "tkinter", "token", "tokenize", "trace",
                "traceback", "tracemalloc", "tty", "turtle", "turtledemo",
                "types", "typing", "unicodedata", "unittest", "urllib",
                "uu", "uuid", "venv", "warnings", "wave", "weakref",
                "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib",
                "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
            })

    def analyze_file(self, file_path: str | Path) -> AnalysisReport:
        """Analyze a single file for imports."""
        self._stdlib_imports = set()
        self._third_party_imports = set()
        self._local_imports = set()
        self._dynamic_imports = []
        self._all_imports = {}
        self._used_names = set()
        self._defined_names = set()
        return super().analyze_file(file_path)

    def visit(self, node: ast.AST) -> None:
        """Visit AST nodes to track imports and usage."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        visitor(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Track import statements."""
        for alias in node.names:
            module_name = alias.name
            local_name = alias.asname or module_name.split(".")[0]

            category = self._categorize_import(module_name)
            self._record_import(module_name, local_name, category, node.lineno, "import")

            self._add_result(
                node,
                f"import {module_name}" + (f" as {alias.asname}" if alias.asname else ""),
                context=f"Category: {category}",
                category=category,
                metadata={"module": module_name, "alias": alias.asname, "local_name": local_name},
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from...import statements."""
        module_name = node.module or ""
        level = node.level  # Relative import level

        for alias in node.names:
            imported_name = alias.name
            local_name = alias.asname or imported_name
            full_name = f"{module_name}.{imported_name}" if module_name else imported_name

            # Relative imports are always local
            if level > 0:
                category = "local"
            else:
                category = self._categorize_import(module_name)

            self._record_import(full_name, local_name, category, node.lineno, "from_import")

            from_str = "." * level + module_name if level else module_name
            self._add_result(
                node,
                f"from {from_str} import {imported_name}" + (f" as {alias.asname}" if alias.asname else ""),
                context=f"Category: {category}",
                category=category,
                metadata={
                    "module": module_name,
                    "name": imported_name,
                    "alias": alias.asname,
                    "local_name": local_name,
                    "level": level,
                },
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track dynamic imports."""
        func_name = self._get_call_name(node)

        if func_name == "__import__":
            if node.args and isinstance(node.args[0], ast.Constant):
                module = node.args[0].value
                self._dynamic_imports.append({
                    "type": "__import__",
                    "module": module,
                    "line": node.lineno,
                })
                self._add_result(
                    node,
                    f"__import__('{module}')",
                    context="Dynamic import",
                    category="dynamic",
                    metadata={"module": module},
                )
        elif func_name in ("importlib.import_module", "import_module"):
            if node.args and isinstance(node.args[0], ast.Constant):
                module = node.args[0].value
                self._dynamic_imports.append({
                    "type": "importlib",
                    "module": module,
                    "line": node.lineno,
                })
                self._add_result(
                    node,
                    f"importlib.import_module('{module}')",
                    context="Dynamic import via importlib",
                    category="dynamic",
                    metadata={"module": module},
                )

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Track name usage for unused import detection."""
        if isinstance(node.ctx, ast.Load):
            self._used_names.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._defined_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Track attribute access for usage detection."""
        # Get the root name
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name):
            self._used_names.add(root.id)
        self.generic_visit(node)

    def _categorize_import(self, module_name: str) -> str:
        """Categorize an import as stdlib, third_party, or local."""
        if not module_name:
            return "local"

        root_module = module_name.split(".")[0]

        if root_module in self._stdlib_modules:
            return "stdlib"
        elif root_module in KNOWN_THIRD_PARTY:
            return "third_party"
        else:
            # Heuristic: if it looks like a package (has underscores or is lowercase)
            # and isn't stdlib, assume third-party for common patterns
            # Otherwise assume local
            return "local"

    def _record_import(
        self, full_name: str, local_name: str, category: str, line: int, import_type: str
    ) -> None:
        """Record an import for tracking."""
        self._all_imports[local_name] = {
            "full_name": full_name,
            "local_name": local_name,
            "category": category,
            "line": line,
            "type": import_type,
        }

        if category == "stdlib":
            self._stdlib_imports.add(full_name)
        elif category == "third_party":
            self._third_party_imports.add(full_name)
        else:
            self._local_imports.add(full_name)

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the name of a called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name):
                return f"{value.id}.{node.func.attr}"
            return node.func.attr
        return ""

    def get_unused_imports(self) -> list[dict[str, Any]]:
        """
        Find imports that appear unused (heuristic).

        Note: This is a simple heuristic that may have false positives.
        Imports used via getattr, __all__, or in type annotations
        may be incorrectly flagged.
        """
        unused = []
        for local_name, info in self._all_imports.items():
            if local_name not in self._used_names and local_name != "*":
                unused.append({
                    "local_name": local_name,
                    "full_name": info["full_name"],
                    "line": info["line"],
                    "category": info["category"],
                })
        return unused

    def get_imports_by_category(self) -> dict[str, list[str]]:
        """Get imports grouped by category."""
        return {
            "stdlib": sorted(self._stdlib_imports),
            "third_party": sorted(self._third_party_imports),
            "local": sorted(self._local_imports),
            "dynamic": [d["module"] for d in self._dynamic_imports],
        }

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary with import statistics."""
        unused = self.get_unused_imports()

        return {
            "total_imports": len(self._all_imports),
            "stdlib_count": len(self._stdlib_imports),
            "third_party_count": len(self._third_party_imports),
            "local_count": len(self._local_imports),
            "dynamic_count": len(self._dynamic_imports),
            "potentially_unused": len(unused),
            "unused_imports": unused[:10],  # Limit to first 10
            "imports_by_category": self.get_imports_by_category(),
        }
