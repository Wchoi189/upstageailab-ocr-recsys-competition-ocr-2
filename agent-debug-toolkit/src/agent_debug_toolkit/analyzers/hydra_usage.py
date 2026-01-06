"""
HydraUsageAnalyzer - Detect Hydra framework usage patterns.

Finds patterns like:
- @hydra.main() decorators
- hydra.utils.instantiate() calls
- Hydra-specific imports
- Config composition patterns
"""

from __future__ import annotations

import ast
from typing import Any

from agent_debug_toolkit.analyzers.base import BaseAnalyzer


# Hydra-related import names
HYDRA_IMPORTS = frozenset(
    {
        "hydra",
        "hydra.main",
        "hydra.utils",
        "hydra.core",
        "omegaconf",
        "OmegaConf",
        "DictConfig",
        "ListConfig",
    }
)

# Hydra utility functions
HYDRA_UTILS = frozenset(
    {
        "instantiate",
        "call",
        "get_class",
        "get_method",
        "get_object",
    }
)


class HydraUsageAnalyzer(BaseAnalyzer):
    """
    Analyze Python code for Hydra framework usage patterns.

    Detects:
    - @hydra.main() decorated functions (entry points)
    - hydra.utils.instantiate() calls (component creation)
    - hydra.utils.call() for callable instantiation
    - Hydra-related imports
    - _target_ and _recursive_ patterns in dicts

    Categories:
    - 'entry_point': @hydra.main() decorated functions
    - 'instantiation': hydra.utils.instantiate() calls
    - 'import': Hydra/OmegaConf imports
    - 'config_pattern': Hydra-specific config patterns (_target_, etc.)
    """

    name = "HydraUsageAnalyzer"

    def __init__(self):
        super().__init__()
        self._in_function: str | None = None
        self._in_class: str | None = None
        self._imported_names: set[str] = set()

    def visit(self, node: ast.AST) -> None:
        """Visit AST nodes."""
        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        visitor(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Track Hydra-related imports."""
        for alias in node.names:
            module_name = alias.name
            local_name = alias.asname or module_name.split(".")[0]

            if any(module_name.startswith(h) for h in ("hydra", "omegaconf")):
                self._imported_names.add(local_name)

                self._add_result(
                    node=node,
                    pattern=f"import {module_name}"
                    + (f" as {alias.asname}" if alias.asname else ""),
                    context="Hydra framework import",
                    category="import",
                    metadata={"module": module_name, "local_name": local_name},
                )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from X import Y for Hydra packages."""
        if node.module and any(node.module.startswith(h) for h in ("hydra", "omegaconf")):
            for alias in node.names:
                name = alias.name
                local_name = alias.asname or name
                self._imported_names.add(local_name)

                self._add_result(
                    node=node,
                    pattern=f"from {node.module} import {name}"
                    + (f" as {alias.asname}" if alias.asname else ""),
                    context="Hydra/OmegaConf import",
                    category="import",
                    metadata={"module": node.module, "name": name, "local_name": local_name},
                )

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context."""
        old_class = self._in_class
        self._in_class = node.name
        self.generic_visit(node)
        self._in_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check for @hydra.main decorator and track function context."""
        # Check decorators for @hydra.main
        for decorator in node.decorator_list:
            if self._is_hydra_main_decorator(decorator):
                self._record_hydra_entrypoint(node, decorator)

        # Track function context
        old_func = self._in_function
        self._in_function = node.name
        self.generic_visit(node)
        self._in_function = old_func

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        """Detect hydra.utils.instantiate() and similar calls."""
        call_name = self._get_full_call_name(node)

        # Check for instantiate patterns
        if self._is_instantiate_call(call_name):
            self._record_instantiation(node, call_name)

        # Check for call patterns
        elif self._is_hydra_call(call_name):
            self._record_hydra_util_call(node, call_name)

        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        """Detect Hydra-specific patterns in dictionaries (_target_, _recursive_, etc.)."""
        hydra_keys = []

        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value.startswith("_") and key.value.endswith("_"):
                    hydra_keys.append(key.value)

        if hydra_keys:
            pattern = "{" + ", ".join(f'"{k}": ...' for k in hydra_keys) + "}"

            context_parts = []
            if self._in_class:
                context_parts.append(f"in {self._in_class}")
            if self._in_function:
                context_parts.append(f".{self._in_function}()")

            self._add_result(
                node=node,
                pattern=pattern,
                context="".join(context_parts) + " - Hydra structured config pattern",
                category="config_pattern",
                metadata={"hydra_keys": hydra_keys},
            )

        self.generic_visit(node)

    def _is_hydra_main_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is @hydra.main(...)."""
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Attribute) and func.attr == "main":
                if isinstance(func.value, ast.Name) and func.value.id == "hydra":
                    return True
        elif isinstance(decorator, ast.Attribute):
            if decorator.attr == "main":
                if isinstance(decorator.value, ast.Name) and decorator.value.id == "hydra":
                    return True
        return False

    def _record_hydra_entrypoint(self, node: ast.FunctionDef, decorator: ast.expr) -> None:
        """Record a Hydra entry point function."""
        # Extract decorator arguments
        metadata: dict[str, Any] = {
            "function_name": node.name,
            "is_entry_point": True,
        }

        if isinstance(decorator, ast.Call):
            # Extract config_path and config_name if available
            for keyword in decorator.keywords:
                if keyword.arg in ("config_path", "config_name", "version_base"):
                    metadata[keyword.arg] = self._get_constant_value(keyword.value)

        pattern = f"@hydra.main() -> {node.name}()"

        self._add_result(
            node=node,
            pattern=pattern,
            context="Hydra application entry point",
            category="entry_point",
            metadata=metadata,
        )

    def _is_instantiate_call(self, call_name: str) -> bool:
        """Check if call is hydra.utils.instantiate() or similar."""
        patterns = (
            "hydra.utils.instantiate",
            "utils.instantiate",
            "instantiate",  # If imported directly
        )
        return call_name in patterns or call_name.endswith(".instantiate")

    def _is_hydra_call(self, call_name: str) -> bool:
        """Check if call is a Hydra utility function."""
        if not call_name:
            return False

        # Check common patterns
        parts = call_name.split(".")
        last_part = parts[-1] if parts else ""

        return last_part in HYDRA_UTILS

    def _record_instantiation(self, node: ast.Call, call_name: str) -> None:
        """Record an instantiate() call."""
        # Get the config argument
        config_arg = None
        if node.args:
            config_arg = self._unparse_safe(node.args[0])
        elif node.keywords:
            for kw in node.keywords:
                if kw.arg == "config" or kw.arg is None:  # positional as keyword
                    config_arg = self._unparse_safe(kw.value)
                    break

        context_parts = []
        if self._in_class:
            context_parts.append(f"in {self._in_class}")
        if self._in_function:
            context_parts.append(f".{self._in_function}()")

        pattern = f"instantiate({config_arg or '...'})"

        metadata: dict[str, Any] = {
            "call_name": call_name,
            "config_arg": config_arg,
            "class": self._in_class,
            "function": self._in_function,
        }

        # Extract _recursive_ if specified
        for kw in node.keywords:
            if kw.arg == "_recursive_":
                metadata["recursive"] = self._get_constant_value(kw.value)

        self._add_result(
            node=node,
            pattern=pattern,
            context="".join(context_parts) + " - Creates object from config",
            category="instantiation",
            metadata=metadata,
        )

    def _record_hydra_util_call(self, node: ast.Call, call_name: str) -> None:
        """Record other Hydra utility calls."""
        args_str = ", ".join(self._unparse_safe(arg) for arg in node.args[:2])
        pattern = f"{call_name.split('.')[-1]}({args_str})"

        self._add_result(
            node=node,
            pattern=pattern,
            context="Hydra utility function",
            category="hydra_util",
            metadata={"call_name": call_name},
        )

    def _get_full_call_name(self, node: ast.Call) -> str:
        """Get the full dotted name of a call."""
        if isinstance(node.func, ast.Attribute):
            parts = []
            current: ast.expr = node.func

            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.append(current.id)

            parts.reverse()
            return ".".join(parts)

        elif isinstance(node.func, ast.Name):
            return node.func.id

        return ""

    def _get_constant_value(self, node: ast.expr) -> Any:
        """Get the value from a Constant node."""
        if isinstance(node, ast.Constant):
            return node.value
        return self._unparse_safe(node)

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary with Hydra usage breakdown."""
        base_summary = super()._generate_summary()

        entry_points = len([r for r in self._results if r.category == "entry_point"])
        instantiations = len([r for r in self._results if r.category == "instantiation"])
        imports = len([r for r in self._results if r.category == "import"])

        base_summary["entry_points"] = entry_points
        base_summary["instantiations"] = instantiations
        base_summary["hydra_imports"] = imports

        return base_summary
