"""
ConfigAccessAnalyzer - Detect configuration access patterns in Python code.

Finds patterns like:
- cfg.decoder, cfg.model.head
- self.cfg.encoder, self.architecture_config_obj.decoder
- config['model']['decoder']
- getattr(cfg, 'encoder')
"""

from __future__ import annotations

import ast
from typing import Any

from agent_debug_toolkit.analyzers.base import BaseAnalyzer


# Common configuration variable names to look for
CONFIG_VAR_NAMES = frozenset({
    "cfg", "config", "conf", "settings", "opts", "options",
    "architecture_config", "architecture_config_obj", "arch_config",
    "model_config", "component_config", "overrides",
    "merged_config", "base_config", "default_config",
})

# Component names that are particularly interesting
COMPONENT_NAMES = frozenset({
    "encoder", "decoder", "head", "loss", "backbone",
    "model", "architecture", "optimizer", "scheduler",
})


class ConfigAccessAnalyzer(BaseAnalyzer):
    """
    Analyze Python code for configuration access patterns.

    Detects:
    - Attribute access: cfg.decoder, self.cfg.model
    - Subscript access: cfg['decoder'], config['model']['head']
    - getattr() calls: getattr(cfg, 'encoder')
    - hasattr() checks: hasattr(cfg, 'decoder')

    Categorizes findings as:
    - 'attribute_access': Direct attribute access (cfg.X)
    - 'subscript_access': Dictionary-style access (cfg['X'])
    - 'getattr_call': getattr(cfg, 'X')
    - 'hasattr_check': hasattr(cfg, 'X')
    - 'component_access': Access to known components (encoder, decoder, etc.)
    """

    name = "ConfigAccessAnalyzer"

    def __init__(self, config_names: set[str] | None = None):
        """
        Initialize the analyzer.

        Args:
            config_names: Optional set of variable names to track as configs.
                         Defaults to CONFIG_VAR_NAMES.
        """
        super().__init__()
        self.config_names = config_names or CONFIG_VAR_NAMES
        self._in_class: str | None = None
        self._in_function: str | None = None

    def visit(self, node: ast.AST) -> None:
        """Visit an AST node and dispatch to specific handlers."""
        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        visitor(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context."""
        old_class = self._in_class
        self._in_class = node.name
        self.generic_visit(node)
        self._in_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function context."""
        old_func = self._in_function
        self._in_function = node.name
        self.generic_visit(node)
        self._in_function = old_func

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Detect attribute access patterns like cfg.decoder or self.cfg.model.
        """
        chain = self._get_attribute_chain(node)

        if chain:
            root = chain[0]
            attr_path = ".".join(chain)

            # Check if root is a config variable
            if root in self.config_names:
                self._record_config_access(node, attr_path, "attribute_access")

            # Check for self.cfg or self.config patterns
            elif root == "self" and len(chain) > 1 and chain[1] in self.config_names:
                self._record_config_access(node, attr_path, "attribute_access")

            # Check if any part of chain suggests config access
            elif any(part in self.config_names for part in chain):
                self._record_config_access(node, attr_path, "attribute_access")

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """
        Detect subscript access patterns like cfg['decoder'] or config['model']['head'].
        """
        access_path = self._get_subscript_chain(node)

        if access_path:
            root = access_path.split("[")[0]

            # Check if root is a config variable
            if root in self.config_names or root.endswith(".cfg") or root.endswith(".config"):
                self._record_config_access(node, access_path, "subscript_access")

            # Check for self.cfg patterns
            elif root.startswith("self.") and any(n in root for n in self.config_names):
                self._record_config_access(node, access_path, "subscript_access")

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """
        Detect getattr() and hasattr() calls on config objects.
        """
        func_name = self._get_call_name(node)

        if func_name in ("getattr", "hasattr") and len(node.args) >= 2:
            first_arg = node.args[0]
            second_arg = node.args[1]

            # Get the object being accessed
            obj_repr = self._unparse_safe(first_arg)

            # Check if it's a config-like object
            if self._is_config_like(first_arg):
                # Get the attribute name if it's a string literal
                attr_name = self._get_string_value(second_arg)
                if attr_name:
                    pattern = f"{func_name}({obj_repr}, '{attr_name}')"
                else:
                    pattern = f"{func_name}({obj_repr}, {self._unparse_safe(second_arg)})"

                category = "getattr_call" if func_name == "getattr" else "hasattr_check"
                self._record_config_access(node, pattern, category)

        self.generic_visit(node)

    def _record_config_access(
        self,
        node: ast.AST,
        pattern: str,
        category: str
    ) -> None:
        """Record a configuration access finding."""
        # Determine if this accesses a known component
        final_category = category
        accessed_component = None

        for comp in COMPONENT_NAMES:
            if comp in pattern.lower():
                final_category = "component_access"
                accessed_component = comp
                break

        # Build context string
        context_parts = []
        if self._in_class:
            context_parts.append(f"in class {self._in_class}")
        if self._in_function:
            context_parts.append(f"in function {self._in_function}")
        context = ", ".join(context_parts) if context_parts else ""

        metadata: dict[str, Any] = {
            "access_type": category,
            "class": self._in_class,
            "function": self._in_function,
        }
        if accessed_component:
            metadata["component"] = accessed_component

        self._add_result(
            node=node,
            pattern=pattern,
            context=context,
            category=final_category,
            metadata=metadata
        )

    def _get_attribute_chain(self, node: ast.Attribute) -> list[str]:
        """
        Extract the full attribute chain from an Attribute node.

        For node representing 'self.cfg.model.decoder', returns:
        ['self', 'cfg', 'model', 'decoder']
        """
        chain: list[str] = []
        current: ast.expr = node

        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            chain.append(current.id)
        elif isinstance(current, ast.Call):
            # Handle cases like get_config().decoder
            chain.append(f"{self._unparse_safe(current)}()")

        chain.reverse()
        return chain

    def _get_subscript_chain(self, node: ast.Subscript) -> str:
        """
        Extract subscript access chain as a string.

        For cfg['model']['decoder'], returns: "cfg['model']['decoder']"
        """
        parts: list[str] = []
        current: ast.expr = node

        while isinstance(current, ast.Subscript):
            slice_val = current.slice

            if isinstance(slice_val, ast.Constant) and isinstance(slice_val.value, str):
                parts.append(f"['{slice_val.value}']")
            else:
                parts.append(f"[{self._unparse_safe(slice_val)}]")

            current = current.value

        # Get the base
        if isinstance(current, ast.Name):
            base = current.id
        elif isinstance(current, ast.Attribute):
            base = ".".join(self._get_attribute_chain(current))
        else:
            base = self._unparse_safe(current)

        parts.reverse()
        return base + "".join(parts)

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the name of a called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _is_config_like(self, node: ast.expr) -> bool:
        """Check if a node represents a config-like object."""
        if isinstance(node, ast.Name):
            return node.id in self.config_names
        elif isinstance(node, ast.Attribute):
            chain = self._get_attribute_chain(node)
            return any(part in self.config_names for part in chain)
        return False

    def _get_string_value(self, node: ast.expr) -> str | None:
        """Get string value from a Constant node if it's a string."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary with component breakdown."""
        base_summary = super()._generate_summary()

        # Count component accesses
        components: dict[str, int] = {}
        for r in self._results:
            comp = r.metadata.get("component")
            if comp:
                components[comp] = components.get(comp, 0) + 1

        base_summary["by_component"] = components
        return base_summary
