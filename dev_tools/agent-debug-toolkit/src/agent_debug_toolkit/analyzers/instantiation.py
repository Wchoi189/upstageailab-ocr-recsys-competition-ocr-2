"""
ComponentInstantiationTracker - Track component factory pattern usage.

Specifically designed for projects using registry/factory patterns like:
- get_encoder_by_cfg(cfg.encoder)
- get_decoder_by_cfg(cfg.decoder)
- registry.create_architecture_components(name, **configs)
- ComponentRegistry.get(name)
"""

from __future__ import annotations

import ast
import re
from typing import Any

from agent_debug_toolkit.analyzers.base import BaseAnalyzer


# Common factory function patterns
FACTORY_PATTERNS = [
    re.compile(r"get_(\w+)_by_cfg"),  # get_encoder_by_cfg, get_decoder_by_cfg
    re.compile(r"create_(\w+)"),  # create_encoder, create_model
    re.compile(r"build_(\w+)"),  # build_encoder, build_loss
    re.compile(r"make_(\w+)"),  # make_optimizer
    re.compile(r"(\w+)_from_cfg"),  # encoder_from_cfg
    re.compile(r"(\w+)_factory"),  # model_factory
]

# Registry method patterns
REGISTRY_METHODS = frozenset(
    {
        "get",
        "create",
        "register",
        "create_architecture_components",
        "list_encoders",
        "list_decoders",
        "list_heads",
        "list_losses",
    }
)

# Component type names
COMPONENT_TYPES = frozenset(
    {
        "encoder",
        "decoder",
        "head",
        "loss",
        "backbone",
        "optimizer",
        "scheduler",
        "model",
        "architecture",
        "transform",
        "augmentation",
        "dataset",
        "dataloader",
    }
)


class ComponentInstantiationTracker(BaseAnalyzer):
    """
    Track component instantiation via factory patterns and registries.

    This analyzer helps trace how components are created, which is essential
    for debugging "wrong component" bugs where the wrong encoder/decoder/etc
    is instantiated due to configuration issues.

    Detects:
    - get_*_by_cfg() factory function calls
    - Registry.create() / Registry.get() patterns
    - Direct component class instantiation
    - create_architecture_components() patterns

    Categories:
    - 'factory_call': get_encoder_by_cfg(), etc.
    - 'registry_call': registry.create(), registry.get()
    - 'direct_instantiation': EncoderClass(), DecoderClass()
    - 'architecture_creation': create_architecture_components()
    """

    name = "ComponentInstantiationTracker"

    def __init__(self, additional_factories: list[str] | None = None):
        """
        Initialize the tracker.

        Args:
            additional_factories: Additional factory function names to track
        """
        super().__init__()
        self._in_function: str | None = None
        self._in_class: str | None = None
        self._additional_factories = set(additional_factories or [])

        # Track instantiation flow
        self._instantiation_order: list[dict[str, Any]] = []

    def visit(self, node: ast.AST) -> None:
        """Visit AST nodes."""
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

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track assignments that create components."""
        # Get target name
        target_name = None
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
            elif isinstance(target, ast.Attribute):
                target_name = self._unparse_safe(target)

        if isinstance(node.value, ast.Call):
            self._check_component_call(node.value, target_name, node)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for component creation calls."""
        self._check_component_call(node, None, node)
        self.generic_visit(node)

    def _check_component_call(
        self, call_node: ast.Call, target: str | None, parent_node: ast.AST
    ) -> None:
        """Check if a call creates a component."""
        call_name = self._get_call_name(call_node)
        full_call_name = self._get_full_call_name(call_node)

        if not call_name:
            return

        category = None
        component_type = None
        config_source = None

        # Check factory patterns
        for pattern in FACTORY_PATTERNS:
            match = pattern.match(call_name)
            if match:
                category = "factory_call"
                component_type = match.group(1) if match.groups() else call_name

                # Get config argument
                if call_node.args:
                    config_source = self._unparse_safe(call_node.args[0])
                break

        # Check registry method patterns
        if not category and call_name in REGISTRY_METHODS:
            category = "registry_call"

            # Try to extract component type from context
            if "create_architecture_components" in full_call_name:
                category = "architecture_creation"
                if call_node.args:
                    component_type = self._unparse_safe(call_node.args[0])
            elif call_node.args:
                config_source = self._unparse_safe(call_node.args[0])

        # Check additional factories
        if not category and call_name in self._additional_factories:
            category = "factory_call"
            if call_node.args:
                config_source = self._unparse_safe(call_node.args[0])

        # Check direct component class instantiation
        if not category and self._looks_like_component_class(call_name):
            category = "direct_instantiation"
            component_type = call_name

        if category:
            self._record_instantiation(
                node=parent_node,
                call_name=full_call_name or call_name,
                category=category,
                component_type=component_type,
                config_source=config_source,
                target=target,
            )

    def _record_instantiation(
        self,
        node: ast.AST,
        call_name: str,
        category: str,
        component_type: str | None,
        config_source: str | None,
        target: str | None,
    ) -> None:
        """Record a component instantiation finding."""
        context_parts = []
        if self._in_class:
            context_parts.append(f"in {self._in_class}")
        if self._in_function:
            context_parts.append(f".{self._in_function}()")

        # Build pattern string
        if config_source:
            pattern = f"{call_name}({config_source})"
        else:
            pattern = f"{call_name}(...)"

        if target:
            pattern = f"{target} = {pattern}"

        # Additional context about what's being created
        creation_context = []
        if component_type:
            creation_context.append(f"creates {component_type}")
        if config_source:
            creation_context.append(f"from {config_source}")

        context = "".join(context_parts)
        if creation_context:
            context += " - " + ", ".join(creation_context)

        metadata: dict[str, Any] = {
            "call_name": call_name,
            "component_type": component_type,
            "config_source": config_source,
            "target_variable": target,
            "class": self._in_class,
            "function": self._in_function,
        }

        self._add_result(
            node=node, pattern=pattern, context=context, category=category, metadata=metadata
        )

        # Track order
        self._instantiation_order.append(
            {
                "line": node.lineno,
                "component": component_type,
                "source": config_source,
                "target": target,
            }
        )

    def _looks_like_component_class(self, name: str) -> bool:
        """Check if a name looks like a component class (e.g., FPNDecoder, ResNetEncoder)."""
        if not name or not name[0].isupper():
            return False

        name_lower = name.lower()
        return any(comp in name_lower for comp in COMPONENT_TYPES)

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the simple name of a called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

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

    def get_instantiation_flow(self) -> list[dict[str, Any]]:
        """Get the ordered flow of component instantiations."""
        return sorted(self._instantiation_order, key=lambda x: x["line"])

    def find_component_source(self, component_name: str) -> list[dict[str, Any]]:
        """
        Find where a specific component type is instantiated and what config sources it.

        Args:
            component_name: e.g., "decoder", "FPNDecoder"

        Returns:
            List of instantiation records for that component
        """
        results = []
        component_lower = component_name.lower()

        for result in self._results:
            comp_type = result.metadata.get("component_type", "")
            if comp_type and component_lower in comp_type.lower():
                results.append(
                    {
                        "line": result.line,
                        "pattern": result.pattern,
                        "config_source": result.metadata.get("config_source"),
                        "function": result.metadata.get("function"),
                    }
                )

        return results

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary with component breakdown."""
        base_summary = super()._generate_summary()

        # Count by component type
        components: dict[str, int] = {}
        for r in self._results:
            comp = r.metadata.get("component_type")
            if comp:
                components[comp] = components.get(comp, 0) + 1

        base_summary["by_component_type"] = components
        base_summary["instantiation_count"] = len(self._instantiation_order)

        return base_summary
