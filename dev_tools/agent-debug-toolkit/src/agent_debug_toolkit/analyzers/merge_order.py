"""
MergeOrderTracker - Track OmegaConf.merge() operations and their precedence.

This analyzer is crucial for debugging configuration precedence issues where
later merges override earlier ones. It tracks:

- OmegaConf.merge() calls and their argument order
- OmegaConf.create() calls that initialize configs
- OmegaConf.update() calls that modify configs
- Variable assignments that use merged results
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from agent_debug_toolkit.analyzers.base import BaseAnalyzer


@dataclass
class MergeOperation:
    """
    Represents a single configuration merge operation.

    Attributes:
        line: Line number where merge occurs
        operation: Type of operation (merge, create, update)
        sources: List of source arguments being merged
        target: Variable receiving the merged result
        priority: Merge priority (later = higher priority for same keys)
        code: Full source code of the merge statement
    """

    line: int
    operation: str
    sources: list[str]
    target: str
    priority: int
    code: str
    column: int = 0

    def __str__(self) -> str:
        sources_str = " + ".join(self.sources)
        return f"[P{self.priority}] {self.target} = {self.operation}({sources_str})"


class MergeOrderTracker(BaseAnalyzer):
    """
    Track OmegaConf merge operations to debug configuration precedence.

    This analyzer specifically targets the pattern discovered in OCRModel where:
    1. Architecture config is merged first (lowest priority)
    2. Top-level overrides are merged second
    3. Explicit component_overrides are merged last (highest priority)

    In OmegaConf.merge(a, b), values from 'b' override values from 'a'.
    So the LAST merged config has HIGHEST priority.

    Categories:
    - 'merge': OmegaConf.merge() calls
    - 'create': OmegaConf.create() initialization
    - 'update': OmegaConf.update() modifications
    - 'assignment': Config variable assignments
    """

    name = "MergeOrderTracker"

    def __init__(self):
        super().__init__()
        self._merge_operations: list[MergeOperation] = []
        self._priority_counter = 0
        self._in_function: str | None = None
        self._in_class: str | None = None

        # Track processed call nodes to avoid duplicates
        self._processed_calls: set[int] = set()

        # Track variable definitions for flow analysis
        self._var_definitions: dict[str, int] = {}  # var_name -> line

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
        """Track function context and reset priority for each function."""
        old_func = self._in_function
        old_priority = self._priority_counter

        self._in_function = node.name
        self._priority_counter = 0

        self.generic_visit(node)

        self._in_function = old_func
        self._priority_counter = old_priority

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track assignments that involve OmegaConf operations."""
        # Get target variable name
        target_name = None
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
            elif isinstance(target, ast.Attribute):
                target_name = self._unparse_safe(target)

        # Check if the value is an OmegaConf call
        if isinstance(node.value, ast.Call):
            # Mark this call as processed so visit_Call doesn't duplicate it
            self._processed_calls.add(id(node.value))
            self._check_omegaconf_call(node.value, target_name, node)

        # Track variable definitions
        if target_name:
            self._var_definitions[target_name] = node.lineno

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check standalone OmegaConf calls (without assignment)."""
        # Skip if already processed by visit_Assign
        if id(node) in self._processed_calls:
            self.generic_visit(node)
            return
        self._check_omegaconf_call(node, None, node)
        self.generic_visit(node)

    def _check_omegaconf_call(
        self, call_node: ast.Call, target: str | None, parent_node: ast.AST
    ) -> None:
        """
        Check if a Call node is an OmegaConf operation.

        Detects patterns:
        - OmegaConf.merge(a, b)
        - OmegaConf.create({...})
        - OmegaConf.update(cfg, key, value)
        """
        call_name = self._get_full_call_name(call_node)

        if not call_name:
            return

        operation = None
        sources: list[str] = []

        if call_name in ("OmegaConf.merge", "omegaconf.OmegaConf.merge"):
            operation = "merge"
            # All args are sources being merged
            sources = [self._unparse_safe(arg) for arg in call_node.args]

        elif call_name in ("OmegaConf.create", "omegaconf.OmegaConf.create"):
            operation = "create"
            if call_node.args:
                sources = [self._unparse_safe(call_node.args[0])]
            else:
                sources = ["{}"]

        elif call_name in ("OmegaConf.update", "omegaconf.OmegaConf.update"):
            operation = "update"
            if len(call_node.args) >= 2:
                sources = [
                    self._unparse_safe(call_node.args[0]),  # config
                    self._unparse_safe(call_node.args[1]),  # key
                ]
                if len(call_node.args) >= 3:
                    sources.append(self._unparse_safe(call_node.args[2]))  # value

        elif call_name in ("OmegaConf.to_container", "omegaconf.OmegaConf.to_container"):
            operation = "to_container"
            if call_node.args:
                sources = [self._unparse_safe(call_node.args[0])]

        if operation:
            self._priority_counter += 1

            merge_op = MergeOperation(
                line=parent_node.lineno,
                column=getattr(parent_node, "col_offset", 0),
                operation=operation,
                sources=sources,
                target=target or "<inline>",
                priority=self._priority_counter,
                code=self._get_code_snippet(parent_node.lineno, context_lines=0).strip(),
            )

            self._merge_operations.append(merge_op)
            self._record_merge_finding(merge_op, parent_node)

    def _record_merge_finding(self, op: MergeOperation, node: ast.AST) -> None:
        """Record a merge operation as an analysis finding."""
        # Build detailed context
        context_parts = []
        if self._in_class:
            context_parts.append(f"in {self._in_class}")
        if self._in_function:
            context_parts.append(f".{self._in_function}()")

        if op.operation == "merge":
            # For merge ops, explain precedence
            if len(op.sources) >= 2:
                context_parts.append(f"← {op.sources[-1]} wins on conflicts")

        context = "".join(context_parts)

        metadata: dict[str, Any] = {
            "operation": op.operation,
            "sources": op.sources,
            "target": op.target,
            "priority": op.priority,
            "class": self._in_class,
            "function": self._in_function,
        }

        # Determine category
        category = f"omegaconf_{op.operation}"

        self._add_result(
            node=node, pattern=str(op), context=context, category=category, metadata=metadata
        )

    def _get_full_call_name(self, node: ast.Call) -> str:
        """Get the full dotted name of a call (e.g., 'OmegaConf.merge')."""
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

    def get_merge_sequence(self) -> list[MergeOperation]:
        """
        Get the ordered sequence of merge operations.

        Returns merge operations sorted by (function, priority) to show
        the order in which merges affect the final configuration.
        """
        return sorted(self._merge_operations, key=lambda x: (x.line, x.priority))

    def explain_precedence(self) -> str:
        """
        Generate a human-readable explanation of merge precedence.

        This is the key output for debugging config precedence issues.
        """
        if not self._merge_operations:
            return "No OmegaConf merge operations found."

        lines = ["## Configuration Merge Precedence Analysis", ""]

        # Group by function
        by_function: dict[str, list[MergeOperation]] = {}
        for op in self._merge_operations:
            key = f"{self._in_class or 'module'}.{op.target or 'inline'}"
            by_function.setdefault(key, []).append(op)

        for func_name, ops in by_function.items():
            lines.append(f"### {func_name}")
            lines.append("")
            lines.append("| Priority | Line | Operation | Winner on Conflict |")
            lines.append("|----------|------|-----------|-------------------|")

            for op in sorted(ops, key=lambda x: x.priority):
                winner = op.sources[-1] if op.sources else "N/A"
                lines.append(f"| P{op.priority} | {op.line} | `{op.operation}` | `{winner}` |")

            lines.append("")

            # Explain final precedence
            merge_ops = [op for op in ops if op.operation == "merge"]
            if merge_ops:
                final_merge = max(merge_ops, key=lambda x: x.priority)
                lines.append(f"> **Highest Priority**: The last merge at line {final_merge.line}")
                lines.append(
                    f"> wins for any conflicting keys. Source: `{final_merge.sources[-1] if final_merge.sources else 'unknown'}`"
                )
                lines.append("")

        return "\n".join(lines)

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary with merge precedence info."""
        base_summary = super()._generate_summary()

        merge_count = len([op for op in self._merge_operations if op.operation == "merge"])
        create_count = len([op for op in self._merge_operations if op.operation == "create"])

        base_summary["merge_operations"] = merge_count
        base_summary["create_operations"] = create_count
        base_summary["total_omegaconf_ops"] = len(self._merge_operations)

        # Identify highest priority merge
        merge_ops = [op for op in self._merge_operations if op.operation == "merge"]
        if merge_ops:
            highest = max(merge_ops, key=lambda x: x.priority)
            base_summary["highest_priority_merge"] = {
                "line": highest.line,
                "sources": highest.sources,
                "winner": highest.sources[-1] if highest.sources else None,
            }

        return base_summary
