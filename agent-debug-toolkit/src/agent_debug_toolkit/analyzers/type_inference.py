"""
TypeInferenceAnalyzer - Infer variable and function types from assignments.

Tracks:
- Assignment types: x = 5 → int, y = "foo" → str
- Function return types: Inferred from return statements
- Type hints: Explicit annotations
- Type narrowing: if isinstance(x, Foo): ...

Limitations:
- No cross-file analysis (single-file scope)
- No runtime type inference
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseAnalyzer, AnalysisReport, AnalysisResult


@dataclass
class TypeInfo:
    """Information about an inferred type."""

    name: str
    inferred_type: str
    source: str  # "assignment", "annotation", "return", "parameter"
    line: int
    confidence: str  # "high", "medium", "low"


@dataclass
class FunctionTypeInfo:
    """Type information for a function."""

    name: str
    qualified_name: str
    params: dict[str, str]  # param_name -> type
    return_type: str | None
    line: int


class TypeInferenceAnalyzer(BaseAnalyzer):
    """
    Analyze Python code to infer variable and function types.

    Tracks assignments, type annotations, and return types to build
    a type map for the analyzed file. Detects type conflicts where
    the same variable is assigned different types.
    """

    name = "TypeInferenceAnalyzer"

    def __init__(self):
        super().__init__()
        self._variables: dict[str, list[TypeInfo]] = defaultdict(list)
        self._functions: dict[str, FunctionTypeInfo] = {}
        self._type_conflicts: list[tuple[str, list[TypeInfo]]] = []
        self._class_stack: list[str] = []
        self._current_function: str | None = None

    def analyze_file(self, file_path: str | Path) -> AnalysisReport:
        """Analyze a single file for type information."""
        path = Path(file_path).resolve()
        self._current_file = str(path)
        self._results = []
        self._reset_state()

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

        # Run the visitor
        self.visit(tree)

        # Detect type conflicts
        self._detect_conflicts()

        # Generate results
        self._generate_results()

        return AnalysisReport(
            analyzer_name=self.name,
            target_path=str(path),
            results=self._results.copy(),
            summary=self._generate_summary(),
        )

    def _reset_state(self) -> None:
        """Reset internal state for a new file."""
        self._variables = defaultdict(list)
        self._functions = {}
        self._type_conflicts = []
        self._class_stack = []
        self._current_function = None

    def analyze_source(self, source: str, filename: str = "<string>") -> AnalysisReport:
        """Analyze Python source code directly."""
        self._current_file = filename
        self._source_lines = source.splitlines()
        self._results = []
        self._reset_state()

        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            return AnalysisReport(
                analyzer_name=self.name,
                target_path=filename,
                summary={"error": f"Syntax error: {e}"},
            )

        # Run the visitor
        self.visit(tree)

        # Detect type conflicts
        self._detect_conflicts()

        # Generate results
        self._generate_results()

        return AnalysisReport(
            analyzer_name=self.name,
            target_path=filename,
            results=self._results.copy(),
            summary=self._generate_summary(),
        )

    def visit(self, node: ast.AST) -> None:
        """Visit AST nodes to extract type information."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        visitor(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context for qualified names."""
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze function for type information."""
        self._analyze_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Analyze async function for type information."""
        self._analyze_function(node)

    def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Extract type information from a function definition."""
        # Build qualified name
        if self._class_stack:
            qualified_name = f"{'.'.join(self._class_stack)}.{node.name}"
        else:
            qualified_name = node.name

        old_function = self._current_function
        self._current_function = qualified_name

        # Extract parameter types
        params: dict[str, str] = {}
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation:
                params[arg.arg] = self._annotation_to_str(arg.annotation)
            else:
                params[arg.arg] = "Unknown"

        # Handle *args and **kwargs
        if node.args.vararg:
            arg_name = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                params[arg_name] = self._annotation_to_str(node.args.vararg.annotation)
            else:
                params[arg_name] = "Unknown"

        if node.args.kwarg:
            kwarg_name = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                params[kwarg_name] = self._annotation_to_str(node.args.kwarg.annotation)
            else:
                params[kwarg_name] = "Unknown"

        # Extract return type annotation
        return_type = None
        if node.returns:
            return_type = self._annotation_to_str(node.returns)
        else:
            # Infer from return statements
            return_type = self._infer_return_type(node)

        self._functions[qualified_name] = FunctionTypeInfo(
            name=node.name,
            qualified_name=qualified_name,
            params=params,
            return_type=return_type,
            line=node.lineno,
        )

        # Visit body for variable assignments
        self.generic_visit(node)
        self._current_function = old_function

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments: x: int = 5."""
        if isinstance(node.target, ast.Name):
            type_str = self._annotation_to_str(node.annotation)
            self._variables[node.target.id].append(
                TypeInfo(
                    name=node.target.id,
                    inferred_type=type_str,
                    source="annotation",
                    line=node.lineno,
                    confidence="high",
                )
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle regular assignments: x = 5."""
        inferred_type = self._infer_type_from_value(node.value)

        for target in node.targets:
            if isinstance(target, ast.Name):
                self._variables[target.id].append(
                    TypeInfo(
                        name=target.id,
                        inferred_type=inferred_type,
                        source="assignment",
                        line=node.lineno,
                        confidence="medium" if inferred_type != "Unknown" else "low",
                    )
                )
            elif isinstance(target, ast.Tuple):
                # Handle tuple unpacking: a, b = 1, 2
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self._variables[elt.id].append(
                            TypeInfo(
                                name=elt.id,
                                inferred_type="Unknown",
                                source="assignment",
                                line=node.lineno,
                                confidence="low",
                            )
                        )
        self.generic_visit(node)

    def _infer_type_from_value(self, node: ast.expr) -> str:
        """Infer type from an expression."""
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "None"
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            if node.elts:
                elem_type = self._infer_type_from_value(node.elts[0])
                return f"list[{elem_type}]"
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Set):
            return "set"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        elif isinstance(node, ast.ListComp):
            return "list"
        elif isinstance(node, ast.DictComp):
            return "dict"
        elif isinstance(node, ast.SetComp):
            return "set"
        elif isinstance(node, ast.GeneratorExp):
            return "Generator"
        elif isinstance(node, ast.Call):
            # Try to infer from the call target
            if isinstance(node.func, ast.Name):
                # Common constructors
                if node.func.id in ("list", "dict", "set", "tuple", "str", "int", "float", "bool"):
                    return node.func.id
                # Likely a class instantiation
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return node.func.attr
            return "Unknown"
        elif isinstance(node, ast.BinOp):
            # Binary operations often preserve type
            left_type = self._infer_type_from_value(node.left)
            if left_type in ("int", "float", "str"):
                return left_type
            return "Unknown"
        elif isinstance(node, ast.Compare):
            return "bool"
        elif isinstance(node, ast.BoolOp):
            return "bool"
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return "bool"
            return self._infer_type_from_value(node.operand)
        elif isinstance(node, ast.IfExp):
            # Ternary expression - return body type
            return self._infer_type_from_value(node.body)
        elif isinstance(node, ast.Lambda):
            return "Callable"
        elif isinstance(node, ast.Await):
            return "Awaitable"

        return "Unknown"

    def _infer_return_type(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
        """Infer function return type from return statements."""
        return_types: set[str] = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                if child.value is None:
                    return_types.add("None")
                else:
                    return_types.add(self._infer_type_from_value(child.value))

        if not return_types:
            return "None"
        elif len(return_types) == 1:
            return return_types.pop()
        else:
            # Multiple return types
            return " | ".join(sorted(return_types))

    def _annotation_to_str(self, annotation: ast.expr) -> str:
        """Convert annotation AST to string."""
        try:
            return ast.unparse(annotation)
        except Exception:
            return "Unknown"

    def _detect_conflicts(self) -> None:
        """Detect variables with conflicting types."""
        for var_name, type_infos in self._variables.items():
            types = set(t.inferred_type for t in type_infos if t.inferred_type != "Unknown")
            if len(types) > 1:
                self._type_conflicts.append((var_name, type_infos))

    def _generate_results(self) -> None:
        """Generate analysis results from collected type info."""
        # Report type conflicts
        for var_name, type_infos in self._type_conflicts:
            types = [t.inferred_type for t in type_infos if t.inferred_type != "Unknown"]
            lines = [t.line for t in type_infos]

            self._add_result(
                node=ast.parse("x").body[0],  # Dummy node
                pattern=f"Type conflict: {var_name}",
                context=f"Variable assigned as: {', '.join(set(types))}",
                category="type_conflict",
                metadata={
                    "variable": var_name,
                    "types": list(set(types)),
                    "lines": lines,
                },
            )
            # Fix the line number for the result
            self._results[-1].line = type_infos[0].line

        # Report functions with inferred types
        for func_name, func_info in self._functions.items():
            if func_info.return_type and func_info.return_type != "None":
                category = "function_type"
                pattern = f"Function: {func_info.name}"
                context = f"Returns: {func_info.return_type}"
                if func_info.params:
                    annotated = [f"{k}: {v}" for k, v in func_info.params.items() if v != "Unknown"]
                    if annotated:
                        context += f", Params: {', '.join(annotated)}"

                self._results.append(
                    AnalysisResult(
                        file=self._current_file,
                        line=func_info.line,
                        column=0,
                        pattern=pattern,
                        context=context,
                        category=category,
                        metadata={
                            "function": func_name,
                            "params": func_info.params,
                            "return_type": func_info.return_type,
                        },
                    )
                )

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        total_vars = len(self._variables)
        typed_vars = sum(
            1 for infos in self._variables.values() if any(t.inferred_type != "Unknown" for t in infos)
        )

        return {
            "total_variables": total_vars,
            "typed_variables": typed_vars,
            "total_functions": len(self._functions),
            "type_conflicts": len(self._type_conflicts),
            "by_category": {
                "type_conflict": len(self._type_conflicts),
                "function_type": len(self._functions),
            },
        }

    def get_variable_types(self) -> dict[str, list[TypeInfo]]:
        """Get all inferred variable types."""
        return dict(self._variables)

    def get_function_types(self) -> dict[str, FunctionTypeInfo]:
        """Get all function type information."""
        return self._functions

    def get_type_conflicts(self) -> list[tuple[str, list[TypeInfo]]]:
        """Get all detected type conflicts."""
        return self._type_conflicts
