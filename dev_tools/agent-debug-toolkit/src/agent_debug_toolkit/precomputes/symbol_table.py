import ast
import os
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class SymbolDefinition:
    name: str          # "ResNet"
    full_name: str     # "ocr.core.models.encoders.ResNet"
    file_path: str     # "/abs/path/ocr/models/encoders.py"
    line_number: int   # 45
    kind: str          # "class" | "function"

class SymbolTable:
    def __init__(self, root_path: str, module_root: str):
        """
        Args:
            root_path: Absolute path to the directory to scan.
            module_root: Absolute path to the root of the python package (e.g. src/).
                         Used to resolve the module path for files found in root_path.
        """
        self.root_path = root_path
        self.module_root = module_root
        self._symbols: Dict[str, SymbolDefinition] = {}

    def build(self):
        """Builds the symbol table by scanning the directory."""
        self._symbols.clear()
        ignore_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "site-packages", "outputs", "logs", "data"}

        for root, dirs, files in os.walk(self.root_path):
            # Modify dirs in-place to prune traversal
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]

            for file in files:
                if file.endswith(".py"):
                    self._process_file(os.path.join(root, file))

    def lookup(self, full_name: str) -> Optional[SymbolDefinition]:
        """Look up a symbol by its fully qualified name."""
        return self._symbols.get(full_name)

    def _process_file(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)

            module_name = self._get_module_name(file_path)
            if not module_name:
                return

            self._visit_node(tree, module_name, file_path)

        except Exception:
            # Silently ignore parse errors
            pass

    def _visit_node(self, node: ast.AST, module_name: str, file_path: str, parent_scope: str = ""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
                name = child.name

                # Construct qualified name (e.g., ParentClass.ChildClass)
                if parent_scope:
                    qualified_name = f"{parent_scope}.{name}"
                else:
                    qualified_name = name

                full_name = f"{module_name}.{qualified_name}"
                kind = "class" if isinstance(child, ast.ClassDef) else "function"

                # Store definitions
                self._symbols[full_name] = SymbolDefinition(
                    name=name,
                    full_name=full_name,
                    file_path=file_path,
                    line_number=child.lineno,
                    kind=kind
                )

                # Recurse with updated scope
                self._visit_node(child, module_name, file_path, qualified_name)

    def _get_module_name(self, file_path: str) -> Optional[str]:
        rel_path = os.path.relpath(file_path, self.module_root)
        if rel_path.startswith(".."):
            return None

        base = os.path.splitext(rel_path)[0]
        # Replace path separators with dots
        return base.replace(os.sep, ".")
