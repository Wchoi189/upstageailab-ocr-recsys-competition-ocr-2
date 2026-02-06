import os
import tempfile
import pytest
from agent_debug_toolkit.precomputes.symbol_table import SymbolTable

class TestSymbolTable:
    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create src/pkg structure
            src = os.path.join(tmpdir, "src")
            pkg = os.path.join(src, "pkg")
            os.makedirs(pkg)

            # src/pkg/__init__.py
            with open(os.path.join(pkg, "__init__.py"), "w") as f:
                f.write("")

            # src/pkg/module.py
            with open(os.path.join(pkg, "module.py"), "w") as f:
                f.write("class MyClass:\n    pass\n\ndef my_func():\n    pass\n")

            # src/pkg/nested.py
            with open(os.path.join(pkg, "nested.py"), "w") as f:
                f.write("class Outer:\n    class Inner:\n        def method(self):\n            pass\n")

            yield tmpdir

    def test_basic_lookup(self, workspace):
        src_path = os.path.join(workspace, "src")
        table = SymbolTable(root_path=src_path, module_root=src_path)
        table.build()

        # Test class lookup
        sym = table.lookup("pkg.module.MyClass")
        assert sym is not None
        assert sym.name == "MyClass"
        assert sym.kind == "class"
        assert sym.full_name == "pkg.module.MyClass"

        # Test function lookup
        sym = table.lookup("pkg.module.my_func")
        assert sym is not None
        assert sym.name == "my_func"
        assert sym.kind == "function"

    def test_nested_lookup(self, workspace):
        src_path = os.path.join(workspace, "src")
        table = SymbolTable(root_path=src_path, module_root=src_path)
        table.build()

        # Outer class
        assert table.lookup("pkg.nested.Outer") is not None

        # Inner class
        inner = table.lookup("pkg.nested.Outer.Inner")
        assert inner is not None
        assert inner.name == "Inner"
        assert inner.full_name == "pkg.nested.Outer.Inner"

        # Method in Inner class
        method = table.lookup("pkg.nested.Outer.Inner.method")
        assert method is not None
        assert method.kind == "function"
