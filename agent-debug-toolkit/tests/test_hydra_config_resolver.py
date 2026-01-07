import os
import yaml # type: ignore
import pytest
import tempfile
from agent_debug_toolkit.precomputes.symbol_table import SymbolTable
from agent_debug_toolkit.analyzers.hydra_config_resolver import HydraConfigResolver

class TestHydraConfigResolver:
    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create code
            src = os.path.join(tmpdir, "src")
            pkg = os.path.join(src, "pkg")
            os.makedirs(pkg)

            with open(os.path.join(pkg, "__init__.py"), "w") as f:
                f.write("")
            with open(os.path.join(pkg, "mod.py"), "w") as f:
                f.write("class MyClass:\n    pass\n")

            # Create configs
            conf = os.path.join(tmpdir, "conf")
            os.makedirs(conf)

            # static.yaml
            with open(os.path.join(conf, "static.yaml"), "w") as f:
                yaml.dump({"_target_": "pkg.mod.MyClass"}, f)

            # dynamic.yaml
            with open(os.path.join(conf, "dynamic.yaml"), "w") as f:
                yaml.dump({"_target_": "${ctx}.MyClass"}, f)

            # relative.yaml
            with open(os.path.join(conf, "relative.yaml"), "w") as f:
                yaml.dump({"_target_": ".MyClass"}, f)

            # unresolved.yaml
            with open(os.path.join(conf, "unresolved.yaml"), "w") as f:
                yaml.dump({"_target_": "pkg.mod.NonExistent"}, f)

            yield tmpdir

    def test_resolution(self, workspace):
        src_path = os.path.join(workspace, "src")
        conf_path = os.path.join(workspace, "conf")

        # Build symbol table
        table = SymbolTable(root_path=src_path, module_root=src_path)
        table.build()

        resolver = HydraConfigResolver(symbol_table=table)
        mappings = resolver.analyze_config_dir(conf_path)

        assert len(mappings) == 4

        results = {os.path.basename(m.config_file): m.resolution for m in mappings}

        # Test Static
        res = results["static.yaml"]
        assert res.status == "RESOLVED"
        assert res.symbol is not None
        assert res.target == "pkg.mod.MyClass"

        # Test Dynamic
        res = results["dynamic.yaml"]
        assert res.status == "UNRESOLVED_DYNAMIC"

        # Test Relative
        res = results["relative.yaml"]
        assert res.status == "UNRESOLVED_RELATIVE"

        # Test Not Found
        res = results["unresolved.yaml"]
        assert res.status == "NOT_FOUND"
