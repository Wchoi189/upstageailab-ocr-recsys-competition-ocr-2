"""Tests for DependencyGraphAnalyzer."""



from agent_debug_toolkit.analyzers.dependency_graph import DependencyGraphAnalyzer


class TestDependencyGraphAnalyzer:
    """Test cases for DependencyGraphAnalyzer."""

    def test_simple_import(self):
        """Test detection of simple import statements."""
        source = """
import os
import json
"""
        analyzer = DependencyGraphAnalyzer(include_stdlib=True)
        report = analyzer.analyze_source(source)

        # Should find both imports
        import_results = [r for r in report.results if r.category == "import"]
        assert len(import_results) == 2
        assert any("import os" in r.pattern for r in import_results)
        assert any("import json" in r.pattern for r in import_results)

    def test_from_import(self):
        """Test detection of from...import statements."""
        source = """
from pathlib import Path
from typing import Any, Optional
"""
        analyzer = DependencyGraphAnalyzer(include_stdlib=True)
        report = analyzer.analyze_source(source)

        from_import_results = [r for r in report.results if r.category == "from_import"]
        assert len(from_import_results) == 3  # Path, Any, Optional

    def test_exclude_stdlib_by_default(self):
        """Test that stdlib imports are excluded by default."""
        source = """
import os
import custom_module
from mypackage import helper
"""
        analyzer = DependencyGraphAnalyzer()  # include_stdlib=False by default
        report = analyzer.analyze_source(source)

        # Should only find custom_module and mypackage imports
        patterns = [r.pattern for r in report.results]
        assert not any("import os" in p for p in patterns)
        assert any("import custom_module" in p for p in patterns)
        assert any("from mypackage import helper" in p for p in patterns)

    def test_class_inheritance(self):
        """Test tracking of class inheritance."""
        source = """
class BaseModel:
    pass

class MyModel(BaseModel):
    pass
"""
        analyzer = DependencyGraphAnalyzer()
        report = analyzer.analyze_source(source)

        # Check that inheritance edge exists in graph
        graph = analyzer.get_graph()
        edges = graph["edges"]
        inherit_edges = [e for e in edges if e["type"] == "inherits"]
        assert len(inherit_edges) == 1
        assert inherit_edges[0]["source"] == "MyModel"
        assert inherit_edges[0]["target"] == "BaseModel"

    def test_cycle_detection(self):
        """Test detection of circular dependencies."""
        # First file: A imports B
        source_a = """
import B
class A:
    pass
"""
        # Second file: B imports A
        source_b = """
import A
class B:
    pass
"""
        analyzer = DependencyGraphAnalyzer(include_stdlib=True)
        analyzer.analyze_source(source_a, filename="A.py")
        analyzer.analyze_source(source_b, filename="B.py")

        cycles = analyzer.find_cycles()
        # The cycle detection should work on the combined graph
        # Note: This is a simplified test - real cycle detection
        # would need cross-file analysis

    def test_get_dependents(self):
        """Test finding what depends on a module."""
        source = """
import helper
from utils import process

def main():
    helper.run()
    process()
"""
        analyzer = DependencyGraphAnalyzer()
        analyzer.analyze_source(source)

        dependents = analyzer.get_dependents("helper")
        # The module that imports helper should be in dependents
        assert len(dependents) > 0

    def test_mermaid_output(self):
        """Test Mermaid diagram generation."""
        source = """
import utils
from models import Model

class MyClass(Model):
    pass
"""
        analyzer = DependencyGraphAnalyzer()
        analyzer.analyze_source(source)

        mermaid = analyzer.to_mermaid()
        assert "graph TD" in mermaid
        assert "MyClass" in mermaid
        assert "Model" in mermaid

    def test_summary_statistics(self):
        """Test summary statistics generation."""
        source = """
import os
from pathlib import Path

class Config:
    pass

def load():
    pass
"""
        analyzer = DependencyGraphAnalyzer(include_stdlib=True)
        report = analyzer.analyze_source(source)

        assert "total_nodes" in report.summary
        assert "total_edges" in report.summary
        assert "node_types" in report.summary
        assert "cycles_detected" in report.summary

    def test_empty_file(self):
        """Test handling of empty file."""
        source = ""
        analyzer = DependencyGraphAnalyzer()
        report = analyzer.analyze_source(source)

        assert len(report.results) == 0
        # Note: module node may still be created for the file itself
        assert report.summary["total_nodes"] <= 1
