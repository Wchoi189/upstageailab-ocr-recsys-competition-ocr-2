"""Tests for ImportTracker."""


from agent_debug_toolkit.analyzers.import_tracker import ImportTracker


class TestImportTracker:
    """Test cases for ImportTracker."""

    def test_stdlib_imports(self):
        """Test detection of standard library imports."""
        source = """
import os
import json
from pathlib import Path
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        stdlib_results = [r for r in report.results if r.category == "stdlib"]
        assert len(stdlib_results) == 3

    def test_third_party_imports(self):
        """Test detection of third-party imports."""
        source = """
import numpy
import pandas as pd
from torch import nn
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        third_party_results = [r for r in report.results if r.category == "third_party"]
        assert len(third_party_results) == 3

    def test_local_imports(self):
        """Test detection of local/project imports."""
        source = """
import mymodule
from mypackage.utils import helper
from . import sibling
from ..parent import config
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        local_results = [r for r in report.results if r.category == "local"]
        # Relative imports are always local
        assert len(local_results) >= 2

    def test_dynamic_imports(self):
        """Test detection of dynamic imports."""
        source = """
import importlib

module = __import__('dynamic_module')
plugin = importlib.import_module('plugins.custom')
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        dynamic_results = [r for r in report.results if r.category == "dynamic"]
        assert len(dynamic_results) == 2

    def test_import_with_alias(self):
        """Test import with alias tracking."""
        source = """
import numpy as np
from pandas import DataFrame as DF
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        # Check that aliases are tracked in metadata
        for result in report.results:
            if "numpy" in result.pattern:
                assert result.metadata.get("alias") == "np"
                assert result.metadata.get("local_name") == "np"
            if "DataFrame" in result.pattern:
                assert result.metadata.get("alias") == "DF"
                assert result.metadata.get("local_name") == "DF"

    def test_unused_import_detection(self):
        """Test detection of potentially unused imports."""
        source = """
import os
import unused_module
from typing import Optional

def main():
    print(os.getcwd())
    x: Optional[str] = None
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        unused = analyzer.get_unused_imports()
        unused_names = [u["local_name"] for u in unused]
        assert "unused_module" in unused_names
        # os and Optional should be used
        assert "os" not in unused_names

    def test_imports_by_category(self):
        """Test grouping imports by category."""
        source = """
import os
import numpy
import mymodule
"""
        analyzer = ImportTracker()
        analyzer.analyze_source(source)

        by_category = analyzer.get_imports_by_category()
        assert "stdlib" in by_category
        assert "third_party" in by_category
        assert "local" in by_category
        assert "os" in by_category["stdlib"]
        assert "numpy" in by_category["third_party"]
        assert "mymodule" in by_category["local"]

    def test_relative_import_levels(self):
        """Test tracking of relative import levels."""
        source = """
from . import sibling
from .. import parent
from ...grandparent import config
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        for result in report.results:
            assert result.category == "local"  # All relative imports are local
            level = result.metadata.get("level", 0)
            if "sibling" in result.pattern:
                assert level == 1
            elif "...grandparent" in result.pattern:
                assert level == 3
            elif ".." in result.pattern:
                assert level >= 2

    def test_summary_statistics(self):
        """Test summary statistics generation."""
        source = """
import os
import json
import numpy
from mypackage import utils
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        assert "total_imports" in report.summary
        assert "stdlib_count" in report.summary
        assert "third_party_count" in report.summary
        assert "local_count" in report.summary
        assert report.summary["stdlib_count"] == 2
        assert report.summary["third_party_count"] == 1
        assert report.summary["local_count"] == 1

    def test_empty_file(self):
        """Test handling of file with no imports."""
        source = """
def main():
    print("Hello")
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        assert len(report.results) == 0
        assert report.summary["total_imports"] == 0

    def test_star_import(self):
        """Test handling of star imports."""
        source = """
from typing import *
from os.path import *
"""
        analyzer = ImportTracker()
        report = analyzer.analyze_source(source)

        star_imports = [r for r in report.results if "*" in r.pattern]
        assert len(star_imports) == 2
