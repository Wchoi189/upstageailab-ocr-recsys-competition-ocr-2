"""Tests for TypeInferenceAnalyzer."""

from agent_debug_toolkit.analyzers.type_inference import TypeInferenceAnalyzer


class TestTypeInferenceAnalyzer:
    """Test cases for TypeInferenceAnalyzer."""

    def test_literal_int_assignment(self):
        """Test inference of integer literals."""
        source = """
x = 5
y = 42
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        variables = analyzer.get_variable_types()
        assert "x" in variables
        assert any(t.inferred_type == "int" for t in variables["x"])

    def test_literal_str_assignment(self):
        """Test inference of string literals."""
        source = """
name = "hello"
message = 'world'
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        variables = analyzer.get_variable_types()
        assert "name" in variables
        assert any(t.inferred_type == "str" for t in variables["name"])

    def test_list_assignment(self):
        """Test inference of list types."""
        source = """
items = [1, 2, 3]
empty_list = []
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        variables = analyzer.get_variable_types()
        assert "items" in variables
        assert any("list" in t.inferred_type for t in variables["items"])

    def test_dict_assignment(self):
        """Test inference of dict types."""
        source = """
data = {"key": "value"}
empty = {}
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        variables = analyzer.get_variable_types()
        assert "data" in variables
        assert any(t.inferred_type == "dict" for t in variables["data"])

    def test_type_annotation(self):
        """Test explicit type annotations."""
        source = """
x: int = 5
name: str = "hello"
items: list[int] = [1, 2, 3]
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        variables = analyzer.get_variable_types()
        assert "x" in variables
        assert any(t.inferred_type == "int" and t.source == "annotation" for t in variables["x"])

    def test_type_conflict_detection(self):
        """Test detection of type conflicts."""
        source = """
x = 5
x = "hello"
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        conflicts = analyzer.get_type_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0][0] == "x"

    def test_function_return_type_annotation(self):
        """Test function return type from annotation."""
        source = """
def get_value() -> int:
    return 42
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        functions = analyzer.get_function_types()
        assert "get_value" in functions
        assert functions["get_value"].return_type == "int"

    def test_function_return_type_inference(self):
        """Test function return type inference."""
        source = """
def compute():
    return 42
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        functions = analyzer.get_function_types()
        assert "compute" in functions
        assert functions["compute"].return_type == "int"

    def test_function_parameter_annotations(self):
        """Test function parameter type annotations."""
        source = """
def process(x: int, name: str) -> bool:
    return x > 0
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        functions = analyzer.get_function_types()
        assert "process" in functions
        assert functions["process"].params.get("x") == "int"
        assert functions["process"].params.get("name") == "str"

    def test_class_method_types(self):
        """Test type inference for class methods."""
        source = """
class MyClass:
    def get_value(self) -> int:
        return self.x

    def set_value(self, value: int) -> None:
        self.x = value
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        functions = analyzer.get_function_types()
        assert "MyClass.get_value" in functions
        assert "MyClass.set_value" in functions

    def test_multiple_return_types(self):
        """Test inference with multiple return types."""
        source = """
def maybe_get(condition):
    if condition:
        return 42
    return None
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        functions = analyzer.get_function_types()
        assert "maybe_get" in functions
        # Should detect multiple return types
        return_type = functions["maybe_get"].return_type
        assert return_type is not None
        assert "int" in return_type or "None" in return_type

    def test_no_type_conflicts_for_same_type(self):
        """Test no conflict when same type is assigned."""
        source = """
x = 1
x = 2
x = 3
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        conflicts = analyzer.get_type_conflicts()
        assert len(conflicts) == 0

    def test_summary_statistics(self):
        """Test summary contains expected statistics."""
        source = """
x: int = 5
y = "hello"

def func() -> bool:
    return True
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        assert "total_variables" in report.summary
        assert "total_functions" in report.summary
        assert "type_conflicts" in report.summary

    def test_empty_file(self):
        """Test handling of empty file."""
        source = """
# Just a comment
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        assert report.summary["total_variables"] == 0
        assert report.summary["total_functions"] == 0

    def test_boolean_expression_type(self):
        """Test boolean expression type inference."""
        source = """
result = x > 5
is_valid = a and b
"""
        analyzer = TypeInferenceAnalyzer()
        report = analyzer.analyze_source(source)

        variables = analyzer.get_variable_types()
        assert "result" in variables
        assert any(t.inferred_type == "bool" for t in variables["result"])
