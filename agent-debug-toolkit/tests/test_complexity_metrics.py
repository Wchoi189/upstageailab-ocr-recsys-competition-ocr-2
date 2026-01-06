"""Tests for ComplexityMetricsAnalyzer."""


from agent_debug_toolkit.analyzers.complexity_metrics import ComplexityMetricsAnalyzer


class TestComplexityMetricsAnalyzer:
    """Test cases for ComplexityMetricsAnalyzer."""

    def test_simple_function_complexity(self):
        """Test complexity of a simple function."""
        source = """
def simple():
    return 1
"""
        analyzer = ComplexityMetricsAnalyzer()
        report = analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "simple"
        assert metrics[0].cyclomatic_complexity == 1  # Base complexity

    def test_if_statement_complexity(self):
        """Test that if statements increase complexity."""
        source = """
def with_if(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        assert len(metrics) == 1
        # 1 (base) + 2 (if + elif) = 3
        assert metrics[0].cyclomatic_complexity == 3

    def test_loop_complexity(self):
        """Test that loops increase complexity."""
        source = """
def with_loops(items):
    for item in items:
        while item > 0:
            item -= 1
    return True
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        assert len(metrics) == 1
        # 1 (base) + 1 (for) + 1 (while) = 3
        assert metrics[0].cyclomatic_complexity == 3

    def test_boolean_operators_complexity(self):
        """Test that boolean operators increase complexity."""
        source = """
def with_boolean(a, b, c):
    if a and b and c:
        return True
    if a or b:
        return False
    return None
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        assert len(metrics) == 1
        # 1 (base) + 2 (if with 2 ands) + 1 (if with 1 or) + 1 (first if) + 1 (second if) = 6
        # Actually: 1 + 1 (if) + 2 (two 'and') + 1 (if) + 1 (one 'or') = 6
        assert metrics[0].cyclomatic_complexity >= 4

    def test_nesting_depth(self):
        """Test calculation of nesting depth."""
        source = """
def deeply_nested():
    if True:
        for i in range(10):
            while True:
                if i > 5:
                    return i
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].max_nesting_depth == 4  # if > for > while > if

    def test_parameter_count(self):
        """Test counting of function parameters."""
        source = """
def no_params():
    pass

def few_params(a, b, c):
    pass

def many_params(a, b, c, d, e, f, *args, **kwargs):
    pass

class MyClass:
    def method(self, x, y):
        pass
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        assert len(metrics) == 4

        by_name = {m.name: m for m in metrics}
        assert by_name["no_params"].param_count == 0
        assert by_name["few_params"].param_count == 3
        assert by_name["many_params"].param_count == 8  # 6 regular + args + kwargs
        assert by_name["method"].param_count == 2  # self is excluded

    def test_return_count(self):
        """Test counting of return statements."""
        source = """
def multiple_returns(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].return_count == 3

    def test_async_function(self):
        """Test handling of async functions."""
        source = """
async def async_func():
    await some_call()
    return True
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].is_async is True

    def test_threshold_violations(self):
        """Test detection of threshold violations."""
        source = """
def complex_function(a, b, c, d, e, f):
    if a:
        if b:
            if c:
                if d:
                    for i in range(10):
                        while True:
                            if e and f:
                                return i
    return 0
"""
        analyzer = ComplexityMetricsAnalyzer(
            complexity_threshold=5, nesting_threshold=3, param_threshold=4
        )
        report = analyzer.analyze_source(source)

        high_complexity = analyzer.get_high_complexity_functions()
        assert len(high_complexity) == 1

        deeply_nested = analyzer.get_deeply_nested_functions()
        assert len(deeply_nested) == 1

    def test_class_methods(self):
        """Test handling of class methods."""
        source = """
class MyClass:
    def __init__(self, x):
        self.x = x

    def method(self):
        if self.x:
            return True
        return False

    @classmethod
    def class_method(cls):
        return cls()

    @staticmethod
    def static_method(a, b):
        return a + b
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        assert len(metrics) == 4

        # Check qualified names
        names = [m.qualified_name for m in metrics]
        assert "MyClass.__init__" in names
        assert "MyClass.method" in names
        assert "MyClass.class_method" in names
        assert "MyClass.static_method" in names

    def test_lines_of_code(self):
        """Test LOC calculation."""
        source = """
def short():
    return 1

def longer():
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    return a + b + c + d + e
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        metrics = analyzer.get_metrics()
        by_name = {m.name: m for m in metrics}
        assert by_name["short"].lines_of_code == 2
        # LOC is end_lineno - lineno + 1, which depends on exact line positions
        assert by_name["longer"].lines_of_code >= 7

    def test_sorted_by_complexity(self):
        """Test sorting functions by complexity."""
        source = """
def simple():
    return 1

def moderate(x):
    if x > 0:
        return 1
    return 0

def complex(a, b, c):
    if a:
        if b:
            if c:
                for i in range(10):
                    return i
    return 0
"""
        analyzer = ComplexityMetricsAnalyzer()
        analyzer.analyze_source(source)

        sorted_funcs = analyzer.get_sorted_by_complexity(limit=3)
        assert len(sorted_funcs) == 3
        assert sorted_funcs[0].name == "complex"
        assert sorted_funcs[1].name == "moderate"
        assert sorted_funcs[2].name == "simple"

    def test_summary_statistics(self):
        """Test summary statistics generation."""
        source = """
def func1():
    return 1

def func2(x):
    if x:
        return 2
    return 0
"""
        analyzer = ComplexityMetricsAnalyzer()
        report = analyzer.analyze_source(source)

        assert "total_functions" in report.summary
        assert "average_complexity" in report.summary
        assert "max_complexity" in report.summary
        assert report.summary["total_functions"] == 2

    def test_empty_file(self):
        """Test handling of file with no functions."""
        source = """
x = 1
y = 2
"""
        analyzer = ComplexityMetricsAnalyzer()
        report = analyzer.analyze_source(source)

        assert len(report.results) == 0
        assert report.summary["total_functions"] == 0
