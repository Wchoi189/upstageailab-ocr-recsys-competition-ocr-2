"""
CLI interface for Agent Debug Toolkit.

Provides commands for analyzing Python code for configuration patterns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Check if typer is available
try:
    import typer
    from typer import Argument, Option

    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False
    # Create minimal fallback
    typer = None  # type: ignore


def create_app() -> "typer.Typer":
    """Create the CLI application."""
    if not HAS_TYPER:
        raise ImportError(
            "Typer is required for CLI. Install with: pip install agent-debug-toolkit[cli]"
        )

    app = typer.Typer(
        name="adt",
        help="Agent Debug Toolkit - AST-based analysis for configuration debugging",
        add_completion=False,
    )

    @app.command("analyze-config")
    def analyze_config(
        path: str = Argument(..., help="Path to Python file or directory to analyze"),
        component: Optional[str] = Option(
            None, "--component", "-c", help="Filter by component name"
        ),
        output: str = Option(
            "json", "--output", "-o", help="Output format: json, markdown, or text"
        ),
        recursive: bool = Option(
            True, "--recursive/--no-recursive", "-r", help="Recurse into directories"
        ),
    ) -> None:
        """
        Analyze configuration access patterns in Python code.

        Finds cfg.X, self.cfg.X, config['key'] patterns and categorizes them.
        """
        from agent_debug_toolkit.analyzers.config_access import ConfigAccessAnalyzer

        analyzer = ConfigAccessAnalyzer()
        target = Path(path)

        if target.is_file():
            report = analyzer.analyze_file(target)
        elif target.is_dir():
            report = analyzer.analyze_directory(target, recursive=recursive)
        else:
            typer.echo(f"Error: Path not found: {path}", err=True)
            raise typer.Exit(1)

        # Filter by component if specified
        if component:
            report.results = report.filter_by_component(component)
            report.summary["filtered_by"] = component

        _output_report(report, output)

    @app.command("trace-merges")
    def trace_merges(
        file: str = Argument(..., help="Path to Python file to analyze"),
        output: str = Option(
            "markdown", "--output", "-o", help="Output format: json, markdown, or text"
        ),
        explain: bool = Option(
            True, "--explain/--no-explain", help="Include precedence explanation"
        ),
    ) -> None:
        """
        Trace OmegaConf.merge() operations and their precedence.

        Essential for debugging configuration override issues.
        """
        from agent_debug_toolkit.analyzers.merge_order import MergeOrderTracker

        target = Path(file)
        if not target.is_file():
            typer.echo(f"Error: File not found: {file}", err=True)
            raise typer.Exit(1)

        analyzer = MergeOrderTracker()
        report = analyzer.analyze_file(target)

        if output == "markdown" and explain:
            # Add precedence explanation to the output
            typer.echo(analyzer.explain_precedence())
            typer.echo("\n---\n")

        _output_report(report, output)

    @app.command("find-hydra")
    def find_hydra(
        path: str = Argument(..., help="Path to Python file or directory to analyze"),
        output: str = Option(
            "json", "--output", "-o", help="Output format: json, markdown, or text"
        ),
        recursive: bool = Option(
            True, "--recursive/--no-recursive", "-r", help="Recurse into directories"
        ),
    ) -> None:
        """
        Find Hydra framework usage patterns.

        Detects @hydra.main decorators, instantiate() calls, and config patterns.
        """
        from agent_debug_toolkit.analyzers.hydra_usage import HydraUsageAnalyzer

        analyzer = HydraUsageAnalyzer()
        target = Path(path)

        if target.is_file():
            report = analyzer.analyze_file(target)
        elif target.is_dir():
            report = analyzer.analyze_directory(target, recursive=recursive)
        else:
            typer.echo(f"Error: Path not found: {path}", err=True)
            raise typer.Exit(1)

        _output_report(report, output)

    @app.command("find-instantiations")
    def find_instantiations(
        path: str = Argument(..., help="Path to Python file or directory to analyze"),
        component: Optional[str] = Option(
            None, "--component", "-c", help="Filter by component type"
        ),
        output: str = Option(
            "json", "--output", "-o", help="Output format: json, markdown, or text"
        ),
        recursive: bool = Option(
            True, "--recursive/--no-recursive", "-r", help="Recurse into directories"
        ),
    ) -> None:
        """
        Find component instantiation sites.

        Tracks get_*_by_cfg(), registry.create(), and direct class instantiation.
        """
        from agent_debug_toolkit.analyzers.instantiation import ComponentInstantiationTracker

        analyzer = ComponentInstantiationTracker()
        target = Path(path)

        if target.is_file():
            report = analyzer.analyze_file(target)
        elif target.is_dir():
            report = analyzer.analyze_directory(target, recursive=recursive)
        else:
            typer.echo(f"Error: Path not found: {path}", err=True)
            raise typer.Exit(1)

        # Filter by component if specified
        if component:
            report.results = [
                r
                for r in report.results
                if component.lower() in (r.metadata.get("component_type") or "").lower()
                or component.lower() in r.pattern.lower()
            ]
            report.summary["filtered_by"] = component

        _output_report(report, output)

    @app.command("analyze-dependencies")
    def analyze_dependencies(
        path: str = Argument(..., help="Path to Python file or directory to analyze"),
        output: str = Option(
            "json", "--output", "-o", help="Output format: json, markdown, or mermaid"
        ),
        include_stdlib: bool = Option(
            False, "--include-stdlib", help="Include standard library imports"
        ),
        detect_cycles: bool = Option(
            True, "--detect-cycles/--no-cycles", help="Detect circular dependencies"
        ),
        recursive: bool = Option(
            True, "--recursive/--no-recursive", "-r", help="Recurse into directories"
        ),
    ) -> None:
        """
        Build module dependency graph.

        Traces imports, class inheritance, and function calls to map dependencies.
        """
        from agent_debug_toolkit.analyzers.dependency_graph import DependencyGraphAnalyzer

        analyzer = DependencyGraphAnalyzer(include_stdlib=include_stdlib)
        target = Path(path)

        if target.is_file():
            report = analyzer.analyze_file(target)
        elif target.is_dir():
            report = analyzer.analyze_directory(target, recursive=recursive)
        else:
            typer.echo(f"Error: Path not found: {path}", err=True)
            raise typer.Exit(1)

        if output == "mermaid":
            typer.echo(analyzer.to_mermaid())
        else:
            _output_report(report, output)

    @app.command("analyze-imports")
    def analyze_imports(
        path: str = Argument(..., help="Path to Python file or directory to analyze"),
        output: str = Option(
            "json", "--output", "-o", help="Output format: json, markdown, or text"
        ),
        show_unused: bool = Option(
            True, "--show-unused/--no-unused", help="Show potentially unused imports"
        ),
        recursive: bool = Option(
            True, "--recursive/--no-recursive", "-r", help="Recurse into directories"
        ),
    ) -> None:
        """
        Categorize and analyze imports.

        Groups imports by stdlib, third-party, and local. Detects potentially unused.
        """
        from agent_debug_toolkit.analyzers.import_tracker import ImportTracker

        analyzer = ImportTracker()
        target = Path(path)

        if target.is_file():
            report = analyzer.analyze_file(target)
        elif target.is_dir():
            report = analyzer.analyze_directory(target, recursive=recursive)
        else:
            typer.echo(f"Error: Path not found: {path}", err=True)
            raise typer.Exit(1)

        if not show_unused and "unused_imports" in report.summary:
            del report.summary["unused_imports"]

        _output_report(report, output)

    @app.command("analyze-complexity")
    def analyze_complexity(
        path: str = Argument(..., help="Path to Python file or directory to analyze"),
        output: str = Option(
            "json", "--output", "-o", help="Output format: json, markdown, or text"
        ),
        threshold: int = Option(
            10, "--threshold", "-t", help="Complexity threshold for warnings"
        ),
        sort_by: str = Option(
            "complexity", "--sort-by", "-s", help="Sort by: complexity, nesting, loc"
        ),
        recursive: bool = Option(
            True, "--recursive/--no-recursive", "-r", help="Recurse into directories"
        ),
    ) -> None:
        """
        Calculate code complexity metrics.

        Measures cyclomatic complexity, nesting depth, LOC, and parameter count.
        """
        from agent_debug_toolkit.analyzers.complexity_metrics import ComplexityMetricsAnalyzer

        analyzer = ComplexityMetricsAnalyzer(complexity_threshold=threshold)
        target = Path(path)

        if target.is_file():
            report = analyzer.analyze_file(target)
        elif target.is_dir():
            report = analyzer.analyze_directory(target, recursive=recursive)
        else:
            typer.echo(f"Error: Path not found: {path}", err=True)
            raise typer.Exit(1)

        _output_report(report, output)

    @app.command("full-analysis")
    def full_analysis(
        path: str = Argument(..., help="Path to Python file or directory to analyze"),
        output: str = Option(
            "markdown", "--output", "-o", help="Output format: json, markdown, or text"
        ),
        recursive: bool = Option(
            True, "--recursive/--no-recursive", "-r", help="Recurse into directories"
        ),
    ) -> None:
        """
        Run all analyzers and produce a comprehensive report.

        Combines config access, merge tracking, Hydra usage, instantiation,
        dependency graph, import tracking, and complexity analysis.
        """
        from agent_debug_toolkit.analyzers.config_access import ConfigAccessAnalyzer
        from agent_debug_toolkit.analyzers.merge_order import MergeOrderTracker
        from agent_debug_toolkit.analyzers.hydra_usage import HydraUsageAnalyzer
        from agent_debug_toolkit.analyzers.instantiation import ComponentInstantiationTracker
        from agent_debug_toolkit.analyzers.dependency_graph import DependencyGraphAnalyzer
        from agent_debug_toolkit.analyzers.import_tracker import ImportTracker
        from agent_debug_toolkit.analyzers.complexity_metrics import ComplexityMetricsAnalyzer

        target = Path(path)
        is_file = target.is_file()
        is_dir = target.is_dir()

        if not is_file and not is_dir:
            typer.echo(f"Error: Path not found: {path}", err=True)
            raise typer.Exit(1)

        analyzers = [
            ConfigAccessAnalyzer(),
            MergeOrderTracker(),
            HydraUsageAnalyzer(),
            ComponentInstantiationTracker(),
            DependencyGraphAnalyzer(),
            ImportTracker(),
            ComplexityMetricsAnalyzer(),
        ]

        all_reports = []
        for analyzer in analyzers:
            if is_file:
                report = analyzer.analyze_file(target)
            else:
                report = analyzer.analyze_directory(target, recursive=recursive)
            all_reports.append(report)

        if output == "json":
            combined = {"target": str(target), "reports": [r.to_dict() for r in all_reports]}
            typer.echo(json.dumps(combined, indent=2))
        else:
            for report in all_reports:
                typer.echo(report.to_markdown())
                typer.echo("\n---\n")

    @app.command("context-tree")
    def context_tree(
        path: str = Argument(..., help="Directory to analyze"),
        depth: int = Option(3, "--depth", "-d", help="Maximum directory depth to traverse"),
        output: str = Option(
            "markdown", "--output", "-o", help="Output format: json, markdown, or tree"
        ),
    ):
        """
        Generate annotated directory tree with semantic context.

        Extracts docstrings, exports, and key definitions to provide
        rich context for AI agents navigating the codebase.
        """
        from pathlib import Path

        path_obj = Path(path)
        if not path_obj.exists():
            typer.echo(f"Error: Path not found: {path}", err=True)
            raise typer.Exit(1)
        if not path_obj.is_dir():
            typer.echo(f"Error: Not a directory: {path}", err=True)
            raise typer.Exit(1)

        from agent_debug_toolkit.analyzers.context_tree import ContextTreeAnalyzer, format_tree_markdown

        analyzer = ContextTreeAnalyzer(max_depth=depth)
        report = analyzer.analyze_directory(path)

        if "error" in report.summary:
            typer.echo(f"Error: {report.summary['error']}", err=True)
            raise typer.Exit(1)

        if output == "markdown":
            typer.echo(format_tree_markdown(report))
        else:
            _output_report(report, output)

    @app.command("intelligent-search")
    def intelligent_search(
        path: str = Argument(..., help="Symbol name or qualified path to search for"),
        root: str = Option(".", "--root", "-r", help="Root directory to search within"),
        fuzzy: bool = Option(True, "--fuzzy/--no-fuzzy", help="Enable fuzzy matching for typos"),
        threshold: float = Option(0.6, "--threshold", "-t", help="Minimum similarity for fuzzy matches (0.0-1.0)"),
        output: str = Option(
            "markdown", "--output", "-o", help="Output format: json or markdown"
        ),
    ):
        """
        Search for symbols by name or qualified path.

        Supports:
        - Qualified path resolution (ocr.core.models.X → file location)
        - Reverse lookup (class name → all import paths)
        - Fuzzy matching for typos (default: enabled)
        """
        from pathlib import Path

        root_obj = Path(root)
        if not root_obj.exists():
            typer.echo(f"Error: Root path not found: {root}", err=True)
            raise typer.Exit(1)

        from agent_debug_toolkit.analyzers.intelligent_search import (
            IntelligentSearcher,
            format_search_results_markdown,
        )

        # Use current directory as module root (heuristic)
        module_root = root_obj.resolve()
        searcher = IntelligentSearcher(root_obj, module_root)

        results = searcher.search(path, fuzzy=fuzzy, threshold=threshold)

        if output == "json":
            import json
            typer.echo(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            typer.echo(format_search_results_markdown(results, path))

    @app.command("sg-search")
    def sg_search_cmd(
        pattern: str = Option(..., "--pattern", "-p", help="AST pattern to search for (e.g., 'isinstance($CFG, dict)')"),
        path: str = Option(".", "--path", help="Path to search (file or directory)"),
        lang: Optional[str] = Option(None, "--lang", "-l", help="Language (python, javascript, etc.)"),
        max_results: Optional[int] = Option(None, "--max", "-m", help="Maximum number of results"),
        output: str = Option("markdown", "--output", "-o", help="Output format: json or markdown"),
    ):
        """
        Search code using ast-grep structural patterns.

        Examples:
          adt sg-search --pattern "def $NAME($$$)" --lang python
          adt sg-search --pattern "isinstance($CFG, dict)" --path ocr/
          adt sg-search -p "function $NAME($$$)" --path src/ -l javascript
        """
        from agent_debug_toolkit.astgrep import sg_search

        report = sg_search(
            pattern=pattern,
            path=path,
            lang=lang,
            max_results=max_results,
            output_format=output,
        )

        if not report.success:
            typer.echo(f"Error: {report.message}", err=True)
            raise typer.Exit(1)

        if output == "markdown":
            typer.echo(report.to_markdown())
        else:
            typer.echo(report.to_json())

    @app.command("sg-lint")
    def sg_lint_cmd(
        path: str = Argument(..., help="Path to lint (file or directory)"),
        rule_file: Optional[str] = Option(None, "--rule-file", "-r", help="Path to YAML rule file"),
        rule: Optional[str] = Option(None, "--rule", help="Inline YAML rule"),
    ):
        """
        Run ast-grep lint rules against code.

        Requires either --rule-file or --rule to be specified.

        Examples:
          adt sg-lint . --rule-file rules/no-print.yaml
          adt sg-lint src/ --rule "rule: {pattern: 'print($$$)'}"
        """
        from agent_debug_toolkit.astgrep import sg_lint

        if not rule_file and not rule:
            typer.echo("Error: Either --rule-file or --rule must be specified", err=True)
            raise typer.Exit(1)

        report = sg_lint(path=path, rule=rule, rule_file=rule_file)

        if not report.success:
            typer.echo(f"Error: {report.message}", err=True)
            raise typer.Exit(1)

        typer.echo(report.to_json())

    @app.command("dump-ast")
    def dump_ast_cmd(
        code: str = Argument(..., help="Code snippet to parse"),
        lang: str = Option("python", "--lang", "-l", help="Language of the code"),
    ):
        """
        Dump the AST (Abstract Syntax Tree) for a code snippet.

        Useful for understanding code structure and debugging patterns.

        Examples:
          adt dump-ast "def hello(): pass" --lang python
          adt dump-ast "function foo() {}" --lang javascript
        """
        from agent_debug_toolkit.astgrep import dump_syntax_tree

        result = dump_syntax_tree(code=code, lang=lang)
        typer.echo(result)

    @app.command("ts-parse")
    def ts_parse_cmd(
        code: str = Argument(..., help="Code snippet to parse"),
        lang: str = Option("python", "--lang", "-l", help="Language of the code"),
        max_depth: int = Option(5, "--depth", "-d", help="Maximum AST depth to display"),
        output: str = Option("json", "--output", "-o", help="Output format: json"),
    ):
        """
        Parse code into an AST using tree-sitter.

        Provides detailed AST node information with line/column positions.

        Examples:
          adt ts-parse "class Foo: pass" --lang python
          adt ts-parse "const x = 1;" --lang javascript --depth 10
        """
        from agent_debug_toolkit.treesitter import parse_code

        report = parse_code(code=code, lang=lang, max_depth=max_depth)

        if not report.success:
            typer.echo(f"Error: {report.message}", err=True)
            raise typer.Exit(1)

        typer.echo(report.to_json())

    @app.command("ts-query")
    def ts_query_cmd(
        code: str = Argument(..., help="Code snippet to query"),
        query: str = Argument(..., help="Tree-sitter query in S-expression format"),
        lang: str = Option("python", "--lang", "-l", help="Language of the code"),
        max_results: int = Option(50, "--max", "-m", help="Maximum number of results"),
        output: str = Option("json", "--output", "-o", help="Output format: json"),
    ):
        """
        Run a tree-sitter query against code.

        Query syntax uses S-expressions to match AST patterns.

        Examples:
          adt ts-query "def foo(): pass" "(function_definition) @fn" --lang python
          adt ts-query "x = 1; y = 2" "(assignment) @assign" --lang python
        """
        from agent_debug_toolkit.treesitter import run_query

        report = run_query(
            code=code,
            query=query,
            lang=lang,
            max_results=max_results,
        )

        if not report.success:
            typer.echo(f"Error: {report.message}", err=True)
            raise typer.Exit(1)

        typer.echo(report.to_json())

    def _output_report(report, output_format: str) -> None:
        """Output a report in the specified format."""
        if output_format == "json":
            typer.echo(report.to_json())
        elif output_format == "markdown":
            typer.echo(report.to_markdown())
        else:  # text
            typer.echo(f"\n{report.analyzer_name} Results for {report.target_path}")
            typer.echo("=" * 60)
            for r in report.results:
                typer.echo(str(r))
            typer.echo(f"\nTotal: {len(report.results)} findings")

    return app



# Create the app instance for entry point
app = create_app() if HAS_TYPER else None


def main() -> None:
    """Entry point for the CLI."""
    if not HAS_TYPER:
        print("Error: Typer is required for CLI.", file=sys.stderr)
        print("Install with: pip install agent-debug-toolkit[cli]", file=sys.stderr)
        sys.exit(1)

    app()


if __name__ == "__main__":
    main()
