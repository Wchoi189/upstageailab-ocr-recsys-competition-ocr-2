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

        Combines config access, merge tracking, Hydra usage, and instantiation analysis.
        """
        from agent_debug_toolkit.analyzers.config_access import ConfigAccessAnalyzer
        from agent_debug_toolkit.analyzers.merge_order import MergeOrderTracker
        from agent_debug_toolkit.analyzers.hydra_usage import HydraUsageAnalyzer
        from agent_debug_toolkit.analyzers.instantiation import ComponentInstantiationTracker

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
