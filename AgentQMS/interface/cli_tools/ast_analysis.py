#!/usr/bin/env python3
"""
Agent AST Analysis Tool (Agent-Only Version)
Provides AI agents with AST-based code analysis capabilities
"""

import os
import sys

from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path


def agent_ast_analysis():
    """Agent AST analysis interface."""
    print("🤖 Agent AST Analysis Tool (AGENT-ONLY)")
    print("=======================================")
    print()
    print("⚠️  WARNING: This tool is for AI agents only!")
    print("   Humans should use the main project tools.")
    print()

    ensure_project_root_on_sys_path()

    # Import and run the main AST tool with modified arguments
    try:
        # Get command line arguments
        args = sys.argv[1:]

        # If no arguments provided, show help
        if not args:
            print("Usage: python ast_analysis.py <command> [options]")
            print()
            print("Commands:")
            print("  analyze [path]          Analyze code structure")
            print("  generate-tests <file>   Generate test scaffolds")
            print("  extract-docs <file>     Extract documentation")
            print("  check-quality [path]    Check code quality")
            print()
            print("Examples:")
            print("  python ast_analysis.py analyze")
            print("  python ast_analysis.py analyze src/")
            print("  python ast_analysis.py generate-tests myfile.py")
            return 0

        # Map agent commands to CLI subcommands
        command = args[0]
        cli_args = []

        if command == "analyze":
            cli_args = ["analyze"]
            if (
                len(args) > 1 and args[1] != "--path"
            ):  # Handle case where PATH is passed
                cli_args.append(args[1])
            else:
                cli_args.append(".")  # Default to current directory
        elif command == "generate-tests":
            if len(args) < 2:
                print("❌ Error: generate-tests requires a file path")
                return 1
            cli_args = ["generate-tests", args[1]]
        elif command == "extract-docs":
            if len(args) < 2:
                print("❌ Error: extract-docs requires a file path")
                return 1
            cli_args = ["extract-docs", args[1]]
        elif command == "check-quality":
            cli_args = ["check-quality"]
            if (
                len(args) > 1 and args[1] != "--path"
            ):  # Handle case where PATH is passed
                cli_args.append(args[1])
            else:
                cli_args.append(".")  # Default to current directory
        else:
            print(f"❌ Unknown command: {command}")
            return 1

        # Replace sys.argv for the CLI tool
        original_argv = sys.argv
        sys.argv = [sys.argv[0], *cli_args]

        try:
            try:
                from AgentQMS.scripts.ast_analysis_cli import main
            except ImportError:
                from scripts.ast_analysis_cli import (
                    main,  # pragma: no cover - legacy fallback
                )

            main()
        finally:
            sys.argv = original_argv

    except ImportError as e:
        print(f"❌ Error importing AST analysis tool: {e}")
        print("   Make sure you're running from the agent directory")
        print(f"   Debug: Current path: {os.getcwd()}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(agent_ast_analysis())
