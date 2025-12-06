#!/usr/bin/env python3
"""
Deprecated Code Registry for AgentQMS

Maintains a registry of deprecated code symbols and enforces that
artifacts do not reference them.

Configuration: .agentqms/state/deprecated.yaml

Entry format:
  symbol_name:
    file: path/to/deprecated/file.py
    replacement: path/to/replacement.py
    removal_plan: "Link to removal plan or issue"
    removal_date: "YYYY-MM-DD"
    description: "What this symbol did"
    block_modifications: true  # If true, artifact validation fails if referenced

Usage:
    python deprecated_registry.py register SYMBOL FILE REPLACEMENT
    python deprecated_registry.py list
    python deprecated_registry.py validate --file artifact.md
    python deprecated_registry.py validate --all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from AgentQMS.agent_tools.utils.paths import get_project_root
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


class DeprecatedRegistry:
    """Manages and validates against deprecated code symbols."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the deprecated registry.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()

        self.project_root = Path(project_root)
        self.registry_file = self.project_root / ".agentqms" / "state" / "deprecated.yaml"
        self._registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load deprecated symbols from registry file."""
        if not self.registry_file.exists():
            return {}

        try:
            with open(self.registry_file, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise RuntimeError(f"Failed to load deprecated registry: {e}") from e

    def _save_registry(self) -> None:
        """Save registry to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.registry_file, "w", encoding="utf-8") as f:
            yaml.dump(self._registry, f, default_flow_style=False, sort_keys=False)

    def register(
        self,
        symbol: str,
        file: str,
        replacement: str,
        removal_plan: str = "",
        removal_date: str = "",
        description: str = "",
    ) -> bool:
        """Register a deprecated symbol.

        Args:
            symbol: Name of deprecated symbol
            file: File containing the symbol
            replacement: Path to replacement
            removal_plan: Link to removal plan/issue
            removal_date: Planned removal date (YYYY-MM-DD)
            description: Description of the symbol

        Returns:
            True if registered successfully
        """
        self._registry[symbol] = {
            "file": file,
            "replacement": replacement,
        }

        if removal_plan:
            self._registry[symbol]["removal_plan"] = removal_plan
        if removal_date:
            self._registry[symbol]["removal_date"] = removal_date
        if description:
            self._registry[symbol]["description"] = description

        self._registry[symbol]["block_modifications"] = True

        self._save_registry()
        return True

    def unregister(self, symbol: str) -> bool:
        """Remove a symbol from the registry.

        Args:
            symbol: Name of symbol to remove

        Returns:
            True if removed
        """
        if symbol in self._registry:
            del self._registry[symbol]
            self._save_registry()
            return True
        return False

    def list_deprecated(self) -> list[dict[str, Any]]:
        """Get list of all deprecated symbols.

        Returns:
            List of deprecated symbol info
        """
        result = []
        for symbol, info in self._registry.items():
            result.append({
                "symbol": symbol,
                **info,
            })
        return result

    def find_references(self, text: str) -> list[str]:
        """Find deprecated symbol references in text.

        Args:
            text: Text to search

        Returns:
            List of found deprecated symbols
        """
        found = []
        for symbol in self._registry.keys():
            # Match word boundaries to avoid false positives
            if re.search(rf"\b{re.escape(symbol)}\b", text):
                found.append(symbol)
        return found

    def validate_file(self, filepath: Path) -> dict[str, Any]:
        """Validate artifact file for deprecated references.

        Args:
            filepath: Path to artifact file

        Returns:
            Validation result with violations list
        """
        if not filepath.exists():
            return {
                "status": "error",
                "file": str(filepath),
                "reason": "File not found",
            }

        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception as e:
            return {
                "status": "error",
                "file": str(filepath),
                "reason": f"Failed to read file: {e}",
            }

        found = self.find_references(content)

        if not found:
            return {
                "status": "pass",
                "file": str(filepath),
                "violations": [],
            }

        # Check if any found symbols block modifications
        violations = []
        for symbol in found:
            info = self._registry[symbol]
            if info.get("block_modifications", False):
                violations.append({
                    "symbol": symbol,
                    "file": info.get("file"),
                    "replacement": info.get("replacement"),
                    "removal_plan": info.get("removal_plan"),
                    "message": f"Artifact references deprecated symbol: {symbol}",
                })

        return {
            "status": "fail" if violations else "pass",
            "file": str(filepath),
            "violations": violations,
            "found_symbols": found,
        }

    def validate_directory(self, directory: Path) -> list[dict[str, Any]]:
        """Validate all artifacts in directory.

        Args:
            directory: Directory to search for .md files

        Returns:
            List of validation results
        """
        results = []
        for artifact_file in directory.rglob("*.md"):
            result = self.validate_file(artifact_file)
            if result.get("violations"):
                results.append(result)
        return results


def main() -> int:
    """Command-line interface for deprecated registry."""
    parser = argparse.ArgumentParser(
        description="Manage deprecated code symbol registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a deprecated symbol
  %(prog)s register PathUtils ocr/utils/paths.py ocr/core/path_manager.py

  # List all deprecated symbols
  %(prog)s list

  # Validate an artifact for deprecated references
  %(prog)s validate --file docs/artifacts/my_assessment.md

  # Validate all artifacts
  %(prog)s validate --all

  # Check specific directory
  %(prog)s validate --directory docs/artifacts/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register deprecated symbol")
    register_parser.add_argument("symbol", help="Symbol name")
    register_parser.add_argument("file", help="File containing symbol")
    register_parser.add_argument("replacement", help="Replacement path")
    register_parser.add_argument("--removal-plan", "-p", help="Link to removal plan")
    register_parser.add_argument("--removal-date", "-d", help="Planned removal date")
    register_parser.add_argument("--description", "-s", help="Description")

    # Unregister command
    unregister_parser = subparsers.add_parser("unregister", help="Unregister symbol")
    unregister_parser.add_argument("symbol", help="Symbol name to remove")

    # List command
    subparsers.add_parser("list", help="List deprecated symbols")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate files")
    validate_group = validate_parser.add_mutually_exclusive_group(required=True)
    validate_group.add_argument("--file", "-f", type=Path, help="File to validate")
    validate_group.add_argument("--all", action="store_true", help="Validate all artifacts")
    validate_group.add_argument("--directory", "-d", type=Path, help="Directory to validate")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        registry = DeprecatedRegistry()

        if args.command == "register":
            if registry.register(
                args.symbol,
                args.file,
                args.replacement,
                removal_plan=args.removal_plan or "",
                removal_date=args.removal_date or "",
                description=args.description or "",
            ):
                print(f"✅ Registered deprecated symbol: {args.symbol}")
            else:
                print(f"❌ Failed to register {args.symbol}")
                return 1

        elif args.command == "unregister":
            if registry.unregister(args.symbol):
                print(f"✅ Unregistered: {args.symbol}")
            else:
                print(f"❌ Symbol not found: {args.symbol}")
                return 1

        elif args.command == "list":
            deprecated = registry.list_deprecated()
            if not deprecated:
                print("✅ No deprecated symbols registered")
                return 0

            print(f"Deprecated Symbols ({len(deprecated)} total):")
            print("=" * 70)
            for item in deprecated:
                symbol = item["symbol"]
                file = item.get("file", "unknown")
                replacement = item.get("replacement", "unknown")
                removal_date = item.get("removal_date", "TBD")
                print(f"\n{symbol}")
                print(f"  File: {file}")
                print(f"  Replacement: {replacement}")
                print(f"  Removal date: {removal_date}")
                if item.get("removal_plan"):
                    print(f"  Removal plan: {item['removal_plan']}")

        elif args.command == "validate":
            if args.file:
                result = registry.validate_file(args.file)
                print(f"File: {result['file']}")
                print(f"Status: {result['status'].upper()}")

                if result.get("violations"):
                    print(f"\nViolations found ({len(result['violations'])}):")
                    for violation in result["violations"]:
                        print(f"  - {violation['symbol']}: {violation['message']}")
                        print(f"    Replace with: {violation['replacement']}")
                        if violation.get("removal_plan"):
                            print(f"    Removal plan: {violation['removal_plan']}")
                    return 1
                else:
                    print("✅ No deprecated references found")

            elif args.all:
                artifacts_dir = registry.project_root / "docs" / "artifacts"
                results = registry.validate_directory(artifacts_dir)

                if not results:
                    print("✅ All artifacts valid - no deprecated references")
                    return 0

                print(f"⚠️  {len(results)} artifact(s) with deprecated references:")
                for result in results:
                    print(f"\n{result['file']}")
                    for violation in result["violations"]:
                        print(f"  - {violation['symbol']}: {violation['message']}")
                return 1

            elif args.directory:
                if not args.directory.exists():
                    print(f"❌ Directory not found: {args.directory}")
                    return 1

                results = registry.validate_directory(args.directory)
                if not results:
                    print("✅ No deprecated references found in directory")
                    return 0

                print(f"⚠️  {len(results)} file(s) with deprecated references")
                return 1

        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
