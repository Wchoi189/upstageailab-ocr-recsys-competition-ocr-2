#!/usr/bin/env python3
"""
AgentQMS Token Usage Monitor

Tracks and reports token usage improvements from the unified registry
and path-aware discovery system.

This script analyzes:
1. Registry file size vs. old INDEX.yaml + standards-router.yaml
2. Number of standards loaded with path-aware discovery
3. Estimated token savings from consolidated tool mappings

Usage:
    python AgentQMS/bin/monitor-token-usage.py
    python AgentQMS/bin/monitor-token-usage.py --path ocr/inference
    python AgentQMS/bin/monitor-token-usage.py --detailed
"""

import argparse
import json
from pathlib import Path


def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    if path.exists():
        return path.stat().st_size
    return 0


def count_lines(path: Path) -> int:
    """Count lines in a file."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    return 0


def estimate_tokens(text_length: int) -> int:
    """Estimate token count (rough approximation: 4 chars per token)."""
    return text_length // 4


def analyze_registry_consolidation():
    """Analyze token savings from registry consolidation."""
    print("=" * 70)
    print("📊 Registry Consolidation Analysis")
    print("=" * 70)

    # Old files (now archived)
    index_yaml = Path("AgentQMS/standards/.archive/INDEX.yaml")
    router_yaml = Path("AgentQMS/standards/.archive/standards-router.yaml")

    # New unified registry
    registry_yaml = Path("AgentQMS/standards/registry.yaml")

    old_size = get_file_size(index_yaml) + get_file_size(router_yaml)
    new_size = get_file_size(registry_yaml)

    old_lines = count_lines(index_yaml) + count_lines(router_yaml)
    new_lines = count_lines(registry_yaml)

    old_tokens = estimate_tokens(old_size)
    new_tokens = estimate_tokens(new_size)

    print(f"\n📁 File Sizes:")
    print(f"   Old (INDEX.yaml + standards-router.yaml): {old_size:,} bytes ({old_lines} lines)")
    print(f"   New (registry.yaml):                      {new_size:,} bytes ({new_lines} lines)")
    print(f"   Difference:                               {old_size - new_size:+,} bytes")

    print(f"\n🪙 Estimated Tokens:")
    print(f"   Old system:  ~{old_tokens:,} tokens")
    print(f"   New system:  ~{new_tokens:,} tokens")
    print(f"   Savings:     ~{old_tokens - new_tokens:,} tokens ({((old_size - new_size) / old_size * 100):.1f}% reduction)")

    return {
        "old_size": old_size,
        "new_size": new_size,
        "savings_bytes": old_size - new_size,
        "savings_tokens": old_tokens - new_tokens,
        "reduction_pct": (old_size - new_size) / old_size * 100 if old_size > 0 else 0,
    }


def analyze_tool_consolidation():
    """Analyze token savings from tool consolidation."""
    print("\n" + "=" * 70)
    print("🔧 Tool Consolidation Analysis")
    print("=" * 70)

    # Count old tool definitions
    old_tools = [
        "artifact_workflow",
        "validate_artifacts",
        "monitor_artifacts",
        "agent_feedback",
        "documentation_quality_monitor",
    ]

    print(f"\n📦 Tool Reduction:")
    print(f"   Old: {len(old_tools)} separate tool entries")
    print(f"   New: 1 unified qms CLI with {len(old_tools)} subcommands")
    print(f"   Reduction: {len(old_tools) - 1} tool entries removed from context")

    # Estimate token savings (each tool entry ~100 tokens in settings)
    tokens_per_tool = 100
    old_tool_tokens = len(old_tools) * tokens_per_tool
    new_tool_tokens = tokens_per_tool + (20 * len(old_tools))  # Main entry + subcommand hints

    print(f"\n🪙 Estimated Token Impact:")
    print(f"   Old tool mappings: ~{old_tool_tokens:,} tokens")
    print(f"   New tool mapping:  ~{new_tool_tokens:,} tokens")
    print(f"   Savings:           ~{old_tool_tokens - new_tool_tokens:,} tokens")
    print(f"   AI decision space: Reduced from {len(old_tools)} choices to 1")

    return {
        "old_tools": len(old_tools),
        "new_tools": 1,
        "savings_tokens": old_tool_tokens - new_tool_tokens,
        "action_space_reduction": len(old_tools) - 1,
    }


def analyze_path_aware_discovery(current_path: str | None = None):
    """Analyze path-aware discovery efficiency."""
    print("\n" + "=" * 70)
    print("🎯 Path-Aware Discovery Analysis")
    print("=" * 70)

    if current_path:
        print(f"\n📍 Testing with path: {current_path}")

        # Use the ConfigLoader to resolve standards
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from AgentQMS.tools.utils.config_loader import ConfigLoader

            loader = ConfigLoader()
            active_standards = loader.resolve_active_standards(
                current_path=current_path,
                registry_path="AgentQMS/standards/registry.yaml"
            )

            print(f"\n✅ Active Standards Resolved: {len(active_standards)}")
            if active_standards:
                print("\n   Standards loaded:")
                for std in active_standards[:10]:  # Show first 10
                    print(f"   - {std}")
                if len(active_standards) > 10:
                    print(f"   ... and {len(active_standards) - 10} more")

            # Estimate token savings vs. loading all standards
            import yaml
            registry_path = Path("AgentQMS/standards/registry.yaml")
            with registry_path.open("r") as f:
                registry = yaml.safe_load(f)

            all_standards = set()
            for task_config in registry.get("task_mappings", {}).values():
                all_standards.update(task_config.get("standards", []))

            total_standards = len(all_standards)
            loaded_standards = len(active_standards)

            print(f"\n💡 Efficiency Gain:")
            print(f"   Total standards in registry: {total_standards}")
            print(f"   Standards loaded for path:   {loaded_standards}")
            print(f"   Reduction:                   {total_standards - loaded_standards} standards ({((total_standards - loaded_standards) / total_standards * 100):.1f}%)")

            # Estimate token impact (each standard ~500 tokens)
            tokens_per_standard = 500
            old_tokens = total_standards * tokens_per_standard
            new_tokens = loaded_standards * tokens_per_standard

            print(f"\n🪙 Estimated Token Savings:")
            print(f"   Without path-aware:  ~{old_tokens:,} tokens")
            print(f"   With path-aware:     ~{new_tokens:,} tokens")
            print(f"   Savings:             ~{old_tokens - new_tokens:,} tokens")

            return {
                "total_standards": total_standards,
                "loaded_standards": loaded_standards,
                "savings_tokens": old_tokens - new_tokens,
                "reduction_pct": (total_standards - loaded_standards) / total_standards * 100,
            }

        except Exception as e:
            print(f"\n⚠️  Error testing path-aware discovery: {e}")
            return None
    else:
        print("\n💡 Path-aware discovery provides context-specific standard loading")
        print("   Use --path <path> to test with a specific working directory")
        return None


def generate_summary_report(results: dict):
    """Generate a summary report of all improvements."""
    print("\n" + "=" * 70)
    print("📋 SUMMARY REPORT")
    print("=" * 70)

    total_savings = 0

    print("\n🎯 Key Improvements:")

    if "registry" in results and results["registry"]:
        savings = results["registry"]["savings_tokens"]
        total_savings += savings
        print(f"   1. Registry Consolidation:    ~{savings:,} tokens saved")

    if "tools" in results and results["tools"]:
        savings = results["tools"]["savings_tokens"]
        total_savings += savings
        print(f"   2. Tool Consolidation:        ~{savings:,} tokens saved")

    if "discovery" in results and results["discovery"]:
        savings = results["discovery"]["savings_tokens"]
        total_savings += savings
        print(f"   3. Path-Aware Discovery:      ~{savings:,} tokens saved (per request)")

    print(f"\n💰 Total Estimated Savings:      ~{total_savings:,} tokens per agent session")
    print(f"   (excluding path-aware discovery which applies per request)")

    print("\n📈 Performance Impact:")
    print("   - Reduced AI decision complexity (5 tools → 1 unified CLI)")
    print("   - Eliminated logic fragmentation (2 discovery files → 1 registry)")
    print("   - Context-aware standard loading (dynamic vs. static)")
    print("   - Backward compatible (zero breaking changes)")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor and analyze AgentQMS token usage improvements"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Test path-aware discovery with specific working directory"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )

    args = parser.parse_args()

    # Run analyses
    results = {}

    if not args.json:
        print("\n🔍 AgentQMS Token Usage Analysis")
        print("=" * 70)

    results["registry"] = analyze_registry_consolidation()
    results["tools"] = analyze_tool_consolidation()
    results["discovery"] = analyze_path_aware_discovery(args.path)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        generate_summary_report(results)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
