#!/usr/bin/env python3
"""
Phase 6.5: Mechanized Architecture Graph Generator

Transforms the "Ghost System" into a mechanized architecture by:
1. Fixing DOT syntax issues
2. Implementing dual-edge strategy (solid=dependency, dashed=governance)
3. Adding functional domain grouping for Tier 2
4. Creating critical path chain
5. Adding architectural directives (legend, hub nodes)
6. Implementing cherry-picked architectural mappings
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "AgentQMS" / "standards" / "registry.yaml"
CACHE_PATH = PROJECT_ROOT / "AgentQMS" / "standards" / ".ads_cache.pickle"
DOT_OUTPUT = PROJECT_ROOT / "AgentQMS" / "standards" / "architecture_map.dot"

# Architectural Mappings: Define logical connections beyond declared dependencies
# Format: (source, target, edge_type)
# edge_type: "dependency" (solid, upward flow) or "governance" (dashed, downward flow)

GOVERNANCE_MAPPINGS = [
    # Tier 1 → Tier 2: Constitutional laws govern framework components
    ("SC-001", "FW-005", "governance"),  # Naming conventions govern artifact templates
    ("SC-001", "FW-016", "governance"),  # Naming conventions govern git conventions
    ("SC-002", "FW-026", "governance"),  # Type system governs data structures
    ("SC-003", "FW-005", "governance"),  # Artifact rules govern artifact template config
    ("SC-003", "FW-032", "governance"),  # Artifact rules govern template defaults
    ("SC-007", "FW-001", "governance"),  # System architecture governs agent architecture
    ("SC-008", "FW-029", "governance"),  # Validation protocols govern pydantic validation
    ("SC-008", "FW-033", "governance"),  # Validation protocols govern testing

    # Tier 1 → Tier 3: Constitutional laws govern agents
    ("SC-001", "AG-002", "governance"),  # Naming conventions govern agent config
    ("SC-007", "AG-006", "governance"),  # System architecture governs ollama models (already exists)

    # Tier 2 → Tier 3: Framework governs agents
    ("FW-007", "AG-002", "governance"),  # Interface contracts govern agent behavior
]

DEPENDENCY_MAPPINGS = [
    # Tier 2 → Tier 2: Framework internal dependencies
    ("FW-026", "FW-027", "dependency"),  # Data structures used by postprocessing
    ("FW-026", "FW-028", "dependency"),  # Data structures used by preprocessing
    ("FW-026", "FW-012", "dependency"),  # Data structures used by coordinate transforms
    ("FW-001", "FW-002", "dependency"),  # Architecture defines feedback protocol
    ("FW-019", "FW-017", "dependency"),  # Hydra rules feed into hydra architecture
    ("FW-011", "FW-019", "dependency"),  # Config standards feed into hydra rules
    ("FW-030", "FW-029", "dependency"),  # Python core feeds into pydantic validation

    # Tier 2 → Tier 3: Framework consumed by agents
    ("FW-034", "AG-002", "dependency"),  # Tool catalog used by agent config (already exists via AG-006)
    ("FW-001", "AG-002", "dependency"),  # Agent architecture used by agent config
    ("FW-001", "AG-003", "dependency"),  # Agent architecture used by multi-agent system

    # Tier 3 → Tier 4: Agents orchestrated by workflows
    ("AG-002", "WF-001", "dependency"),  # Agent config used by experiment workflow
    ("AG-004", "WF-001", "dependency"),  # Qwen agent used by experiment workflow
    ("FW-002", "WF-002", "dependency"),  # Feedback protocol used by middleware policies
]

# Critical Path: The backbone chain of critical standards
CRITICAL_PATH = [
    "SC-002",  # Type System
    "FW-026",  # Data Structures
    "FW-001",  # Architecture
    "FW-011",  # Safety/Config
    "AG-002",  # Agent Identity
    "WF-001",  # Execution
]

# Tier 2 Functional Domains
TIER2_DOMAINS = {
    "core_infra": {
        "label": "Core Infrastructure",
        "standards": ["FW-001", "FW-002", "FW-004", "FW-030"],
        "color": "#E6F0FF",
    },
    "configuration": {
        "label": "Configuration & Hydra",
        "standards": ["FW-011", "FW-017", "FW-019", "FW-008", "FW-009", "FW-010"],
        "color": "#FFF4E6",
    },
    "ocr_engine": {
        "label": "OCR Data Pipeline",
        "standards": ["FW-020", "FW-026", "FW-027", "FW-028", "FW-012", "FW-023", "FW-024"],
        "color": "#E6FFF0",
    },
    "validation": {
        "label": "Safety & Validation",
        "standards": ["FW-029", "FW-033", "FW-034", "FW-037"],
        "color": "#FFE6F0",
    },
    "patterns": {
        "label": "Patterns & Design",
        "standards": ["FW-003", "FW-007", "FW-018", "FW-035"],
        "color": "#F0E6FF",
    },
    "tooling": {
        "label": "Tooling & DevEx",
        "standards": ["FW-005", "FW-006", "FW-013", "FW-014", "FW-015", "FW-016",
                      "FW-021", "FW-022", "FW-025", "FW-031", "FW-032", "FW-036"],
        "color": "#E6E6E6",
    },
}


def load_registry() -> dict[str, Any]:
    """Load the registry.yaml file and merge with cache for full metadata."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    # Try to load cache for full metadata (including priority)
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
                full_standards = cache.get("standards", {})

                # Merge priority info from cache into registry standards
                for std_id, std_data in registry.get("standards", {}).items():
                    if std_id in full_standards:
                        std_data["priority"] = full_standards[std_id].get("priority", "medium")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load cache: {e}")

    return registry


def find_domain_for_standard(std_id: str) -> str:
    """Find which functional domain a Tier 2 standard belongs to."""
    for domain_id, domain_info in TIER2_DOMAINS.items():
        if std_id in domain_info["standards"]:
            return domain_id
    return "other"


def generate_mechanized_graph(
    standards: dict[str, dict[str, Any]],
    include_legend: bool = True,
    include_domains: bool = True,
) -> str:
    """
    Generate mechanized architecture graph with proper dependencies.

    Args:
        standards: Dictionary of standard IDs to their metadata
        include_legend: Whether to include the legend subgraph
        include_domains: Whether to group Tier 2 by functional domains

    Returns:
        DOT format string
    """
    lines = [
        "digraph AgentQMS_Architecture {",
        '  rankdir=TB;',  # Top to bottom for governance flow
        '  node [shape=box, style=rounded, fontname="Arial"];',
        '  edge [fontname="Arial", fontsize=10];',
        '  compound=true;',  # Enable edges to/from clusters
        "",
    ]

    # Group standards by tier
    tier_groups = defaultdict(list)
    for std_id, header in standards.items():
        tier = header.get("tier", 0)
        tier_groups[tier].append((std_id, header))

    # Tier colors
    tier_colors = {
        1: "#FFE6E6",  # Light red - Constitution
        2: "#E6F3FF",  # Light blue - Framework (will be overridden by domains)
        3: "#E6FFE6",  # Light green - Agents
        4: "#FFF9E6",  # Light yellow - Workflows
    }

    tier_names = {
        1: "Constitution (Tier 1)",
        2: "Framework (Tier 2)",
        3: "Agents (Tier 3)",
        4: "Workflows (Tier 4)",
    }

    # Generate Tier 1 (Constitution)
    if 1 in tier_groups:
        lines.append('  subgraph cluster_tier1 {')
        lines.append(f'    label="{tier_names[1]}";')
        lines.append('    style=filled;')
        lines.append(f'    color="{tier_colors[1]}";')
        lines.append('    fontsize=14;')
        lines.append('    fontname="Arial Bold";')
        lines.append("")

        for std_id, header in sorted(tier_groups[1]):
            desc = header.get("description", "")
            if desc and len(desc) > 40:
                desc = desc[:37] + "..."
            label = f"{std_id}\\n{desc}" if desc else std_id
            priority = header.get("priority", "medium")

            # Style by priority - FIX: Remove trailing comma
            if priority == "critical":
                style = 'penwidth=2, color=red'
            elif priority == "high":
                style = 'penwidth=1.5, color=orange'
            else:
                style = None

            if style:
                lines.append(f'    "{std_id}" [label="{label}", {style}];')
            else:
                lines.append(f'    "{std_id}" [label="{label}"];')

        lines.append("  }")
        lines.append("")

    # Generate Tier 2 (Framework) with functional domains
    if 2 in tier_groups and include_domains:
        lines.append('  subgraph cluster_tier2 {')
        lines.append(f'    label="{tier_names[2]}";')
        lines.append('    style=filled;')
        lines.append('    color="#CCCCCC";')  # Gray border for main tier
        lines.append('    fontsize=14;')
        lines.append('    fontname="Arial Bold";')
        lines.append("")

        # Group standards by domain
        domain_standards = defaultdict(list)
        for std_id, header in tier_groups[2]:
            domain = find_domain_for_standard(std_id)
            domain_standards[domain].append((std_id, header))

        # Generate domain subgraphs
        for domain_id in ["core_infra", "ocr_engine", "configuration", "validation", "patterns", "tooling"]:
            if domain_id not in domain_standards or not domain_standards[domain_id]:
                continue

            domain_info = TIER2_DOMAINS.get(domain_id, {})
            domain_label = domain_info.get("label", domain_id)
            domain_color = domain_info.get("color", "#FFFFFF")

            lines.append(f'    subgraph cluster_{domain_id} {{')
            lines.append(f'      label="{domain_label}";')
            lines.append('      style=filled;')
            lines.append(f'      color="{domain_color}";')
            lines.append('      fontsize=11;')
            lines.append("")

            for std_id, header in sorted(domain_standards[domain_id]):
                desc = header.get("description", "")
                if desc and len(desc) > 40:
                    desc = desc[:37] + "..."
                label = f"{std_id}\\n{desc}" if desc else std_id
                priority = header.get("priority", "medium")

                # Style by priority - FIX: Remove trailing comma
                if priority == "critical":
                    style = 'penwidth=2, color=red'
                elif priority == "high":
                    style = 'penwidth=1.5, color=orange'
                else:
                    style = None

                if style:
                    lines.append(f'      "{std_id}" [label="{label}", {style}];')
                else:
                    lines.append(f'      "{std_id}" [label="{label}"];')

            lines.append("    }")
            lines.append("")

        lines.append("  }")
        lines.append("")

    elif 2 in tier_groups:
        # Fallback: No domain grouping
        lines.append('  subgraph cluster_tier2 {')
        lines.append(f'    label="{tier_names[2]}";')
        lines.append('    style=filled;')
        lines.append(f'    color="{tier_colors[2]}";')
        lines.append('    fontsize=14;')
        lines.append('    fontname="Arial Bold";')
        lines.append("")

        for std_id, header in sorted(tier_groups[2]):
            desc = header.get("description", "")
            if desc and len(desc) > 40:
                desc = desc[:37] + "..."
            label = f"{std_id}\\n{desc}" if desc else std_id
            priority = header.get("priority", "medium")

            if priority == "critical":
                style = 'penwidth=2, color=red'
            elif priority == "high":
                style = 'penwidth=1.5, color=orange'
            else:
                style = None

            if style:
                lines.append(f'    "{std_id}" [label="{label}", {style}];')
            else:
                lines.append(f'    "{std_id}" [label="{label}"];')

        lines.append("  }")
        lines.append("")

    # Generate Tier 3 (Agents)
    if 3 in tier_groups:
        lines.append('  subgraph cluster_tier3 {')
        lines.append(f'    label="{tier_names[3]}";')
        lines.append('    style=filled;')
        lines.append(f'    color="{tier_colors[3]}";')
        lines.append('    fontsize=14;')
        lines.append('    fontname="Arial Bold";')
        lines.append("")

        for std_id, header in sorted(tier_groups[3]):
            desc = header.get("description", "")
            if desc and len(desc) > 40:
                desc = desc[:37] + "..."
            label = f"{std_id}\\n{desc}" if desc else std_id
            priority = header.get("priority", "medium")

            if priority == "critical":
                style = 'penwidth=2, color=red'
            elif priority == "high":
                style = 'penwidth=1.5, color=orange'
            else:
                style = None

            if style:
                lines.append(f'    "{std_id}" [label="{label}", {style}];')
            else:
                lines.append(f'    "{std_id}" [label="{label}"];')

        lines.append("  }")
        lines.append("")

    # Generate Tier 4 (Workflows)
    if 4 in tier_groups:
        lines.append('  subgraph cluster_tier4 {')
        lines.append(f'    label="{tier_names[4]}";')
        lines.append('    style=filled;')
        lines.append(f'    color="{tier_colors[4]}";')
        lines.append('    fontsize=14;')
        lines.append('    fontname="Arial Bold";')
        lines.append("")

        for std_id, header in sorted(tier_groups[4]):
            desc = header.get("description", "")
            if desc and len(desc) > 40:
                desc = desc[:37] + "..."
            label = f"{std_id}\\n{desc}" if desc else std_id
            priority = header.get("priority", "medium")

            if priority == "critical":
                style = 'penwidth=2, color=red'
            elif priority == "high":
                style = 'penwidth=1.5, color=orange'
            else:
                style = None

            if style:
                lines.append(f'    "{std_id}" [label="{label}", {style}];')
            else:
                lines.append(f'    "{std_id}" [label="{label}"];')

        lines.append("  }")
        lines.append("")

    # Add legend
    if include_legend:
        lines.append('  subgraph cluster_legend {')
        lines.append('    label="Legend";')
        lines.append('    style=filled;')
        lines.append('    color="#F5F5F5";')
        lines.append('    fontsize=12;')
        lines.append('    fontname="Arial Bold";')
        lines.append('')
        lines.append('    legend_critical [label="Critical (Red)", penwidth=2, color=red, shape=box, style=rounded];')
        lines.append('    legend_high [label="High Priority (Orange)", penwidth=1.5, color=orange, shape=box, style=rounded];')
        lines.append('    legend_dep [label="Dependency\\n(Solid Arrow)", shape=plaintext];')
        lines.append('    legend_gov [label="Governance\\n(Dashed Arrow)", shape=plaintext];')
        lines.append('')
        lines.append('    legend_critical -> legend_high [style=invis];')
        lines.append('    legend_high -> legend_dep [style=invis];')
        lines.append('    legend_dep -> legend_gov [style=invis];')
        lines.append('  }')
        lines.append('')

    # Add edges: Declared dependencies (from registry)
    lines.append("  // Declared Dependencies (from registry.yaml)")
    for std_id, header in standards.items():
        for dep_id in header.get("dependencies", []):
            if dep_id in standards:
                # Dependencies flow upward (dep -> dependent)
                lines.append(f'  "{dep_id}" -> "{std_id}" [color=blue, penwidth=1.5];')

    lines.append("")

    # Add edges: Governance mappings (dashed, downward)
    lines.append("  // Governance Mappings (Constitutional laws enforce lower tiers)")
    for source, target, edge_type in GOVERNANCE_MAPPINGS:
        if source in standards and target in standards:
            lines.append(f'  "{source}" -> "{target}" [style=dashed, color="#666666", penwidth=1.0];')

    lines.append("")

    # Add edges: Dependency mappings (solid, upward)
    lines.append("  // Architectural Dependencies (Framework consumed by Agents/Workflows)")
    for source, target, edge_type in DEPENDENCY_MAPPINGS:
        if source in standards and target in standards:
            lines.append(f'  "{source}" -> "{target}" [color="#0066CC", penwidth=1.5];')

    lines.append("")

    # Add critical path highlighting
    lines.append("  // Critical Path Chain")
    for i in range(len(CRITICAL_PATH) - 1):
        source = CRITICAL_PATH[i]
        target = CRITICAL_PATH[i + 1]
        if source in standards and target in standards:
            lines.append(f'  "{source}" -> "{target}" [color=red, penwidth=2.5, label="CRITICAL"];')

    lines.append("}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6.5: Mechanized Architecture Graph Generator"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DOT_OUTPUT,
        help="Output DOT file path"
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Exclude legend from graph"
    )
    parser.add_argument(
        "--no-domains",
        action="store_true",
        help="Don't group Tier 2 by functional domains"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print graph to stdout without saving"
    )

    args = parser.parse_args()

    print("🎨 Phase 6.5: Mechanized Architecture Graph Generator")
    print("="*60)

    try:
        # Load registry
        print("📋 Loading registry...")
        registry = load_registry()
        standards = registry.get("standards", {})
        print(f"   ✓ Loaded {len(standards)} standards")

        # Generate graph
        print("\n🔧 Generating mechanized architecture graph...")
        dot_content = generate_mechanized_graph(
            standards,
            include_legend=not args.no_legend,
            include_domains=not args.no_domains,
        )

        # Count edges
        governance_edges = sum(1 for line in dot_content.split('\n') if 'style=dashed' in line)
        dependency_edges = sum(1 for line in dot_content.split('\n') if '-> ' in line and 'style=dashed' not in line and 'style=invis' not in line)

        print("   ✓ Generated graph:")
        print(f"     - Governance edges: {governance_edges}")
        print(f"     - Dependency edges: {dependency_edges}")
        print(f"     - Total edges: {governance_edges + dependency_edges}")

        if args.dry_run:
            print("\n📄 Graph content (dry-run):")
            print(dot_content)
        else:
            # Write to file
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(dot_content)
            print(f"\n💾 Saved to: {args.output}")
            print("\n💡 Render with:")
            print(f"   dot -Tpng {args.output} -o {args.output.with_suffix('.png')}")
            print(f"   dot -Tsvg {args.output} -o {args.output.with_suffix('.svg')}")

        print("\n" + "="*60)
        print("✅ Mechanized architecture graph generated successfully!")
        print("="*60 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
