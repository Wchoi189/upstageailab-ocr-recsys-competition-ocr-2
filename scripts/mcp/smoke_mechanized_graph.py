#!/usr/bin/env python3
"""Deterministic smoke checks for mechanized graph generation."""

from __future__ import annotations

from AgentQMS.tools.generate_mechanized_graph import generate_mechanized_graph


def _fixture_standards() -> dict[str, dict]:
    return {
        "SC-001": {"tier": 1, "description": "Naming", "priority": "high", "dependencies": []},
        "SC-002": {"tier": 1, "description": "Type System", "priority": "critical", "dependencies": []},
        "FW-001": {"tier": 2, "description": "Architecture", "dependencies": ["SC-002"]},
        "FW-005": {"tier": 2, "description": "Templates", "dependencies": []},
        "FW-011": {"tier": 2, "description": "Safety Config", "dependencies": ["FW-001"]},
        "FW-026": {"tier": 2, "description": "Data Structures", "dependencies": ["SC-002"]},
        "AG-002": {"tier": 3, "description": "Agent Identity", "dependencies": ["FW-001"]},
        "WF-001": {"tier": 4, "description": "Execution Workflow", "dependencies": ["AG-002"]},
    }


def _assert_contains(dot: str, needle: str) -> None:
    assert needle in dot, f"missing DOT fragment: {needle}"


def main() -> int:
    standards = _fixture_standards()
    dot_a = generate_mechanized_graph(standards, include_legend=True, include_domains=True)
    dot_b = generate_mechanized_graph(standards, include_legend=True, include_domains=True)

    assert dot_a == dot_b, "graph output must be deterministic for identical input"
    _assert_contains(dot_a, "subgraph cluster_tier1")
    _assert_contains(dot_a, "subgraph cluster_tier2")
    _assert_contains(dot_a, "subgraph cluster_legend")
    _assert_contains(dot_a, '"SC-001" -> "FW-005" [style=dashed, color="#666666", penwidth=1.0];')
    _assert_contains(dot_a, '"SC-002" -> "FW-026" [color=blue, penwidth=1.5];')
    _assert_contains(dot_a, '"SC-002" -> "FW-026" [color=red, penwidth=2.5, label="CRITICAL"];')

    dot_no_legend = generate_mechanized_graph(standards, include_legend=False, include_domains=False)
    assert "cluster_legend" not in dot_no_legend, "legend must be removed when include_legend=False"
    assert "cluster_core_infra" not in dot_no_legend, "domains must be disabled when include_domains=False"
    _assert_contains(dot_no_legend, "subgraph cluster_tier2")

    print("mechanized_graph_smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
