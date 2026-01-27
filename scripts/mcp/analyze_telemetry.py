#!/usr/bin/env python3
"""
Telemetry Analysis Script.
Parses .mcp-telemetry.jsonl and generates insights on tool usage and metadata.
"""
import json
from collections import Counter, defaultdict
from AgentQMS.tools.utils.system.paths import get_project_root

def analyze_telemetry():
    # Find telemetry file
    root = get_project_root()
    telemetry_file = root / "AgentQMS" / ".mcp-telemetry.jsonl"

    if not telemetry_file.exists():
        print(f"No telemetry file found at {telemetry_file}")
        return

    print(f"📊 Analyzing telemetry from: {telemetry_file}")
    print("-" * 50)

    events = []
    with open(telemetry_file, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not events:
        print("No events found.")
        return

    # Basic Stats
    total = len(events)
    status_counts = Counter(e.get('status', 'unknown') for e in events)

    print(f"Total Events: {total}")
    for status, count in status_counts.items():
        print(f"  - {status}: {count}")
    print("-" * 50)

    # Metadata Analysis
    metadata_keys = set()
    usage_by_key = defaultdict(Counter)
    tool_metadata = defaultdict(list)

    for e in events:
        meta = e.get('metadata', {})
        tool = e.get('tool_name', 'unknown')

        if meta:
            tool_metadata[tool].append(meta)
            for k, v in meta.items():
                metadata_keys.add(k)
                usage_by_key[k][str(v)] += 1

    if not metadata_keys:
        print("No metadata logged yet.")
    else:
        print("🔍 Metadata Insights:")
        for key in sorted(metadata_keys):
            print(f"\n[ {key} ]")
            for val, count in usage_by_key[key].most_common(5):
                print(f"  {val}: {count}")

    print("-" * 50)
    print("🛠️  Tool Breakdown with Context:")
    for tool, metalist in tool_metadata.items():
        if not metalist:
            continue
        print(f"\nTool: {tool}")
        # Aggregate logic for this tool
        # e.g. what bundles were requested?
        local_counts = defaultdict(Counter)
        for m in metalist:
            for k, v in m.items():
                local_counts[k][str(v)] += 1

        for k, counter in local_counts.items():
             top = counter.most_common(3)
             formatted = ", ".join([f"{val}({cnt})" for val, cnt in top])
             print(f"  - {k}: {formatted}")

if __name__ == "__main__":
    analyze_telemetry()
