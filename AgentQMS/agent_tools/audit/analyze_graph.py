import json
import xml.etree.ElementTree as ET


def analyze_graph(graphml_path, staleness_report_path, output_path):
    # Parse GraphML
    tree = ET.parse(graphml_path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

    graph = root.find("g:graph", ns)
    nodes = graph.findall("g:node", ns)
    edges = graph.findall("g:edge", ns)

    # Count incoming edges
    incoming_counts = {node.get("id"): 0 for node in nodes}
    for edge in edges:
        target = edge.get("target")
        if target in incoming_counts:
            incoming_counts[target] += 1

    # Load staleness report for recency
    with open(staleness_report_path) as f:
        staleness_data = json.load(f)

    file_metadata = {item["file"]: item for item in staleness_data}

    # Combine incoming counts and recency
    # Score = incoming_count * (1 if recent, 0.5 if old) - simplified
    # Actually let's just use incoming count and recency as filters

    results = []
    for node_id, count in incoming_counts.items():
        metadata = file_metadata.get(node_id, {})
        last_modified = metadata.get("last_modified", "1970-01-01T00:00:00")
        staleness_score = metadata.get("staleness_score", 0)

        results.append({"file": node_id, "incoming_references": count, "last_modified": last_modified, "staleness_score": staleness_score})

    # Sort by incoming references (desc) then by modified date (desc)
    results.sort(key=lambda x: (x["incoming_references"], x["last_modified"]), reverse=True)

    # Take top 10% (84 files)
    top_n = max(1, len(results) // 10)
    if top_n > 84:
        top_n = 84  # Cap at 84 as per implementation plan

    high_value = results[:top_n]

    with open(output_path, "w") as f:
        json.dump(high_value, f, indent=2)

    print(f"High-value files report saved to {output_path}")
    print(f"Identified {len(high_value)} high-value files.")


if __name__ == "__main__":
    analyze_graph("reports/reference-graph.graphml", "reports/staleness-report.json", "reports/high-value-files.json")
