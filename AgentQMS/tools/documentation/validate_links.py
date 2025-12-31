#!/usr/bin/env python3
"""
Documentation Link Validator

Validates internal and external links in documentation files.
Checks for broken internal references and unreachable external URLs.
"""

import os
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


class LinkValidator:
    """Validates links in documentation files."""

    def __init__(self, docs_root: str):
        self.docs_root = Path(docs_root)
        self.errors: list[str] = []
        self.warnings: list[str] = []

        # Common file extensions for documentation
        self.doc_extensions = {".md", ".markdown", ".txt", ".rst"}

        # Files to skip
        self.skip_files = {"README.md", "CONTRIBUTING.md", "CHANGELOG.md"}

    def find_doc_files(self) -> list[Path]:
        """Find all documentation files in the docs directory."""
        doc_files: list[Path] = []
        for ext in self.doc_extensions:
            doc_files.extend(self.docs_root.rglob(f"*{ext}"))
        return [f for f in doc_files if f.name not in self.skip_files]

    def extract_links(self, file_path: Path) -> list[tuple[str, int]]:
        """Extract all links from a markdown file."""
        links = []
        try:
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    # Find markdown links: [text](url)
                    md_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", line)
                    for text, url in md_links:
                        links.append((url.strip(), line_num))

                    # Find reference-style links: [text][ref] or [ref]: url
                    ref_links = re.findall(r"\[([^\]]+)\]\[([^\]]+)\]", line)
                    for text, ref in ref_links:
                        links.append((f"#{ref}", line_num))

        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")

        return links

    def validate_internal_link(self, url: str, file_path: Path) -> bool:
        """Validate internal links (relative paths and anchors)."""
        if url.startswith(("http://", "https://", "mailto:", "tel:")):
            return True  # External links handled separately

        if url.startswith("#"):
            # Anchor link - check if it exists in the same file
            return self.check_anchor_exists(file_path, url[1:])

        # Relative path
        try:
            target_path = (file_path.parent / url).resolve()
            if target_path.exists():
                return True
            else:
                # Try with .md extension if not present
                if not target_path.suffix and "." not in target_path.name:
                    target_path = target_path.with_suffix(".md")
                    return target_path.exists()
        except Exception:
            pass

        return False

    def check_anchor_exists(self, file_path: Path, anchor: str) -> bool:
        """Check if an anchor exists in a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                # Look for headers that would generate this anchor
                # GitHub-style anchor generation: lowercase, spaces to hyphens
                anchor_pattern = re.compile(
                    rf"^#+\s+{re.escape(anchor.replace('-', ' '))}",
                    re.MULTILINE | re.IGNORECASE,
                )
                return bool(anchor_pattern.search(content))
        except Exception:
            return False

    def validate_external_link(self, url: str) -> bool:
        """Validate external links by making HTTP requests."""
        try:
            # Only check HTTP/HTTPS links
            if not url.startswith(("http://", "https://")):
                return True

            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status < 400
        except Exception:
            return False

    def validate_file(self, file_path: Path) -> None:
        """Validate all links in a single file."""
        links = self.extract_links(file_path)

        for url, line_num in links:
            if not self.validate_internal_link(url, file_path):
                if url.startswith(("http://", "https://")):
                    if not self.validate_external_link(url):
                        self.errors.append(f"Broken external link in {file_path}:{line_num}: {url}")
                else:
                    self.errors.append(f"Broken internal link in {file_path}:{line_num}: {url}")

    def validate_all(self) -> bool:
        """Validate all documentation files."""
        doc_files = self.find_doc_files()

        if not doc_files:
            self.warnings.append("No documentation files found")
            return True

        print(f"Found {len(doc_files)} documentation files to validate")

        for file_path in doc_files:
            print(f"Validating {file_path}")
            self.validate_file(file_path)

        # Report results
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} link errors:")
            for error in self.errors:
                print(f"  - {error}")
            return False
        else:
            print("\n✅ All links are valid!")
            return True

    def build_reference_graph(self) -> dict[str, list[str]]:
        """Build a reference graph of documentation files."""
        doc_files = self.find_doc_files()
        graph: dict[str, list[str]] = {}

        for file_path in doc_files:
            rel_file = str(file_path.relative_to(self.docs_root.parent))
            graph[rel_file] = []
            links = self.extract_links(file_path)

            for url, _ in links:
                if url.startswith(("http", "mailto", "tel")):
                    continue
                if url.startswith("#"):
                    continue

                try:
                    # Resolve internal link
                    target_path = (file_path.parent / url).resolve()
                    if not target_path.exists():
                        if not target_path.suffix and "." not in target_path.name:
                            target_path = target_path.with_suffix(".md")

                    if target_path.exists():
                        rel_target = str(target_path.relative_to(self.docs_root.parent))
                        if rel_target not in graph[rel_file]:
                            graph[rel_file].append(rel_target)
                except Exception:
                    pass

        return graph

    def export_graphml(self, graph: dict[str, list[str]], output_path: Path):
        """Export the reference graph in GraphML format."""
        # Create the root element
        graphml = ET.Element(
            "graphml",
            {
                "xmlns": "http://graphml.graphdrawing.org/xmlns",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
            },
        )

        # Add graph definition
        graph_elem = ET.SubElement(graphml, "graph", {"id": "G", "edgedefault": "directed"})

        # Add nodes
        nodes = set(graph.keys())
        for targets in graph.values():
            nodes.update(targets)

        for node_id in sorted(nodes):
            ET.SubElement(graph_elem, "node", {"id": node_id})

        # Add edges
        edge_id_counter = 0
        for source, targets in graph.items():
            for target in targets:
                ET.SubElement(graph_elem, "edge", {"id": f"e{edge_id_counter}", "source": source, "target": target})
                edge_id_counter += 1

        # Write to file
        tree = ET.ElementTree(graphml)
        with open(output_path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Link Validator")
    parser.add_argument("docs_root", help="Root directory of documentation")
    parser.add_argument("--export-graph", help="Export reference graph to GraphML file")

    args = parser.parse_args()

    if not os.path.exists(args.docs_root):
        print(f"Error: Documentation root '{args.docs_root}' does not exist")
        sys.exit(1)

    validator = LinkValidator(args.docs_root)

    if args.export_graph:
        graph = validator.build_reference_graph()
        validator.export_graphml(graph, Path(args.export_graph))
        print(f"Reference graph exported to {args.export_graph}")
        sys.exit(0)

    success = validator.validate_all()

    if validator.warnings:
        print("\n⚠️ Warnings:")
        for warning in validator.warnings:
            print(f"  - {warning}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
