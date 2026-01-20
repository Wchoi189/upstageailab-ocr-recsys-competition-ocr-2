#!/usr/bin/env python3
"""
core_autopsy.py - Surgical Audit Tool for ocr/core

Analyzes ocr/core for:
1. Illegal Imports (from ocr.domains, ocr.features)
2. Domain Specific Key Terms (DBNet, CRAFT, PARSeq)
3. Complexity Metrics
"""

import ast
import os
import sys
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple

# Configuration
CORE_ROOT = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core")
DOMAINS = ["detection", "recognition", "classification", "kie"]
FORBIDDEN_PREFIXES = ["ocr.domains", "ocr.features"]
DOMAIN_KEYWORDS = [
    "DBNet", "CRAFT", "PARSeq", "CRNN", "SVTR",
    "ResNet", "MobileNet", "VGG", # Architectures might be shared, but suspicious if hardcoded
    "Polygon", "BBox", "Mbox", # Detection specific mostly
    "Accuracy", "NED", "CER", "WER", # Recognition metrics
    "ProbMap", "ThreshMap", "RegionMap", "AffinityMap" # Detection specific
]

def analyze_file(file_path: Path) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            content = f.read()
            tree = ast.parse(content)
        except Exception as e:
            return {"error": str(e)}

    imports = []
    illegal_imports = []

    # AST Import Analysis
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                for prefix in FORBIDDEN_PREFIXES:
                    if alias.name.startswith(prefix):
                        illegal_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                full_module = node.module
                imports.append(full_module)
                for prefix in FORBIDDEN_PREFIXES:
                    if full_module.startswith(prefix):
                        illegal_imports.append(full_module)

    # Keyword Analysis (Simple string matching for now)
    found_keywords = {}
    for kw in DOMAIN_KEYWORDS:
        count = len(re.findall(r"\b" + re.escape(kw) + r"\b", content, re.IGNORECASE))
        if count > 0:
            found_keywords[kw] = count

    return {
        "illegal_imports": illegal_imports,
        "domain_keywords": found_keywords,
        "loc": content.count("\n")
    }

def main():
    print(f"Starting Surgical Autopsy of {CORE_ROOT}...")

    results = {}

    for root, _, files in os.walk(CORE_ROOT):
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = Path(root) / file
            rel_path = file_path.relative_to(CORE_ROOT)

            # Skip valid infrastructure if needed (but we trust no one right now)
            # if "infrastructure" in str(rel_path): continue

            analysis = analyze_file(file_path)
            if "error" in analysis:
                print(f"ERROR processing {rel_path}: {analysis['error']}")
                continue

            if analysis["illegal_imports"] or analysis["domain_keywords"]:
                results[str(rel_path)] = analysis

    # Reporting
    report_path = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/analysis/surgical_audit_2026_01_21/AUTOPSY_REPORT.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# ocr/core Surgical Autopsy Report\n\n")
        f.write("| File | LOC | Illegal Imports | Domain Keywords |\n")
        f.write("|---|---|---|---|\n")

        for file, data in sorted(results.items(), key=lambda x: len(x[1]["illegal_imports"]) + len(x[1]["domain_keywords"]), reverse=True):
            imports_str = "<br>".join(data["illegal_imports"]) if data["illegal_imports"] else "Clean"
            keywords_str = "<br>".join([f"{k} ({v})" for k, v in data["domain_keywords"].items()]) if data["domain_keywords"] else "None"
            f.write(f"| `{file}` | {data['loc']} | {imports_str} | {keywords_str} |\n")

    print(f"Autopsy complete. Report generated at {report_path}")

if __name__ == "__main__":
    main()
