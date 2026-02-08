#!/usr/bin/env python3
"""
Triage context bundle broken references into categories A/B/C
Category A: Intentionally pruned (delete ref)
Category B: Consolidated into specs (update ref)
Category C: Actually missing (restore)
"""

import json
import re
import yaml
from pathlib import Path
from collections import defaultdict

# Paths
BUNDLES_DIR = Path("AgentQMS/.agentqms/plugins/context_bundles")
SPECS_DIR = Path("AgentQMS/specs")
STANDARDS_DB = Path("AgentQMS/.agentqms/standards_db.json")
ARCHIVE_DIR = Path("archive/legacy_standards_dump")

def load_standards_db():
    """Load standards_db.json"""
    with open(STANDARDS_DB) as f:
        return json.load(f)

def get_all_broken_refs():
    """Scan all bundles for broken references"""
    broken_refs = defaultdict(list)

    for bundle_file in BUNDLES_DIR.glob("*.yaml"):
        with open(bundle_file) as f:
            try:
                bundle = yaml.safe_load(f)
            except:
                print(f"⚠️  Failed to parse {bundle_file.name}")
                continue

        # Extract all file references
        refs = extract_references(bundle)

        for ref in refs:
            if not Path(ref).exists():
                broken_refs[ref].append(bundle_file.name)

    return broken_refs

def extract_references(obj, refs=None):
    """Recursively extract path references from YAML structure"""
    if refs is None:
        refs = []

    if isinstance(obj, dict):
        if 'path' in obj:
            refs.append(obj['path'])
        for value in obj.values():
            extract_references(value, refs)
    elif isinstance(obj, list):
        for item in obj:
            extract_references(item, refs)

    return refs

def categorize_reference(ref, standards_db):
    """Categorize broken reference into A/B/C"""

    # Extract basename for matching
    basename = Path(ref).name
    stem = Path(ref).stem

    # Check if in standards_db
    in_db = any(stem in key or basename in key for key in standards_db.keys())

    # Check if consolidated into specs
    spec_mapping = {
        "tool-catalog.yaml": "tier2-framework/discovery.spec.md",
        "utility-scripts-manifest.yaml": "tier2-framework/discovery.spec.md",
        "hydra-v5-rules.yaml": "tier2-framework/runtime.spec.md",
        "naming-conventions.yaml": "tier1-contracts/compliance.spec.md",
        "system-architecture.yaml": "tier1-contracts/architecture.spec.md",
        # Add more mappings as discovered
    }

    if basename in spec_mapping:
        new_location = f"AgentQMS/specs/{spec_mapping[basename]}"
        if Path(new_location).exists():
            return "B", new_location

    # Check if in archive
    if ARCHIVE_DIR.exists():
        # Use glob instead of rglob to avoid pattern issues
        archive_matches = list(ARCHIVE_DIR.glob(f"{basename}")) + \
                         list(ARCHIVE_DIR.glob(f"**/{basename}"))
        if archive_matches:
            # Exists in archive - check if it was verbose/fluff
            if is_likely_fluff(basename):
                return "A", "INTENTIONALLY_REMOVED"
            else:
                return "C", f"RESTORE_FROM_ARCHIVE"

    # Not in DB, not in specs, not in archive
    if in_db:
        return "C", "RESTORE_FROM_STANDARDS_DB"
    else:
        return "A", "DELETE_REF"

def is_likely_fluff(basename):
    """Heuristic to determine if file was likely fluff"""
    fluff_indicators = [
        "tutorial", "example", "guide", "verbose",
        "documentation", "walkthrough", "reference-only"
    ]
    return any(indicator in basename.lower() for indicator in fluff_indicators)

def main():
    print("=== Context Bundle Broken Reference Triage ===\n")

    # Load data
    print("📂 Loading standards_db.json...")
    standards_db = load_standards_db()

    print("🔍 Scanning bundles for broken references...")
    broken_refs = get_all_broken_refs()

    print(f"\n📊 Found {len(broken_refs)} unique broken references\n")

    # Categorize
    results = []
    category_counts = defaultdict(int)

    for ref, bundles in sorted(broken_refs.items()):
        category, new_location = categorize_reference(ref, standards_db)
        category_counts[category] += 1

        results.append({
            "broken_ref": ref,
            "category": category,
            "new_location": new_location,
            "affected_bundles": bundles,
            "bundle_count": len(bundles)
        })

    # Print summary
    print("=" * 80)
    print("CATEGORY SUMMARY")
    print("=" * 80)
    print(f"Type A (Delete):  {category_counts['A']:3d} refs  (intentionally pruned)")
    print(f"Type B (Update):  {category_counts['B']:3d} refs  (consolidated to specs)")
    print(f"Type C (Restore): {category_counts['C']:3d} refs  (actually missing)")
    print(f"{'Total:':<14} {sum(category_counts.values()):3d} refs")
    print()

    # Print detailed breakdown
    print("=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)

    for category in ["B", "C", "A"]:  # B first (easiest), then C, then A
        cat_results = [r for r in results if r["category"] == category]
        if not cat_results:
            continue

        cat_name = {"A": "DELETE", "B": "UPDATE", "C": "RESTORE"}[category]
        print(f"\n### Category {category}: {cat_name} ({len(cat_results)} refs)")
        print("-" * 80)

        for r in sorted(cat_results, key=lambda x: -x["bundle_count"]):
            print(f"\n📄 {Path(r['broken_ref']).name}")
            print(f"   Ref: {r['broken_ref']}")
            print(f"   → {r['new_location']}")
            print(f"   Affects {r['bundle_count']} bundle(s): {', '.join(r['affected_bundles'][:3])}" +
                  ("..." if r['bundle_count'] > 3 else ""))

    # Export to CSV
    csv_path = Path("broken_refs_triage.csv")
    with open(csv_path, "w") as f:
        f.write("broken_ref,category,new_location,bundle_count,affected_bundles\n")
        for r in results:
            bundles_str = ";".join(r["affected_bundles"])
            f.write(f'"{r["broken_ref"]}",{r["category"]},"{r["new_location"]}",'
                   f'{r["bundle_count"]},"{bundles_str}"\n')

    print(f"\n\n✅ Triage results exported to: {csv_path}")
    print(f"\nNext steps:")
    print(f"  1. Review triage results")
    print(f"  2. Adjust spec_mapping in this script if needed")
    print(f"  3. Run fix scripts for each category")

if __name__ == "__main__":
    main()
