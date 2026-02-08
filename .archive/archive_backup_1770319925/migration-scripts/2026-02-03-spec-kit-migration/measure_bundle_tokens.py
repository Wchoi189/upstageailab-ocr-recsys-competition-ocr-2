#!/usr/bin/env python3
"""
Token Budget Measurement Tool for Context Bundles
Measures actual token usage of context bundles using tiktoken
"""

import tiktoken
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Go up from archive/migration-scripts/2026-02-03-spec-kit-migration/
BUNDLES_DIR = PROJECT_ROOT / "AgentQMS" / ".agentqms" / "plugins" / "context_bundles"

# Token budget constraints
MAX_TOKENS = 3000
MAX_FILES = 6

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def read_file_content(file_path: Path) -> str:
    """Safely read file content, return empty string if file doesn't exist."""
    try:
        if file_path.is_file():
            return file_path.read_text(encoding='utf-8')
        elif file_path.is_dir():
            # For directories, just return empty (we can't measure their full content)
            return ""
    except Exception as e:
        print(f"  Warning: Could not read {file_path}: {e}")
    return ""

def measure_bundle(bundle_path: Path) -> Dict:
    """Measure token usage and file count for a single bundle."""
    with open(bundle_path, 'r', encoding='utf-8') as f:
        bundle = yaml.safe_load(f)

    bundle_name = bundle.get('name', bundle_path.stem)
    total_tokens = 0
    total_files = 0
    tier_stats = {}

    tiers = bundle.get('tiers', {})
    for tier_name, tier_config in tiers.items():
        tier_tokens = 0
        tier_files = 0
        files = tier_config.get('files', [])

        for file_entry in files:
            if isinstance(file_entry, dict):
                file_path_str = file_entry.get('path', '')
            else:
                file_path_str = file_entry

            # Resolve path relative to project root
            file_path = PROJECT_ROOT / file_path_str
            content = read_file_content(file_path)

            if content:
                tokens = count_tokens(content)
                tier_tokens += tokens
                tier_files += 1

        tier_stats[tier_name] = {
            'tokens': tier_tokens,
            'files': tier_files
        }
        total_tokens += tier_tokens
        total_files += tier_files

    return {
        'name': bundle_name,
        'total_tokens': total_tokens,
        'total_files': total_files,
        'tier_stats': tier_stats,
        'exceeds_token_limit': total_tokens > MAX_TOKENS,
        'exceeds_file_limit': total_files > MAX_FILES,
        'compliant': total_tokens <= MAX_TOKENS and total_files <= MAX_FILES
    }

def generate_report(results: List[Dict]) -> str:
    """Generate markdown report of bundle token usage."""
    compliant_count = sum(1 for r in results if r['compliant'])
    total_count = len(results)

    report = [
        "# Context Bundle Token Budget Report",
        f"\n**Generated**: {Path(__file__).name}",
        f"**Budget Constraints**: <{MAX_TOKENS} tokens, ≤{MAX_FILES} files per bundle\n",
        f"## Summary\n",
        f"- **Total Bundles**: {total_count}",
        f"- **Compliant**: {compliant_count} / {total_count} ({compliant_count/total_count*100:.1f}%)",
        f"- **Non-Compliant**: {total_count - compliant_count}\n",
        "## Bundle Details\n",
        "| Bundle | Total Tokens | Total Files | Status |\n",
        "|:-------|-------------:|------------:|:------:|"
    ]

    # Sort by compliance (non-compliant first), then by token count
    sorted_results = sorted(results, key=lambda x: (x['compliant'], -x['total_tokens']))

    for result in sorted_results:
        status = "✅" if result['compliant'] else "❌"
        report.append(
            f"| {result['name']:<30} | {result['total_tokens']:>11} | "
            f"{result['total_files']:>11} | {status:^6} |"
        )

    # Detailed breakdown for non-compliant bundles
    non_compliant = [r for r in results if not r['compliant']]
    if non_compliant:
        report.append("\n## Non-Compliant Bundles (Detailed)\n")
        for result in non_compliant:
            report.append(f"### {result['name']}")
            report.append(f"- **Total Tokens**: {result['total_tokens']} (limit: {MAX_TOKENS})")
            report.append(f"- **Total Files**: {result['total_files']} (limit: {MAX_FILES})")

            if result['exceeds_token_limit']:
                excess = result['total_tokens'] - MAX_TOKENS
                report.append(f"- **⚠️ Token Excess**: +{excess} tokens ({excess/MAX_TOKENS*100:.1f}% over)")

            if result['exceeds_file_limit']:
                excess = result['total_files'] - MAX_FILES
                report.append(f"- **⚠️ File Excess**: +{excess} files")

            report.append("\n**Per-Tier Breakdown**:")
            for tier_name, stats in result['tier_stats'].items():
                report.append(f"- `{tier_name}`: {stats['tokens']} tokens, {stats['files']} files")
            report.append("")

    return "\n".join(report)

def main():
    """Measure all bundles and generate report."""
    print(f"Measuring bundles in: {BUNDLES_DIR}")
    print(f"Budget constraints: <{MAX_TOKENS} tokens, ≤{MAX_FILES} files\n")

    bundle_files = sorted(BUNDLES_DIR.glob("*.yaml"))
    if not bundle_files:
        print(f"❌ No bundle files found in {BUNDLES_DIR}")
        return

    results = []
    for bundle_file in bundle_files:
        if bundle_file.name == '.gitignore':
            continue

        print(f"Measuring: {bundle_file.name}")
        result = measure_bundle(bundle_file)
        results.append(result)

        status = "✅" if result['compliant'] else "❌"
        print(f"  {status} {result['total_tokens']} tokens, {result['total_files']} files")

    # Generate and save report
    report = generate_report(results)
    report_path = PROJECT_ROOT / "scripts" / "bundle_token_report.md"
    report_path.write_text(report, encoding='utf-8')

    print(f"\n📊 Report saved to: {report_path}")

    # Print summary
    compliant_count = sum(1 for r in results if r['compliant'])
    print(f"\n✅ {compliant_count}/{len(results)} bundles compliant")

    non_compliant = [r for r in results if not r['compliant']]
    if non_compliant:
        print(f"❌ {len(non_compliant)} bundles exceed limits:")
        for r in non_compliant:
            print(f"   - {r['name']}: {r['total_tokens']} tokens, {r['total_files']} files")

if __name__ == "__main__":
    main()
