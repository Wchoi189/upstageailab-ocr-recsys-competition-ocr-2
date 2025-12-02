#!/usr/bin/env python3
"""
Check Markdown links: verify artifact references are valid.
Scans docs/** and AgentQMS/** for [text](path) links and validates targets exist.

Phase 3: Includes inline artifact reference parsing for comprehensive link integrity.
"""
import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any


def extract_markdown_links(file_path: Path) -> List[Tuple[int, str, str]]:
    """Extract [text](url) links from markdown file. Returns (line_num, text, url)."""
    links = []
    try:
        content = file_path.read_text(encoding="utf-8")
        for i, line in enumerate(content.split("\n"), 1):
            for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', line):
                text, url = match.groups()
                links.append((i, text, url))
    except Exception:
        pass
    return links


def extract_artifact_references(file_path: Path, artifacts_root: Path) -> List[Tuple[int, str]]:
    """Extract inline artifact references (paths to artifacts mentioned in text).
    
    Looks for patterns like:
    - ../assessments/2025-11-29_1810_assessment-...md
    - docs/artifacts/implementation_plans/...
    - Reference: filename.md
    """
    references = []
    try:
        content = file_path.read_text(encoding="utf-8")
        for i, line in enumerate(content.split("\n"), 1):
            # Pattern for artifact file references in text
            artifact_patterns = [
                r'(?:Reference|See|See also|Ref|Related):\s*([^\s]+\.md)',
                r'(?:\.\.\/)+[a-z_]+\/\d{4}-\d{2}-\d{2}_\d{4}_[a-zA-Z_-]+\.md',
                r'docs/artifacts/[a-z_]+/\d{4}-\d{2}-\d{2}_\d{4}_[a-zA-Z_-]+\.md',
            ]
            for pattern in artifact_patterns:
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    ref = match.group(1) if match.lastindex else match.group(0)
                    references.append((i, ref))
    except Exception:
        pass
    return references


def resolve_link(source_file: Path, link_url: str) -> Path | None:
    """Resolve relative markdown link to absolute path."""
    # Skip external URLs, anchors, mailto
    if any(link_url.startswith(p) for p in ["http://", "https://", "#", "mailto:"]):
        return None
    
    # Handle fragment-only anchors
    if link_url.startswith("#"):
        return None
    
    # Remove fragment if present
    if "#" in link_url:
        link_url = link_url.split("#")[0]
    
    if not link_url:
        return None
    
    # Resolve relative to source file's directory
    target = (source_file.parent / link_url).resolve()
    return target


def check_links_in_directory(
    directory: Path,
    project_root: Path,
    check_artifacts_only: bool = False
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Check all links in a directory recursively.
    
    Returns (checked_files, total_links, broken_links)
    """
    broken_links = []
    checked_files = 0
    total_links = 0
    
    for md_file in directory.rglob("*.md"):
        if ".git" in str(md_file):
            continue
        
        # Skip certain directories
        skip_dirs = ["node_modules", "__pycache__", ".venv", "venv"]
        if any(skip_dir in str(md_file) for skip_dir in skip_dirs):
            continue
        
        checked_files += 1
        links = extract_markdown_links(md_file)
        
        for line_num, text, url in links:
            total_links += 1
            
            # If checking artifacts only, skip non-artifact links
            if check_artifacts_only:
                if not any(x in url for x in ["artifacts", "docs/", ".md"]):
                    continue
            
            target = resolve_link(md_file, url)
            
            if target is None:
                # External or anchor link, skip
                continue
            
            if not target.exists():
                try:
                    rel_source = md_file.relative_to(project_root)
                except ValueError:
                    rel_source = md_file
                
                try:
                    resolved_str = str(target.relative_to(project_root))
                except ValueError:
                    resolved_str = str(target)
                
                broken_links.append({
                    "file": str(rel_source),
                    "line": line_num,
                    "text": text,
                    "url": url,
                    "resolved": resolved_str
                })
    
    return checked_files, total_links, broken_links


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check Markdown links in documentation")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--artifacts-only", action="store_true", 
                        help="Only check links to artifact files")
    parser.add_argument("--no-agentqms", action="store_false", dest="include_agentqms",
                        help="Exclude AgentQMS directory from link checking (default: included)")
    args = parser.parse_args()
    
    from AgentQMS.agent_tools.utils.paths import get_project_root
    
    project_root = get_project_root()
    docs_dir = project_root / "docs"
    agentqms_dir = project_root / "AgentQMS"
    
    total_checked = 0
    total_links = 0
    all_broken_links = []
    
    # Check docs directory
    if docs_dir.exists():
        checked, links, broken = check_links_in_directory(
            docs_dir, project_root, args.artifacts_only
        )
        total_checked += checked
        total_links += links
        all_broken_links.extend(broken)
    else:
        if not args.json:
            print("⚠️  docs/ directory not found")
    
    # Check AgentQMS directory if requested
    if args.include_agentqms and agentqms_dir.exists():
        checked, links, broken = check_links_in_directory(
            agentqms_dir, project_root, args.artifacts_only
        )
        total_checked += checked
        total_links += links
        all_broken_links.extend(broken)
    
    if args.json:
        result = {
            "checked_files": total_checked,
            "total_links": total_links,
            "broken_links": all_broken_links,
            "status": "fail" if all_broken_links else "pass"
        }
        print(json.dumps(result, indent=2))
    else:
        print("🔍 Checking links in documentation")
        print(f"\n📊 Checked {total_checked} files, {total_links} links")
        
        if all_broken_links:
            print(f"\n❌ Found {len(all_broken_links)} broken links:\n")
            for link in all_broken_links:
                print(f"  {link['file']}:{link['line']}")
                print(f"    [{link['text']}]({link['url']})")
                print(f"    ⚠️  Target not found: {link['resolved']}\n")
        else:
            print("\n✅ All links valid")
    
    return 1 if all_broken_links else 0


if __name__ == "__main__":
    sys.exit(main())
