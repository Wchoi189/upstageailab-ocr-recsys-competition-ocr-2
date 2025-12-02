#!/usr/bin/env python3
"""
Reindex artifacts: regenerate INDEX.md files deterministically.
Scans docs/artifacts/** by type and builds stable, sorted indexes.
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

def extract_frontmatter(file_path: Path) -> Dict[str, str]:
    """Extract frontmatter from artifact markdown file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return {}
        
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}
        
        fm = {}
        for line in parts[1].strip().split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                fm[key.strip()] = val.strip().strip('"')
        return fm
    except Exception:
        return {}

def generate_index_content(directory: Path, artifact_type: str) -> str:
    """Generate INDEX.md content for a directory."""
    files = sorted([f for f in directory.glob("*.md") if f.name != "INDEX.md"])
    
    # Group by status
    active = []
    completed = []
    other = []
    
    for f in files:
        fm = extract_frontmatter(f)
        title = fm.get("title", f.stem)
        date = fm.get("date", "")
        status = fm.get("status", "unknown")
        type_val = fm.get("type", "")
        
        # Extract first line of content as preview
        try:
            content = f.read_text(encoding="utf-8")
            lines = content.split("---", 2)
            if len(lines) >= 3:
                body = lines[2].strip().split("\n")
                preview = next((line.strip("# ").strip() for line in body if line.strip() and not line.startswith("#")), "")[:100]
            else:
                preview = ""
        except Exception:
            preview = ""
        
        entry = f"- [{title}]({f.name}) (📅 {date}, 📄 {type_val}) - {preview}"
        
        if status in ["active", "draft"]:
            active.append(entry)
        elif status in ["completed", "archived"]:
            completed.append(entry)
        else:
            other.append(entry)
    
    # Build content
    type_title = artifact_type.replace("_", " ").title()
    lines = [
        f"# {type_title}",
        "",
        f"Active {artifact_type.lower()} and development roadmaps.",
        "",
        f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Artifacts**: {len(files)}",
        ""
    ]
    
    if active:
        lines.append(f"## Active ({len(active)})")
        lines.append("")
        lines.extend(active)
        lines.append("")
    
    if completed:
        lines.append(f"## Completed ({len(completed)})")
        lines.append("")
        lines.extend(completed)
        lines.append("")
    
    if other:
        lines.append(f"## Other ({len(other)})")
        lines.append("")
        lines.extend(other)
        lines.append("")
    
    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Active | {len(active)} |")
    lines.append(f"| Completed | {len(completed)} |")
    if other:
        lines.append(f"| Other | {len(other)} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This index is automatically generated. Do not edit manually.*")
    
    return "\n".join(lines) + "\n"

def main():
    """Main entry point."""
    from AgentQMS.agent_tools.utils.paths import get_artifacts_dir
    
    artifacts_root = get_artifacts_dir()
    print(f"🔄 Reindexing artifacts in {artifacts_root}")
    
    # Map directories to artifact types
    type_map = {
        "implementation_plans": "Implementation Plans",
        "assessments": "Assessments",
        "audits": "Audits",
        "bug_reports": "Bug Reports",
        "research": "Research",
        "design_documents": "Design Documents",
        "templates": "Templates",
        "completed_plans": "Completed Plans",
        "completed_plans/completion_summaries": "Completion Summaries",
        "completed_plans/completion_summaries/session_notes": "Session Notes",
    }
    
    regenerated = 0
    for rel_path, type_name in type_map.items():
        target_dir = artifacts_root / rel_path
        if not target_dir.exists():
            continue
        
        index_file = target_dir / "INDEX.md"
        content = generate_index_content(target_dir, type_name)
        index_file.write_text(content, encoding="utf-8")
        print(f"  ✅ {rel_path}/INDEX.md")
        regenerated += 1
    
    print(f"\n✨ Regenerated {regenerated} index files")
    return 0

if __name__ == "__main__":
    sys.exit(main())
