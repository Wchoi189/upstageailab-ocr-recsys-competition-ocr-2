#!/usr/bin/env python3
"""
EDS v1.0 Artifact Fixer

Automatically adds EDS v1.0 compliant frontmatter to legacy experiment artifacts.
Infers type from directory structure and filename patterns.

Usage:
    python fix-legacy-artifacts.py --experiment 20251217_024343_image_enhancements
    python fix-legacy-artifacts.py --all
    python fix-legacy-artifacts.py --dry-run
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path


def infer_artifact_type(file_path: Path, experiment_dir: Path) -> str:
    """Infer artifact type from directory structure and filename."""
    relative_path = file_path.relative_to(experiment_dir)
    parts = relative_path.parts

    # Check parent directory
    if len(parts) > 1:
        parent_dir = parts[-2]
        if parent_dir == 'assessments':
            return 'assessment'
        elif parent_dir == 'reports':
            return 'report'
        elif parent_dir == 'guides':
            return 'guide'
        elif parent_dir == 'scripts':
            return 'script'
        elif parent_dir == 'artifacts':
            # Try to infer from filename
            filename_lower = file_path.name.lower()
            if any(keyword in filename_lower for keyword in ['test', 'result', 'analysis', 'failure', 'implementation']):
                return 'assessment'
            elif any(keyword in filename_lower for keyword in ['metric', 'comparison', 'performance']):
                return 'report'
            elif any(keyword in filename_lower for keyword in ['guide', 'readme', 'setup', 'start']):
                return 'guide'
            else:
                return 'assessment'  # Default

    # Fallback to filename analysis
    filename_lower = file_path.name.lower()
    if any(keyword in filename_lower for keyword in ['guide', 'readme', 'setup', 'quick-start']):
        return 'guide'
    elif any(keyword in filename_lower for keyword in ['test', 'result', 'analysis', 'failure', 'summary', 'implementation']):
        return 'assessment'
    elif any(keyword in filename_lower for keyword in ['metric', 'performance', 'comparison']):
        return 'report'
    else:
        return 'assessment'  # Default


def extract_tags_from_filename(filename: str, experiment_id: str) -> list:
    """Extract relevant tags from filename and experiment ID."""
    tags = []

    # Extract experiment focus from experiment_id
    if 'perspective_correction' in experiment_id:
        tags.append('perspective-correction')
    elif 'image_enhancements' in experiment_id:
        tags.append('image-enhancements')

    # Extract keywords from filename
    filename_lower = filename.lower().replace('.md', '').replace('_', '-')

    keywords = {
        'failure': 'failure-analysis',
        'test': 'testing',
        'analysis': 'analysis',
        'implementation': 'implementation',
        'setup': 'setup',
        'guide': 'guide',
        'result': 'results',
        'metric': 'metrics',
        'performance': 'performance',
        'optimization': 'optimization',
        'rembg': 'rembg',
        'pipeline': 'pipeline',
        'gpu': 'gpu',
    }

    for keyword, tag in keywords.items():
        if keyword in filename_lower and tag not in tags:
            tags.append(tag)

    # Ensure at least one tag
    if not tags:
        tags.append('experiment')

    return tags


def generate_frontmatter(artifact_type: str, experiment_id: str, filename: str) -> str:
    """Generate EDS v1.0 compliant frontmatter."""
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    tags = extract_tags_from_filename(filename, experiment_id)

    frontmatter = f"""---
ads_version: "1.0"
type: "{artifact_type}"
experiment_id: "{experiment_id}"
status: "complete"
created: "{now}"
updated: "{now}"
tags: {tags}
"""

    # Add type-specific fields
    if artifact_type == 'assessment':
        frontmatter += """phase: "phase_0"
priority: "medium"
evidence_count: 0
"""
    elif artifact_type == 'report':
        frontmatter += """metrics: {}
baseline: "unknown"
comparison: "neutral"
"""
    elif artifact_type == 'guide':
        frontmatter += """commands: []
prerequisites: []
"""
    elif artifact_type == 'script':
        frontmatter += """dependencies: []
"""

    frontmatter += "---\n"
    return frontmatter


def fix_artifact(file_path: Path, experiment_id: str, dry_run: bool = False) -> bool:
    """
    Add frontmatter to artifact if missing.

    Returns:
        True if fixed, False if already compliant
    """
    content = file_path.read_text(encoding='utf-8')

    # Check if frontmatter already exists
    if content.startswith('---'):
        # Already has frontmatter, skip
        return False

    # Infer artifact type
    experiment_dir = file_path
    while experiment_dir.name != 'experiments' and experiment_dir.parent != experiment_dir:
        experiment_dir = experiment_dir.parent
    experiment_dir = experiment_dir / experiment_id

    artifact_type = infer_artifact_type(file_path, experiment_dir)

    # Generate frontmatter
    frontmatter = generate_frontmatter(artifact_type, experiment_id, file_path.name)

    # Add frontmatter to content
    new_content = frontmatter + "\n" + content

    if not dry_run:
        file_path.write_text(new_content, encoding='utf-8')
        print(f"   ✅ Fixed: {file_path.name} (added {artifact_type} frontmatter)")
    else:
        print(f"   [DRY RUN] Would fix: {file_path.name} (add {artifact_type} frontmatter)")

    return True


def fix_experiment(experiment_dir: Path, dry_run: bool = False) -> dict:
    """
    Fix all artifacts in an experiment.

    Returns:
        {
            'experiment_id': str,
            'fixed_count': int,
            'skipped_count': int,
        }
    """
    experiment_id = experiment_dir.name
    fixed_count = 0
    skipped_count = 0

    print(f"\n📂 Processing: {experiment_id}")

    # Find all markdown files
    md_files = list(experiment_dir.rglob('*.md'))

    for md_file in md_files:
        # Skip README.md
        if md_file.name == 'README.md':
            skipped_count += 1
            continue

        # Fix artifact
        was_fixed = fix_artifact(md_file, experiment_id, dry_run)

        if was_fixed:
            fixed_count += 1
        else:
            skipped_count += 1
            print(f"   ⏭️  Skipped: {md_file.name} (already has frontmatter)")

    return {
        'experiment_id': experiment_id,
        'fixed_count': fixed_count,
        'skipped_count': skipped_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Fix legacy experiment artifacts")
    parser.add_argument("--experiment", help="Fix specific experiment ID")
    parser.add_argument("--all", action="store_true", help="Fix all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    ai_instructions_dir = script_dir.parent
    tracker_dir = ai_instructions_dir.parent
    experiments_dir = tracker_dir / "experiments"

    if not experiments_dir.exists():
        print(f"❌ ERROR: Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    # Determine which experiments to fix
    if args.experiment:
        experiment_dirs = [experiments_dir / args.experiment]
        if not experiment_dirs[0].exists():
            print(f"❌ ERROR: Experiment not found: {args.experiment}")
            sys.exit(1)
    elif args.all:
        experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    else:
        print("❌ ERROR: Must specify --experiment <id> or --all")
        sys.exit(1)

    print("🔧 EDS v1.0 Legacy Artifact Fixer")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    print(f"📂 Experiments directory: {experiments_dir}")
    print(f"📊 Processing {len(experiment_dirs)} experiment(s)...")

    # Fix experiments
    results = []
    for exp_dir in sorted(experiment_dirs):
        result = fix_experiment(exp_dir, args.dry_run)
        results.append(result)

    # Summary
    total_fixed = sum(r['fixed_count'] for r in results)
    total_skipped = sum(r['skipped_count'] for r in results)

    print("\n" + "="*60)
    if args.dry_run:
        print("🔍 DRY RUN SUMMARY:")
    else:
        print("✅ FIXING COMPLETE:")
    print(f"   Fixed: {total_fixed} artifacts")
    print(f"   Skipped: {total_skipped} artifacts (already compliant or README)")
    print(f"   Total: {total_fixed + total_skipped} artifacts")

    if not args.dry_run and total_fixed > 0:
        print("\n📊 Next steps:")
        print("   1. Run compliance report: python generate-compliance-report.py")
        print("   2. Review fixed artifacts and adjust frontmatter as needed")
        print("   3. Commit changes: git add . && git commit -m 'Add EDS v1.0 frontmatter to legacy artifacts'")


if __name__ == '__main__':
    main()
