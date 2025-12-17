#!/usr/bin/env python3
"""
EDS v1.0 ALL-CAPS Filename Renamer

Renames ALL-CAPS files to comply with EDS v1.0 naming convention.
Pattern: YYYYMMDD_HHMM_{TYPE}_{slug}.md

Usage:
    python rename-all-caps-files.py --experiment 20251217_024343_image_enhancements
    python rename-all-caps-files.py --all
    python rename-all-caps-files.py --dry-run
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


def is_all_caps(filename: str) -> bool:
    """Check if filename is ALL-CAPS (excluding extension)."""
    name_without_ext = filename.replace('.md', '')
    return name_without_ext.replace('_', '').isupper()


def infer_type_from_content(file_path: Path) -> str:
    """Infer artifact type by reading frontmatter."""
    try:
        content = file_path.read_text(encoding='utf-8')

        # Check if frontmatter exists
        if content.startswith('---'):
            # Extract frontmatter
            match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
            if match:
                frontmatter = match.group(1)

                # Look for type field
                type_match = re.search(r'^type:\s*["\']?(\w+)["\']?', frontmatter, re.MULTILINE)
                if type_match:
                    return type_match.group(1)

        # Fallback to filename analysis
        filename_lower = file_path.name.lower()
        if any(kw in filename_lower for kw in ['summary', 'state', 'execution']):
            return 'assessment'
        elif any(kw in filename_lower for kw in ['guide', 'reference', 'instructions']):
            return 'guide'
        elif any(kw in filename_lower for kw in ['roadmap', 'plan']):
            return 'guide'
        else:
            return 'assessment'  # Default

    except Exception:
        return 'assessment'


def generate_new_filename(old_filename: str, artifact_type: str, timestamp: str) -> str:
    """
    Generate EDS v1.0 compliant filename.

    Pattern: YYYYMMDD_HHMM_{TYPE}_{slug}.md
    """
    # Remove extension and convert to lowercase
    name_without_ext = old_filename.replace('.md', '')

    # Convert to lowercase and replace underscores with hyphens
    slug = name_without_ext.lower().replace('_', '-')

    # Truncate if too long
    if len(slug) > 50:
        slug = slug[:50]

    # Build new filename
    new_filename = f"{timestamp}_{artifact_type}_{slug}.md"

    return new_filename


def rename_file(file_path: Path, experiment_id: str, dry_run: bool = False) -> tuple[bool, str, str]:
    """
    Rename ALL-CAPS file to EDS v1.0 compliant name.

    Returns:
        (renamed, old_name, new_name)
    """
    old_filename = file_path.name

    # Check if ALL-CAPS
    if not is_all_caps(old_filename):
        return False, old_filename, old_filename

    # Infer artifact type
    artifact_type = infer_type_from_content(file_path)

    # Generate timestamp (use experiment_id date if available, otherwise now)
    if experiment_id:
        # Extract date from experiment_id (YYYYMMDD_HHMMSS_name)
        match = re.match(r'^(\d{8})_(\d{6})', experiment_id)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)[:4]  # HHMM
            timestamp = f"{date_part}_{time_part}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Generate new filename
    new_filename = generate_new_filename(old_filename, artifact_type, timestamp)

    # Build new path
    new_path = file_path.parent / new_filename

    if not dry_run:
        file_path.rename(new_path)
        print(f"   ✅ Renamed: {old_filename}")
        print(f"      → {new_filename}")
    else:
        print(f"   [DRY RUN] Would rename: {old_filename}")
        print(f"             → {new_filename}")

    return True, old_filename, new_filename


def rename_experiment(experiment_dir: Path, dry_run: bool = False) -> dict:
    """
    Rename all ALL-CAPS files in an experiment.

    Returns:
        {
            'experiment_id': str,
            'renamed_count': int,
            'skipped_count': int,
            'renames': List[Tuple[str, str]]
        }
    """
    experiment_id = experiment_dir.name
    renamed_count = 0
    skipped_count = 0
    renames = []

    print(f"\n📂 Processing: {experiment_id}")

    # Find all markdown files
    md_files = list(experiment_dir.rglob('*.md'))

    for md_file in md_files:
        # Skip README.md
        if md_file.name == 'README.md':
            skipped_count += 1
            continue

        # Rename if ALL-CAPS
        was_renamed, old_name, new_name = rename_file(md_file, experiment_id, dry_run)

        if was_renamed:
            renamed_count += 1
            renames.append((old_name, new_name))
        else:
            if is_all_caps(old_name):
                # Shouldn't happen, but log it
                print(f"   ⚠️  Skipped ALL-CAPS file: {old_name}")
            skipped_count += 1

    return {
        'experiment_id': experiment_id,
        'renamed_count': renamed_count,
        'skipped_count': skipped_count,
        'renames': renames,
    }


def main():
    parser = argparse.ArgumentParser(description="Rename ALL-CAPS experiment artifacts")
    parser.add_argument("--experiment", help="Rename files in specific experiment ID")
    parser.add_argument("--all", action="store_true", help="Rename files in all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without making changes")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    ai_instructions_dir = script_dir.parent
    tracker_dir = ai_instructions_dir.parent
    experiments_dir = tracker_dir / "experiments"

    if not experiments_dir.exists():
        print(f"❌ ERROR: Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    # Determine which experiments to process
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

    print("🔧 EDS v1.0 ALL-CAPS Filename Renamer")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    print(f"📂 Experiments directory: {experiments_dir}")
    print(f"📊 Processing {len(experiment_dirs)} experiment(s)...")

    # Rename files in experiments
    results = []
    for exp_dir in sorted(experiment_dirs):
        result = rename_experiment(exp_dir, args.dry_run)
        results.append(result)

    # Summary
    total_renamed = sum(r['renamed_count'] for r in results)
    total_skipped = sum(r['skipped_count'] for r in results)

    print("\n" + "="*60)
    if args.dry_run:
        print("🔍 DRY RUN SUMMARY:")
    else:
        print("✅ RENAMING COMPLETE:")
    print(f"   Renamed: {total_renamed} files")
    print(f"   Skipped: {total_skipped} files (already compliant)")
    print(f"   Total: {total_renamed + total_skipped} files")

    if total_renamed > 0:
        print("\n📋 Renamed Files:")
        for result in results:
            if result['renamed_count'] > 0:
                print(f"\n   {result['experiment_id']}:")
                for old_name, new_name in result['renames']:
                    print(f"     • {old_name} → {new_name}")

    if not args.dry_run and total_renamed > 0:
        print("\n📊 Next steps:")
        print("   1. Run compliance report: python generate-compliance-report.py")
        print("   2. Test pre-commit hooks: git add . && git commit -m 'Rename ALL-CAPS files'")
        print("   3. Verify all experiments at 100% compliance")


if __name__ == '__main__':
    main()
