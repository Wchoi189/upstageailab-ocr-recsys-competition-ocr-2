#!/usr/bin/env python3
"""Complete AgentQMS framework migration - move all components."""

import shutil
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    new_dir = project_root / "__NEW__"
    agentqms_root = project_root / "AgentQMS"

    print("=" * 60)
    print("AgentQMS Framework Migration")
    print("=" * 60)

    # Components to move
    components = [
        "agent_tools",
        "interface",
        "knowledge",
        "toolkit",
    ]

    files_to_move = [
        "CHANGELOG.md",
    ]

    # Move directories
    for component in components:
        source = new_dir / "AgentQMS" / component
        target = agentqms_root / component

        if not source.exists():
            print(f"⚠️  {component} not found, skipping")
            continue

        if target.exists():
            print(f"⚠️  {component} already exists, skipping")
            continue

        print(f"📦 Moving {component}...")
        try:
            shutil.move(str(source), str(target))
            print(f"   ✓ Moved {component}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            sys.exit(1)

    # Move files
    for filename in files_to_move:
        source = new_dir / "AgentQMS" / filename
        target = agentqms_root / filename

        if not source.exists():
            print(f"⚠️  {filename} not found, skipping")
            continue

        if target.exists():
            print(f"⚠️  {filename} already exists, skipping")
            continue

        print(f"📦 Moving {filename}...")
        try:
            shutil.move(str(source), str(target))
            print(f"   ✓ Moved {filename}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Move conventions subdirectories
    print("\n📦 Moving conventions subdirectories...")
    conventions_new = new_dir / "AgentQMS" / "conventions"
    conventions_root = agentqms_root / "conventions"

    if conventions_new.exists():
        items = ["audit_framework", "templates", "q-manifest.yaml"]
        for item in items:
            source = conventions_new / item
            target = conventions_root / item

            if not source.exists():
                continue

            if target.exists():
                print(f"   ⚠️  {item} already exists, skipping")
                continue

            try:
                shutil.move(str(source), str(target))
                print(f"   ✓ Moved conventions/{item}")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    # Move dot directories
    print("\n📦 Moving dot directories...")
    dot_dirs = [".copilot", ".qwen", ".cursor"]

    for dot_dir in dot_dirs:
        source = new_dir / dot_dir
        target = project_root / dot_dir

        if not source.exists():
            print(f"   ⚠️  {dot_dir} not found, skipping")
            continue

        if target.exists():
            print(f"   ⚠️  {dot_dir} already exists, merging...")
            # Merge: copy new files that don't exist
            for item in source.iterdir():
                src_item = source / item.name
                dst_item = target / item.name

                if not dst_item.exists():
                    if src_item.is_file():
                        shutil.copy2(src_item, dst_item)
                        print(f"   ✓ Added {item.name}")
                    elif src_item.is_dir():
                        shutil.copytree(src_item, dst_item)
                        print(f"   ✓ Added directory {item.name}")
                else:
                    print(f"   ⚠️  {item.name} exists, skipping")
            # Remove source after merging
            try:
                shutil.rmtree(source)
                print(f"   ✓ Removed source {dot_dir}")
            except Exception as e:
                print(f"   ⚠️  Could not remove {dot_dir}: {e}")
        else:
            try:
                shutil.move(str(source), str(target))
                print(f"   ✓ Moved {dot_dir}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
