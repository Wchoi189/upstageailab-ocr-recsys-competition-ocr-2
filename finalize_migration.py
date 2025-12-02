#!/usr/bin/env python3
"""Finalize AgentQMS migration - move remaining components."""

import os
import shutil
import sys

os.chdir('/workspaces/upstageailab-ocr-recsys-competition-ocr-2')
base = os.getcwd()

def move_item(src_rel, dst_rel, desc):
    src = os.path.join(base, src_rel)
    dst = os.path.join(base, dst_rel)

    if not os.path.exists(src):
        print(f"SKIP: {desc} - source not found: {src_rel}")
        return False

    if os.path.exists(dst):
        print(f"SKIP: {desc} - destination exists: {dst_rel}")
        return False

    try:
        shutil.move(src, dst)
        print(f"MOVED: {desc} - {src_rel} -> {dst_rel}")
        return True
    except Exception as e:
        print(f"ERROR: {desc} - {e}")
        return False

print("=" * 60)
print("Finalizing AgentQMS Migration")
print("=" * 60)

# Move main directories
moved_count = 0
for comp in ['agent_tools', 'interface', 'knowledge', 'toolkit']:
    if move_item(f'__NEW__/AgentQMS/{comp}', f'AgentQMS/{comp}', comp):
        moved_count += 1

# Move CHANGELOG
if move_item('__NEW__/AgentQMS/CHANGELOG.md', 'AgentQMS/CHANGELOG.md', 'CHANGELOG.md'):
    moved_count += 1

# Move conventions items
for item in ['audit_framework', 'templates', 'q-manifest.yaml']:
    if move_item(f'__NEW__/AgentQMS/conventions/{item}',
                 f'AgentQMS/conventions/{item}',
                 f'conventions/{item}'):
        moved_count += 1

# Handle dot directories
for dot_dir in ['.copilot', '.qwen', '.cursor']:
    src = os.path.join(base, '__NEW__', dot_dir)
    dst = os.path.join(base, dot_dir)

    if not os.path.exists(src):
        print(f"SKIP: {dot_dir} - source not found")
        continue

    if os.path.exists(dst):
        print(f"MERGE: {dot_dir} - merging with existing...")
        merged = 0
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)
            if not os.path.exists(dst_item):
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dst_item)
                else:
                    shutil.copytree(src_item, dst_item)
                print(f"  ADDED: {item} to {dot_dir}")
                merged += 1
            else:
                print(f"  SKIP: {item} already exists in {dot_dir}")
        if merged > 0:
            shutil.rmtree(src)
            print(f"MERGED: {dot_dir} ({merged} items) and removed source")
        else:
            shutil.rmtree(src)
            print(f"REMOVED: {dot_dir} source (no new items)")
    else:
        if move_item(f'__NEW__/{dot_dir}', dot_dir, dot_dir):
            moved_count += 1

print("=" * 60)
print(f"Migration complete! Moved {moved_count} items.")
print("=" * 60)
sys.exit(0)
