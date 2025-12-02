#!/usr/bin/env python3
"""Force move framework using copy+remove fallback."""

import os
import shutil
import sys

base = '/workspaces/upstageailab-ocr-recsys-competition-ocr-2'
os.chdir(base)

def force_move(src, dst, desc):
    src_path = os.path.join(base, src)
    dst_path = os.path.join(base, dst)

    if not os.path.exists(src_path):
        print(f"SKIP {desc}: Source not found")
        return False

    if os.path.exists(dst_path):
        print(f"SKIP {desc}: Destination exists")
        return False

    try:
        # Try move first
        shutil.move(src_path, dst_path)
        print(f"MOVED {desc}")
        return True
    except OSError:
        # Fallback to copy + remove
        try:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            shutil.rmtree(src_path) if os.path.isdir(src_path) else os.remove(src_path)
            print(f"COPIED+REMOVED {desc}")
            return True
        except Exception as e:
            print(f"ERROR {desc}: {e}")
            return False

print("=" * 60)
print("Force Moving Framework Components")
print("=" * 60)

# Main directories
for comp in ['agent_tools', 'interface', 'knowledge', 'toolkit']:
    force_move(f'__NEW__/AgentQMS/{comp}', f'AgentQMS/{comp}', comp)

# CHANGELOG
force_move('__NEW__/AgentQMS/CHANGELOG.md', 'AgentQMS/CHANGELOG.md', 'CHANGELOG.md')

# Conventions items
for item in ['audit_framework', 'templates', 'q-manifest.yaml']:
    force_move(f'__NEW__/AgentQMS/conventions/{item}',
               f'AgentQMS/conventions/{item}',
               f'conventions/{item}')

# Dot directories
for dot_dir in ['.copilot', '.qwen', '.cursor']:
    src = os.path.join(base, '__NEW__', dot_dir)
    dst = os.path.join(base, dot_dir)

    if not os.path.exists(src):
        continue

    if os.path.exists(dst):
        # Merge
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)
            if not os.path.exists(dst_item):
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dst_item)
                else:
                    shutil.copytree(src_item, dst_item)
                print(f"MERGED {dot_dir}/{item}")
        shutil.rmtree(src)
        print(f"MERGED {dot_dir}")
    else:
        force_move(f'__NEW__/{dot_dir}', dot_dir, dot_dir)

print("=" * 60)
print("Done!")
sys.exit(0)
