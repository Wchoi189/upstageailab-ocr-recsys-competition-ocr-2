#!/usr/bin/env python3
"""Simple script to move framework components."""

import os
import shutil

os.chdir('/workspaces/upstageailab-ocr-recsys-competition-ocr-2')

# Move main directories
dirs_to_move = ['agent_tools', 'interface', 'knowledge', 'toolkit']
for d in dirs_to_move:
    src = f'__NEW__/AgentQMS/{d}'
    dst = f'AgentQMS/{d}'
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f'Moved {d}')
    else:
        print(f'Skipped {d} (exists={os.path.exists(src)}, dst_exists={os.path.exists(dst)})')

# Move CHANGELOG
if os.path.exists('__NEW__/AgentQMS/CHANGELOG.md') and not os.path.exists('AgentQMS/CHANGELOG.md'):
    shutil.move('__NEW__/AgentQMS/CHANGELOG.md', 'AgentQMS/CHANGELOG.md')
    print('Moved CHANGELOG.md')

# Move conventions items
conv_items = ['audit_framework', 'templates', 'q-manifest.yaml']
for item in conv_items:
    src = f'__NEW__/AgentQMS/conventions/{item}'
    dst = f'AgentQMS/conventions/{item}'
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f'Moved conventions/{item}')

# Move dot directories
for dot_dir in ['.copilot', '.qwen', '.cursor']:
    src = f'__NEW__/{dot_dir}'
    dst = dot_dir
    if os.path.exists(src):
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
                    print(f'  Merged {item} into {dot_dir}')
            shutil.rmtree(src)
            print(f'Merged and removed {dot_dir}')
        else:
            shutil.move(src, dst)
            print(f'Moved {dot_dir}')

print('Done!')
