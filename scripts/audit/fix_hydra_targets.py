#!/usr/bin/env python3
"""Update Hydra configs to use full module paths for _target_ fields.

This bypasses __init__.py lazy loading and eliminates circular dependencies.
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Mapping of short module paths to full qualified paths
TARGET_MAPPINGS = {
    # Detection models
    'ocr.domains.detection.models.loss.db_loss.DBLoss': 
        'ocr.domains.detection.models.loss.db_loss.DBLoss',
    'ocr.domains.detection.models.encoders.craft_vgg.CraftVGGEncoder':
        'ocr.domains.detection.models.encoders.craft_vgg.CraftVGGEncoder',
    'ocr.domains.detection.models.decoders.craft_decoder.CraftDecoder':
        'ocr.domains.detection.models.decoders.craft_decoder.CraftDecoder',
    'ocr.domains.detection.models.heads.craft_head.CraftHead':
        'ocr.domains.detection.models.heads.craft_head.CraftHead',
    'ocr.domains.detection.models.decoders.dbpp_decoder.DBPPDecoder':
        'ocr.domains.detection.models.decoders.dbpp_decoder.DBPPDecoder',
    'ocr.domains.detection.models.decoders.fpn_decoder.FPNDecoder':
        'ocr.domains.detection.models.decoders.fpn_decoder.FPNDecoder',
    
    # Core models
    'ocr.core.models.architecture.OCRModel':
        'ocr.core.models.architecture.OCRModel',
    'ocr.core.models.encoder.TimmBackbone':
        'ocr.core.models.encoder.timm_backbone.TimmBackbone',
    'ocr.core.models.encoder.timm_backbone.TimmBackbone':
        'ocr.core.models.encoder.timm_backbone.TimmBackbone',
    
    # Recognition models
    'ocr.domains.recognition.models.PARSeq':
        'ocr.domains.recognition.models.architecture.PARSeq',
}

# Patterns that need to be fixed (module path adjustments)
PATH_FIXES = [
    # encoder module path fix
    ('ocr.core.models.encoder.TimmBackbone', 'ocr.core.models.encoder.timm_backbone.TimmBackbone'),
]

def fix_yaml_file(file_path: Path, dry_run: bool = True) -> Tuple[int, List[str]]:
    """Fix _target_ paths in a YAML file."""
    try:
        content = file_path.read_text()
        original = content
        changes = []
        
        # Apply direct mappings
        for old_path, new_path in TARGET_MAPPINGS.items():
            if f'_target_: {old_path}' in content:
                content = content.replace(f'_target_: {old_path}', f'_target_: {new_path}')
                changes.append(f"_target_: {old_path} → {new_path}")
        
        # Apply pattern fixes
        for old_pattern, new_pattern in PATH_FIXES:
            if f'_target_: {old_pattern}' in content:
                content = content.replace(f'_target_: {old_pattern}', f'_target_: {new_pattern}')
                changes.append(f"_target_: {old_pattern} → {new_pattern}")
        
        if content != original:
            if not dry_run:
                file_path.write_text(content)
            return len(changes), changes
        
        return 0, []
    except Exception as e:
        print(f"ERROR in {file_path}: {e}")
        return 0, []

def scan_for_broken_targets(base: Path) -> List[Tuple[Path, str]]:
    """Scan YAML files for potentially broken _target_ paths."""
    broken = []
    configs_dir = base / 'configs'
    
    for yaml_file in configs_dir.rglob('*.yaml'):
        try:
            content = yaml_file.read_text()
            # Find all _target_ lines
            for line_num, line in enumerate(content.split('\n'), 1):
                if '_target_:' in line:
                    # Check for patterns that might be broken
                    if 'ocr.core.models.encoder.TimmBackbone' in line:
                        broken.append((yaml_file, f"L{line_num}: {line.strip()}"))
                    elif 'ocr.domains.detection.models.encoders.craft_vgg.CraftVGGEncoder' in line:
                        broken.append((yaml_file, f"L{line_num}: {line.strip()}"))
                    elif 'ocr.domains.recognition.models.PARSeq' in line:
                        broken.append((yaml_file, f"L{line_num}: {line.strip()}"))
        except Exception as e:
            continue
    
    return broken

def main():
    dry_run = '--dry-run' in sys.argv
    scan = '--scan' in sys.argv
    
    base = Path('/workspaces/upstageailab-ocr-recsys-competition-ocr-2')
    
    if scan:
        print("Scanning for potentially broken _target_ paths...")
        print("=" * 80)
        broken = scan_for_broken_targets(base)
        if broken:
            for file_path, issue in broken:
                print(f"{file_path.relative_to(base)}: {issue}")
        else:
            print("No obvious issues found")
        return
    
    # Config files that need fixing based on audit
    yaml_files = [
        'configs/experiment/det_resnet50_v1.yaml',
        'configs/model/architectures/craft.yaml',
        'configs/model/architectures/dbnetpp.yaml',
        'configs/model/architectures/dbnet_atomic.yaml',
        'configs/model/architectures/parseq.yaml',
        'configs/model/loss/db_loss.yaml',
    ]
    
    total_changes = 0
    total_files = 0
    
    print(f"{'DRY RUN - ' if dry_run else ''}Fixing Hydra _target_ paths...")
    print("=" * 80)
    
    for yaml_rel in yaml_files:
        yaml_path = base / yaml_rel
        if not yaml_path.exists():
            print(f"SKIP (not found): {yaml_rel}")
            continue
        
        num_changes, changes = fix_yaml_file(yaml_path, dry_run)
        if num_changes > 0:
            total_changes += num_changes
            total_files += 1
            print(f"\n{yaml_rel}: {num_changes} changes")
            for change in changes:
                print(f"  - {change}")
        else:
            print(f"\n{yaml_rel}: No changes needed")
    
    print("\n" + "=" * 80)
    print(f"Total: {total_changes} changes in {total_files} files")
    if dry_run:
        print("\nRun without --dry-run to apply changes")
    else:
        print("\nChanges applied successfully")

if __name__ == '__main__':
    main()
