#!/usr/bin/env python3
"""Batch fix broken imports based on audit analysis.

CONSERVATIVE STRATEGY:
- Only fix imports where the target module is VERIFIED to exist
- Do NOT fix imports to modules that don't exist (comment those out manually)
- Focus on high-impact, low-risk changes
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Import mapping rules - VERIFIED paths only
IMPORT_FIXES = {
    # Registry fixes (4 occurrences, VERIFIED: ocr/core/utils/registry.py exists)
    'from ocr.core import registry': 'from ocr.core.utils.registry import registry',
    'from ocr.core import get_registry': 'from ocr.core.utils.registry import get_registry',
    
    # Text rendering (2 occurrences, VERIFIED: ocr/domains/recognition/utils/visualization.py)
    'from ocr.core.utils.text_rendering import': 'from ocr.domains.recognition.utils.visualization import',
    
    # Perspective correction (3 occurrences, VERIFIED: ocr/domains/detection/utils/perspective_correction.py)
    'from ocr.core.utils.perspective_correction import': 'from ocr.domains.detection.utils.perspective_correction import',
    
    # Communication infrastructure (VERIFIED: ocr/core/infrastructure/communication/)
    'from ocr.communication.rabbitmq_transport import': 'from ocr.core.infrastructure.communication.rabbitmq_transport import',
    'from ocr.communication.slack_service import': 'from ocr.core.infrastructure.communication.slack_service import',
}

# Module path fixes (for actual file moves)
MODULE_RELOCATIONS = {
    'ocr.core.models.encoder.timm_backbone': 'ocr.core.models.encoder.timm_backbone',
    'ocr.core.models.encoder.TimmBackbone': 'ocr.core.models.encoder.timm_backbone.TimmBackbone',
}

def fix_file(file_path: Path, dry_run: bool = True) -> Tuple[int, List[str]]:
    """Fix imports in a single file."""
    try:
        content = file_path.read_text()
        original = content
        changes = []
        
        for old_import, new_import in IMPORT_FIXES.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes.append(f"{old_import} → {new_import}")
        
        if content != original:
            if not dry_run:
                file_path.write_text(content)
            return len(changes), changes
        
        return 0, []
    except Exception as e:
        print(f"ERROR in {file_path}: {e}")
        return 0, []

def main():
    import sys
    dry_run = '--dry-run' in sys.argv
    
    base = Path('/workspaces/upstageailab-ocr-recsys-competition-ocr-2')
    
    # Target files - Core infrastructure only (high-impact, verified fixes)
    files_to_fix = [
        # Utils with registry imports (4x registry)
        'ocr/core/utils/config.py',
        'ocr/core/utils/config_utils.py',
        'ocr/core/utils/wandb_base.py',
        
        # Callbacks with text_rendering imports (2x)
        'ocr/domains/detection/callbacks/wandb.py',
        'ocr/domains/recognition/callbacks/wandb.py',
        
        # Detection utils with perspective_correction (3x)
        'ocr/domains/detection/inference/preprocess.py',
        'scripts/demos/offline_perspective_preprocess_train.py',
        'scripts/demos/test_perspective_on_pseudo_label.py',
        
        # Communication imports (2x)
        'scripts/prototypes/multi_agent/test_linting_loop.py',
        'scripts/prototypes/multi_agent/test_slack.py',
    ]
    
    total_changes = 0
    total_files = 0
    
    print(f"{'DRY RUN - ' if dry_run else ''}Fixing imports...")
    print("=" * 80)
    
    for file_rel in files_to_fix:
        file_path = base / file_rel
        if not file_path.exists():
            print(f"SKIP (not found): {file_rel}")
            continue
        
        num_changes, changes = fix_file(file_path, dry_run)
        if num_changes > 0:
            total_changes += num_changes
            total_files += 1
            print(f"\n{file_rel}: {num_changes} changes")
            for change in changes:
                print(f"  - {change}")
    
    print("\n" + "=" * 80)
    print(f"Total: {total_changes} changes in {total_files} files")
    if dry_run:
        print("\nRun without --dry-run to apply changes")

if __name__ == '__main__':
    main()
