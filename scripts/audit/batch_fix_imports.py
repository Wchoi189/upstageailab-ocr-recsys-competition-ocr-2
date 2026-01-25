#!/usr/bin/env python3
"""Batch fix broken imports based on audit results."""
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Import mapping rules based on audit analysis
IMPORT_FIXES = {
    # ocr.core base components -> interfaces
    'from ocr.core import get_registry': 'from ocr.core.registry import get_registry',
    'from ocr.core import registry': 'from ocr.core import registry',
    'from ocr.core import BaseEncoder': 'from ocr.core.interfaces.models import BaseEncoder',
    'from ocr.core import BaseDecoder': 'from ocr.core.interfaces.models import BaseDecoder',
    'from ocr.core import BaseHead': 'from ocr.core.interfaces.models import BaseHead',
    'from ocr.core import BaseLoss': 'from ocr.core.interfaces.losses import BaseLoss',
    
    # Also fix models.base -> interfaces
    'from ocr.core.models.base import BaseEncoder': 'from ocr.core.interfaces.models import BaseEncoder',
    'from ocr.core.models.base import BaseDecoder': 'from ocr.core.interfaces.models import BaseDecoder',
    'from ocr.core.models.base import BaseHead': 'from ocr.core.interfaces.models import BaseHead',
    'from ocr.core.models.base import BaseLoss': 'from ocr.core.interfaces.losses import BaseLoss',
    
    # ocr.agents -> ocr.core.infrastructure.agents
    'from ocr.agents.base_agent': 'from ocr.core.infrastructure.agents.base_agent',
    'from ocr.agents.llm.base_client': 'from ocr.core.infrastructure.agents.llm.base_client',
    'import ocr.agents.': 'import ocr.core.infrastructure.agents.',
    
    # ocr.detection -> ocr.domains.detection
    'from ocr.detection.': 'from ocr.domains.detection.',
    'import ocr.detection.': 'import ocr.domains.detection.',
    
    # ocr.communication -> ocr.core.infrastructure.communication
    'from ocr.communication.': 'from ocr.core.infrastructure.communication.',
    'import ocr.communication.': 'import ocr.core.infrastructure.communication.',
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
    
    # Target files from audit
    files_to_fix = [
        'ocr/core/infrastructure/agents/coordinator_agent.py',
        'ocr/core/infrastructure/agents/linting_agent.py',
        'ocr/core/infrastructure/agents/llm/grok_client.py',
        'ocr/core/infrastructure/agents/llm/openai_client.py',
        'ocr/core/models/architecture.py',
        'ocr/core/models/encoder/timm_backbone.py',
        'ocr/core/utils/config.py',
        'ocr/core/utils/config_utils.py',
        'ocr/core/utils/wandb_base.py',
        'ocr/domains/detection/models/decoders/craft_decoder.py',
        'ocr/domains/detection/models/decoders/dbpp_decoder.py',
        'ocr/domains/detection/models/decoders/fpn_decoder.py',
        'ocr/domains/detection/models/decoders/pan_decoder.py',
        'ocr/domains/detection/models/decoders/unet.py',
        'ocr/domains/detection/models/encoders/craft_vgg.py',
        'ocr/domains/detection/models/heads/craft_head.py',
        'ocr/domains/detection/models/loss/craft_loss.py',
        'ocr/domains/detection/models/loss/db_loss.py',
        'ocr/domains/recognition/models/loss/cross_entropy_loss.py',
        'scripts/troubleshooting/test_model_forward_backward.py',
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
