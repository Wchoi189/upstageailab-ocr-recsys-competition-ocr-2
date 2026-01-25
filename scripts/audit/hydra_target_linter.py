#!/usr/bin/env python3
"""
Hydra Target Linter - Prevents shallow target paths that cause lazy loading conflicts.

Rule: All _target_ paths must use Full Qualified Names (FQN) including filename.
Example: ocr.core.models.encoder.timm_backbone.TimmBackbone (not ocr.core.models.encoder.TimmBackbone)
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple

def audit_hydra_paths(base_path: Path) -> List[Tuple[str, str, str]]:
    """
    Scan YAML configs for shallow Hydra targets.
    
    Returns:
        List of (file_path, line_number, target_path) for violations
    """
    violations = []
    configs_dir = base_path / "configs"
    
    # Known valid shallow patterns (exceptions)
    ALLOWED_SHALLOW = {
        "torch.optim",  # PyTorch standard
        "torch.nn",
        "lightning.pytorch",
        "hydra",
    }
    
    for yaml_file in configs_dir.rglob("*.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines, 1):
                    if "_target_:" not in line:
                        continue
                    
                    # Extract the target path
                    match = re.search(r'_target_:\s*([^\s#]+)', line)
                    if not match:
                        continue
                    
                    target = match.group(1).strip()
                    
                    # Skip allowed shallow patterns
                    if any(target.startswith(allowed) for allowed in ALLOWED_SHALLOW):
                        continue
                    
                    # Check if it's an ocr.* path
                    if not target.startswith("ocr."):
                        continue
                    
                    # Count dots - should be at least 4 for ocr.domain.category.module.Class
                    # Examples:
                    #   ocr.core.models.encoder.timm_backbone.TimmBackbone (6 dots) ✅
                    #   ocr.core.models.encoder.TimmBackbone (4 dots) ❌ - missing filename
                    #   ocr.domains.detection.models.loss.db_loss.DBLoss (6 dots) ✅
                    
                    parts = target.split('.')
                    
                    # Heuristic: If the last part is NOT capitalized (CamelCase), it's likely a module not a class
                    if len(parts) >= 2:
                        last_part = parts[-1]
                        second_last = parts[-2] if len(parts) > 1 else ""
                        
                        # If last part is lowercase (module name), it's definitely wrong
                        if last_part[0].islower():
                            violations.append((
                                str(yaml_file.relative_to(base_path)),
                                str(line_num),
                                target,
                                "Module path without class (ends with lowercase)"
                            ))
                            continue
                        
                        # If last part is CamelCase but second-last is generic (encoder, decoder, models)
                        # it might be missing the filename
                        GENERIC_MODULES = {'encoder', 'decoder', 'models', 'loss', 'head', 'heads', 'decoders', 'encoders'}
                        if second_last in GENERIC_MODULES and len(parts) < 6:
                            violations.append((
                                str(yaml_file.relative_to(base_path)),
                                str(line_num),
                                target,
                                f"Likely missing filename (path ends at generic module '{second_last}')"
                            ))
        
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}", file=sys.stderr)
    
    return violations

def main():
    base = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2")
    
    violations = audit_hydra_paths(base)
    
    if not violations:
        print("✅ All Hydra targets use full module paths (FQN)")
        return 0
    
    print("🚨 Hydra Target Violations Found")
    print("=" * 80)
    print(f"Rule: Use Full Qualified Names including filename")
    print(f"Example: ocr.core.models.encoder.timm_backbone.TimmBackbone")
    print("=" * 80)
    print()
    
    for file_path, line_num, target, reason in violations:
        print(f"📁 {file_path}:{line_num}")
        print(f"   Target: {target}")
        print(f"   Issue: {reason}")
        print()
    
    print(f"Total violations: {len(violations)}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
