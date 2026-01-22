# Auto-Alignment Script for Hydra Configurations

**Source**: Conversation analysis
**Date**: 2026-01-22
**Context**: Automated Hydra target fixing using runtime reflection
**Python Manager**: uv

---

## Overview

This script uses **Runtime Reflection** to automatically fix Hydra configurations. Instead of guessing where files moved or using static text search, it asks the Python interpreter: *"In the current environment, where does this class actually live?"*

---

## Core Implementation

### auto_align_hydra.py

```python
#!/usr/bin/env python3
"""
Auto-align Hydra configuration targets with actual Python module locations.

Uses runtime reflection to find the true location of classes and updates
YAML configurations accordingly.
"""

import subprocess
import importlib
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, List

# Ensure the local 'ocr' package is in the path
sys.path.insert(0, os.getcwd())


def get_true_path_via_adt(class_name: str) -> Optional[str]:
    """
    Uses ADT intelligent-search to find the current module of a class.

    Args:
        class_name: Name of the class to search for

    Returns:
        Qualified path (e.g., ocr.core.models.X) or None if not found
    """
    cmd = f"adt intelligent-search '{class_name}' --output json"
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode()
        data = json.loads(result)
        return data.get("qualified_path")
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return None


def get_true_path_via_import(class_name: str, search_modules: List[str]) -> Optional[str]:
    """
    Uses Python import to find the actual location of a class.

    Args:
        class_name: Name of the class to find
        search_modules: List of module prefixes to search in

    Returns:
        Qualified path or None if not found
    """
    for module_prefix in search_modules:
        try:
            # Try common locations
            for subpath in ['', '.models', '.metrics', '.utils', '.data']:
                module_name = f"{module_prefix}{subpath}"
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, class_name):
                        cls = getattr(module, class_name)
                        return f"{cls.__module__}.{cls.__name__}"
                except (ImportError, AttributeError):
                    continue
        except Exception:
            continue
    return None


def get_true_path(class_name: str) -> Optional[str]:
    """
    Find the true path of a class using multiple strategies.

    Args:
        class_name: Name of the class to find

    Returns:
        Qualified path or None if not found
    """
    # Strategy 1: Use ADT intelligent-search
    path = get_true_path_via_adt(class_name)
    if path:
        return path

    # Strategy 2: Try direct import from common locations
    search_modules = [
        'ocr.core',
        'ocr.domains.detection',
        'ocr.domains.recognition',
        'ocr.data',
        'ocr.pipelines'
    ]

    path = get_true_path_via_import(class_name, search_modules)
    if path:
        return path

    return None


def parse_audit_log(audit_log_path: str) -> List[Dict[str, str]]:
    """
    Parse audit log to extract broken targets.

    Args:
        audit_log_path: Path to audit results file

    Returns:
        List of dicts with config_file, old_target, class_name
    """
    broken_targets = []
    current_config = None

    with open(audit_log_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Extract config file
            if "[Config]" in line or "File:" in line:
                parts = line.split("]", 1) if "]" in line else line.split(":", 1)
                if len(parts) > 1:
                    current_config = parts[1].strip()

            # Extract broken target
            if "-->" in line and "Target:" in line:
                parts = line.split("Target:", 1)
                if len(parts) > 1:
                    old_target = parts[1].strip()
                    class_name = old_target.split(".")[-1]

                    broken_targets.append({
                        'config_file': current_config,
                        'old_target': old_target,
                        'class_name': class_name
                    })

    return broken_targets


def update_yaml_target(config_file: str, old_target: str, new_target: str) -> bool:
    """
    Update a YAML file to replace old target with new target.

    Args:
        config_file: Path to YAML config file
        old_target: Old target path
        new_target: New target path

    Returns:
        True if successful, False otherwise
    """
    # Backup original file
    backup_path = f"{config_file}.bak"
    try:
        subprocess.run(['cp', config_file, backup_path], check=True)
    except subprocess.CalledProcessError:
        print(f"  ⚠️  Could not create backup of {config_file}")
        return False

    # Update using yq
    yq_cmd = f'yq -i \'(.. | select(. == "{old_target}")) = "{new_target}"\' {config_file}'

    try:
        subprocess.run(yq_cmd, shell=True, check=True, stderr=subprocess.DEVNULL)

        # Verify YAML is still valid
        verify_cmd = f'yq "." {config_file} > /dev/null'
        subprocess.run(verify_cmd, shell=True, check=True, stderr=subprocess.DEVNULL)

        # Remove backup if successful
        os.remove(backup_path)
        return True

    except subprocess.CalledProcessError:
        # Restore backup on failure
        subprocess.run(['mv', backup_path, config_file])
        return False


def heal_configs(audit_log_path: str, dry_run: bool = False) -> Dict[str, int]:
    """
    Parse audit log and apply yq fixes.

    Args:
        audit_log_path: Path to audit results file
        dry_run: If True, only show what would be done

    Returns:
        Dict with counts of successes, failures, not_found
    """
    broken_targets = parse_audit_log(audit_log_path)

    stats = {
        'total': len(broken_targets),
        'success': 0,
        'failure': 0,
        'not_found': 0
    }

    print(f"🔍 Found {stats['total']} broken targets to heal\n")

    for item in broken_targets:
        config_file = item['config_file']
        old_target = item['old_target']
        class_name = item['class_name']

        if not config_file or not os.path.exists(config_file):
            print(f"⚠️  Skipping: Config file not found: {config_file}")
            stats['failure'] += 1
            continue

        print(f"🛠️  Attempting to heal: {class_name}")
        print(f"   Config: {config_file}")
        print(f"   Old target: {old_target}")

        # Find new location
        new_path = get_true_path(class_name)

        if new_path:
            print(f"   ✅ Found new home: {new_path}")

            if not dry_run:
                if update_yaml_target(config_file, old_target, new_path):
                    print(f"   ✅ Updated successfully")
                    stats['success'] += 1
                else:
                    print(f"   ❌ Update failed")
                    stats['failure'] += 1
            else:
                print(f"   [DRY RUN] Would update to: {new_path}")
                stats['success'] += 1
        else:
            print(f"   ❌ Could not locate {class_name}. Manual intervention required.")
            stats['not_found'] += 1

        print()  # Blank line between items

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Auto-align Hydra configurations with actual Python module locations'
    )
    parser.add_argument(
        'audit_log',
        nargs='?',
        default='audit_results.txt',
        help='Path to audit results file (default: audit_results.txt)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    if not os.path.exists(args.audit_log):
        print(f"❌ Audit log not found: {args.audit_log}")
        print("   Run: uv run python scripts/audit/master_audit.py > audit_results.txt")
        sys.exit(1)

    print("🛡️  Starting Hydra Auto-Alignment")
    print("=" * 60)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print("=" * 60)

    stats = heal_configs(args.audit_log, dry_run=args.dry_run)

    print("=" * 60)
    print("📊 Summary:")
    print(f"   Total targets: {stats['total']}")
    print(f"   ✅ Successfully healed: {stats['success']}")
    print(f"   ❌ Failed to update: {stats['failure']}")
    print(f"   🔍 Not found: {stats['not_found']}")

    if stats['not_found'] > 0:
        print("\n⚠️  Some classes could not be located.")
        print("   Review the output above and fix manually.")
        sys.exit(1)
    elif stats['failure'] > 0:
        print("\n❌ Some updates failed.")
        print("   Check the output above for details.")
        sys.exit(1)
    else:
        print("\n✅ All targets healed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

---

## Usage

### Basic Usage

```bash
# Generate audit results
uv run python scripts/audit/master_audit.py > audit_results.txt

# Run auto-alignment
uv run python scripts/audit/auto_align_hydra.py

# Dry run to see what would change
uv run python scripts/audit/auto_align_hydra.py --dry-run
```

### With Custom Audit File

```bash
uv run python scripts/audit/auto_align_hydra.py custom_audit.txt
```

---

## Integration with Workflow

### Complete Healing Workflow

```bash
#!/bin/bash
# scripts/heal_hydra_configs.sh

set -e

echo "🔍 Step 1: Running audit..."
uv run python scripts/audit/master_audit.py > audit_results.txt

echo ""
echo "🔍 Step 2: Checking what would be fixed..."
uv run python scripts/audit/auto_align_hydra.py --dry-run

echo ""
read -p "Proceed with fixes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🛠️  Step 3: Applying fixes..."
    uv run python scripts/audit/auto_align_hydra.py

    echo ""
    echo "✅ Step 4: Verifying fixes..."
    uv run python scripts/audit/master_audit.py > audit_results_after.txt

    echo ""
    echo "📊 Comparison:"
    echo "Before: $(grep -c "Target:" audit_results.txt || echo 0) broken targets"
    echo "After:  $(grep -c "Target:" audit_results_after.txt || echo 0) broken targets"
else
    echo "Aborted."
fi
```

---

## AI Agent Instruction Pattern

### Verification Loop

```markdown
**The refactor has caused module drift. Use the following Healing Protocol:**

1. **Run** `uv run python scripts/audit/auto_align_hydra.py --dry-run`
2. **Review** the output for any `❌ Could not locate` errors
3. **For failures:** Manually use `adt intelligent-search` to find the missing symbol
4. **Execute** `uv run python scripts/audit/auto_align_hydra.py` to apply fixes
5. **Validate:** Run `uv run python scripts/audit/master_audit.py` again.
   The 'Broken Hydra Targets' count must be 0 before proceeding to Phase 2.
```

---

## Advanced Features

### Custom Search Paths

Modify the `search_modules` list in `get_true_path()` to add custom search locations:

```python
search_modules = [
    'ocr.core',
    'ocr.domains.detection',
    'ocr.domains.recognition',
    'ocr.data',
    'ocr.pipelines',
    'your.custom.module'  # Add custom paths here
]
```

### Integration with ADT

The script prioritizes ADT's `intelligent-search` for finding symbols. Ensure ADT is properly configured:

```bash
# Test ADT search
adt intelligent-search "CLEvalMetric" --output json
```

### Batch Processing

For large projects, process in batches:

```python
# Modify heal_configs() to process in batches
def heal_configs_batched(audit_log_path: str, batch_size: int = 10):
    broken_targets = parse_audit_log(audit_log_path)

    for i in range(0, len(broken_targets), batch_size):
        batch = broken_targets[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}...")

        # Process batch...

        # Refresh editable install after each batch
        subprocess.run(['uv', 'pip', 'install', '-e', '.'],
                      stdout=subprocess.DEVNULL)
```

---

## Troubleshooting

### Class Not Found

If `get_true_path()` returns `None`:

1. **Check if class exists**:
   ```bash
   grep -r "class ClassName" ocr/
   ```

2. **Try manual ADT search**:
   ```bash
   adt intelligent-search "ClassName"
   ```

3. **Check if module is importable**:
   ```bash
   uv run python -c "from ocr.module import ClassName"
   ```

### YAML Update Failed

If `update_yaml_target()` fails:

1. **Check YAML syntax**:
   ```bash
   yq '.' config.yaml
   ```

2. **Verify yq is installed**:
   ```bash
   which yq
   yq --version
   ```

3. **Check file permissions**:
   ```bash
   ls -l config.yaml
   ```

---

## See Also

- `migration_guard_implementation.md` - Pre-execution validation
- `yq_mastery_guide.md` - YAML manipulation techniques
- `adt_usage_patterns.md` - ADT usage and integration
- `instruction_patterns.md` - AI agent instruction strategies
