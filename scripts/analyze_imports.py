#!/usr/bin/env python3
"""Analyze import dependencies in training script."""

import ast
from collections import defaultdict
from pathlib import Path


def extract_imports(file_path: Path):
    """Extract all imports from a Python file."""
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read())
    except Exception as e:
        return None, str(e)

    imports = {
        'direct': [],
        'from': []
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports['direct'].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports['from'].append(f"{module}.{alias.name}" if module else alias.name)

    return imports, None


def analyze_training_imports():
    """Analyze imports in training-related files."""
    project_root = Path(__file__).parent.parent

    files_to_check = [
        project_root / "runners" / "train.py",
    ]

    print("IMPORT ANALYSIS FOR TRAINING SCRIPTS")
    print("=" * 90)

    all_imports = defaultdict(list)

    for file_path in files_to_check:
        if not file_path.exists():
            print(f"\n⚠️  File not found: {file_path}")
            continue

        print(f"\n📄 {file_path.relative_to(project_root)}")
        print("-" * 90)

        imports, error = extract_imports(file_path)
        if error:
            print(f"❌ Error parsing file: {error}")
            continue

        # Categorize imports
        heavy_imports = []
        project_imports = []
        stdlib_imports = []

        all_imps = imports['direct'] + imports['from']
        for imp in all_imps:
            if any(heavy in imp for heavy in ['torch', 'transformers', 'lightning', 'wandb', 'albumentations', 'cv2', 'PIL']):
                heavy_imports.append(imp)
                all_imports['heavy'].append((file_path.name, imp))
            elif imp.startswith('ocr.') or imp.startswith('ui.'):
                project_imports.append(imp)
            else:
                stdlib_imports.append(imp)

        print(f"\n🔴 Heavy imports ({len(heavy_imports)}):")
        if heavy_imports:
            for imp in sorted(set(heavy_imports)):
                print(f"  - {imp}")
        else:
            print("  (none)")

        print(f"\n🔵 Project imports ({len(project_imports)}):")
        for imp in sorted(set(project_imports))[:15]:
            print(f"  - {imp}")
        if len(project_imports) > 15:
            print(f"  ... and {len(project_imports) - 15} more")

        print(f"\n⚪ Standard library imports ({len(stdlib_imports)}):")
        for imp in sorted(set(stdlib_imports))[:10]:
            print(f"  - {imp}")
        if len(stdlib_imports) > 10:
            print(f"  ... and {len(stdlib_imports) - 10} more")

    print("\n\n" + "=" * 90)
    print("SUMMARY: Heavy imports across all training files")
    print("=" * 90)

    if all_imports['heavy']:
        for file, imp in sorted(set(all_imports['heavy'])):
            print(f"  {file:30s} → {imp}")
    else:
        print("  ✅ No heavy imports at top level!")

    print("\n\n" + "=" * 90)
    print("RECOMMENDATIONS")
    print("=" * 90)
    if all_imports['heavy']:
        print("❌ Heavy imports found at top level - these slow down startup!")
        print("\nTo fix:")
        print("  1. Move heavy imports inside functions (lazy loading)")
        print("  2. Use importlib for conditional imports")
        print("  3. Consider splitting into lightweight entry point")
    else:
        print("✅ No heavy top-level imports - startup should be fast!")


if __name__ == "__main__":
    analyze_training_imports()
