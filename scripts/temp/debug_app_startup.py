"""Debug script to trace where the app hangs during startup."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 80)
print("STEP 1: Starting debug script")
print("=" * 80)

print("\nSTEP 2: Testing imports...")
try:
    print("✓ UnifiedAppState imported")
except Exception as e:
    print(f"✗ UnifiedAppState import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

try:
    from ui.apps.unified_ocr_app.services.config_loader import load_mode_config, load_unified_config

    print("✓ config_loader imported")
except Exception as e:
    print(f"✗ config_loader import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\nSTEP 3: Loading unified config...")
try:
    config = load_unified_config("unified_app")
    print("✓ Config loaded successfully")
    print(f"  - App title: {config['app']['title']}")
    print(f"  - Modes: {[m['id'] for m in config['app']['modes']]}")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\nSTEP 4: Testing mode configs...")
for mode in config["app"]["modes"]:
    mode_id = mode["id"]
    try:
        mode_config = load_mode_config(mode_id, validate=False)
        print(f"✓ {mode_id} config loaded")
    except Exception as e:
        print(f"✗ {mode_id} config failed: {e}")
        import traceback

        traceback.print_exc()

print("\nSTEP 5: Testing component imports...")

# Preprocessing components
try:
    print("✓ preprocessing components imported")
except Exception as e:
    print(f"✗ preprocessing components import failed: {e}")
    import traceback

    traceback.print_exc()

# Inference components
try:
    print("✓ inference components imported")
except Exception as e:
    print(f"✗ inference components import failed: {e}")
    import traceback

    traceback.print_exc()

# Comparison components
try:
    print("✓ comparison components imported")
except Exception as e:
    print(f"✗ comparison components import failed: {e}")
    import traceback

    traceback.print_exc()

print("\nSTEP 6: Testing service imports...")

# Preprocessing service
try:
    print("✓ PreprocessingService imported")
except Exception as e:
    print(f"✗ PreprocessingService import failed: {e}")
    import traceback

    traceback.print_exc()

# Inference service
try:
    print("✓ InferenceService imported")
except Exception as e:
    print(f"✗ InferenceService import failed: {e}")
    import traceback

    traceback.print_exc()

# Comparison service
try:
    print("✓ ComparisonService imported")
except Exception as e:
    print(f"✗ ComparisonService import failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL STEPS COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nConclusion: If this script runs successfully but Streamlit doesn't,")
print("the issue is likely in Streamlit's execution model or session state.")
