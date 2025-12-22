#!/usr/bin/env python3
"""Test script to diagnose API server startup issues."""

import sys
import time
import traceback


def test_imports():
    """Test imports step by step."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    steps = [
        ("Basic imports", lambda: __import__("yaml")),
        ("FastAPI", lambda: __import__("fastapi")),
        ("Path utils", lambda: __import__("ocr.utils.path_utils")),
        ("ConfigParser", lambda: __import__("ui.utils.config_parser")),
        ("Command builder services", lambda: __import__("ui.apps.command_builder.services.overrides")),
        ("Command utils", lambda: __import__("ui.utils.command")),
        ("UI generator", lambda: __import__("ui.utils.ui_generator")),
        ("Playground API paths", lambda: __import__("services.playground_api.utils.paths")),
        ("Command builder router", lambda: __import__("services.playground_api.routers.command_builder")),
        ("Full app", lambda: __import__("services.playground_api.app")),
    ]

    for name, import_func in steps:
        start = time.time()
        try:
            import_func()
            elapsed = time.time() - start
            status = "✅" if elapsed < 1 else "⚠️"
            print(f"{status} {name}: {elapsed:.3f}s")
            if elapsed > 3:
                print(f"   WARNING: Import took {elapsed:.3f}s (very slow!)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ {name}: FAILED after {elapsed:.3f}s")
            print(f"   Error: {e}")
            traceback.print_exc()
            return False
        sys.stdout.flush()

    return True


def test_app_creation():
    """Test creating the FastAPI app."""
    print("\n" + "=" * 60)
    print("Testing app creation...")
    print("=" * 60)

    try:
        from apps.backend.services.playground_api.app import app

        print("✅ App created successfully")
        print(f"   Routes: {len(app.routes)}")
        return True
    except Exception as e:
        print(f"❌ Failed to create app: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("API Server Startup Diagnostic")
    print("=" * 60)

    if not test_imports():
        print("\n❌ Import test failed. Cannot proceed.")
        sys.exit(1)

    if not test_app_creation():
        print("\n❌ App creation failed.")
        sys.exit(1)

    print("\n✅ All tests passed! The app should be able to start.")
    print("\nTo start the server, run:")
    print("  uv run python run_spa.py --api-only --no-reload")
