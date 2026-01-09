import sys
import time


def profile_import(module_name):
    start = time.time()
    try:
        __import__(module_name)
        elapsed = time.time() - start
        print(f"Import {module_name:<30}: {elapsed:.4f}s")
    except ImportError:
        import traceback

        traceback.print_exc()
        print(f"Import {module_name:<30}: FAILED")
    except Exception as e:
        print(f"Import {module_name:<30}: ERROR {e}")


# print("Profiling key imports...")
# profile_import("torch")
# profile_import("numpy")
# profile_import("cv2")
# profile_import("fastapi")

print("\nProfiling application modules...")
import os

backend_path = os.path.join(os.getcwd(), "apps/ocr-inference-console/backend")
sys.path.insert(0, backend_path)
# We also need project root for apps.shared to work
sys.path.insert(0, os.getcwd())

profile_import("main")

if "torch" in sys.modules:
    print("FAILURE: torch IS loaded after importing main!")
else:
    print("SUCCESS: torch is NOT loaded after importing main!")
