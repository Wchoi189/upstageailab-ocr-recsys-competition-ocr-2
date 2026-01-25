
import sys
import os
from airflow import settings
from airflow.models import DagBag

print("=== Diagnostics ===")
print(f"UID: {os.getuid()}")
print(f"CWD: {os.getcwd()}")
print(f"PYTHONPATH: {sys.path}")
print(f"DAGS Folder (config): {settings.DAGS_FOLDER}")

dags_dir = "/opt/airflow/dags"
print(f"\nListing {dags_dir}:")
try:
    for f in os.listdir(dags_dir):
        path = os.path.join(dags_dir, f)
        perm = oct(os.stat(path).st_mode)[-3:]
        print(f" - {f} ({perm})")
except Exception as e:
    print(f"Error listing dags: {e}")

print("\nTesting Import 'api_clients.upstage':")
try:
    import api_clients.upstage
    print("✅ Import Successful (from api_clients)")
except ImportError:
    print("❌ Import Failed (from api_clients)")
    try:
        import src.api_clients.upstage
        print("✅ Import Successful (from src.api_clients)")
    except ImportError:
        print("❌ Import Failed (from src.api_clients as well)")

print("\nAttempting DagBag parse:")
try:
    dagbag = DagBag(dag_folder=dags_dir, include_examples=False)
    print(f"DagBag sizes: {dagbag.size()}")
    print("Import Errors:")
    if not dagbag.import_errors:
        print(" (None)")
    else:
        for key, val in dagbag.import_errors.items():
            print(f"{key}: {val}")
    print("DAG IDs:")
    if not dagbag.dags:
        print(" (None)")
    else:
        for dag_id in dagbag.dags.keys():
            print(f" - {dag_id}")
except Exception as e:
    print(f"DagBag crash: {e}")
