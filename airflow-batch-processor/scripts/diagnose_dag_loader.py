
import sys
import os
from airflow import settings
from airflow.models import DagBag, DagModel
from airflow.utils.session import create_session
import subprocess

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

print("\n=== Database State (DagModel) ===")
try:
    with create_session() as session:
        dags = session.query(DagModel).all()
        print(f"Total DAGs in DB: {len(dags)}")
        for d in dags:
            print(f" - {d.dag_id} (Active: {d.is_active}, Paused: {d.is_paused}, File: {d.fileloc})")
except Exception as e:
    print(f"DB Error: {e}")

print("\n=== CLI Check (airflow dags list) ===")
try:
    # Run looking for our specific dag
    result = subprocess.run(["airflow", "dags", "list"], capture_output=True, text=True)
    print("Output head (first 5 lines):")
    print("\n".join(result.stdout.splitlines()[:5]))
    if "batch_processor_dag" in result.stdout:
        print("✅ 'batch_processor_dag' found in CLI list")
    else:
        print("❌ 'batch_processor_dag' NOT found in CLI list")
except Exception as e:
    print(f"CLI execution failed: {e}")
