import requests
import os
import sys

# Configuration
AIRFLOW_URL = "http://172.17.0.1:8080"
DAG_ID = "batch_processor_dag"
DAG_RUN_ID = "manual__2024-01-26T20:00:00+00:00"  # Updated ID

def get_token():
    try:
        # Use existing script logic or just read env if we exported it
        # For simplicity, let's assume valid token is passed via env or just use the helper
        token = os.popen("./airflow-batch-processor/scripts/get_airflow_token.sh | grep -A 1 'Token:' | tail -n 1").read().strip()
        return token
    except Exception as e:
        print(f"Error getting token: {e}")
        return None

def check_status():
    token = get_token()
    if not token:
        print("Could not get token.")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Get Task Instances
    url = f"{AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns/{DAG_RUN_ID}/taskInstances"
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        tis = data.get("task_instances", [])
        print(f"{'Task ID':<20} | {'State':<15} | {'Try Number':<10}")
        print("-" * 50)
        for ti in tis:
            print(f"{ti['task_id']:<20} | {str(ti['state']):<15} | {ti['try_number']:<10}")
            
    except Exception as e:
        print(f"Error calling API: {e}")
        print(resp.text if 'resp' in locals() else "")

if __name__ == "__main__":
    check_status()
