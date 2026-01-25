from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# Import from project src (mounted via PYTHONPATH)
from api_clients.upstage import submit_document_parse_job


def preprocess(**context):
    # Scan the data directory for images
    data_dir = "/opt/airflow/data"
    supported_extensions = (".jpg", ".jpeg", ".png", ".pdf")
    
    # Get list of files
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return []
        
    files = [
        f for f in os.listdir(data_dir) 
        if f.lower().endswith(supported_extensions)
    ]
    
    # Create input items
    items = []
    for f in files:
        items.append({
            "id": f,
            "path": os.path.join(data_dir, f)
        })
    
    print(f"Preprocess: Found {len(items)} items to process in {data_dir}")
    return items


def call_upstage_api(**context):
    # Example: read env for API key; in real use, read from Airflow Connection/Secret
    api_key = os.getenv("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY2")
    if not api_key:
        raise RuntimeError("Missing UPSTAGE_API_KEY/UPSTAGE_API_KEY2 in environment or secrets.")

    # Demo payload; replace with batch mapping logic
    items = context.get("ti").xcom_pull(task_ids="preprocess") or [
        {"id": "sample-1", "path": "/data/sample.jpg"}
    ]
    results = []
    for item in items:
        res = submit_document_parse_job(item["path"], api_key=api_key)
        results.append({"id": item["id"], "result": res})
    return results


def validate(**context):
    results = context.get("ti").xcom_pull(task_ids="api_call") or []
    print(f"Validate: {len(results)} items")


def export(**context):
    results = context.get("ti").xcom_pull(task_ids="api_call") or []
    if not results:
        print("Export: No results to export.")
        return

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/opt/airflow/data/output_{timestamp}.json"
    
    import json
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Export: Wrote {len(results)} results to {output_file}")


def cleanup(**context):
    print("Cleanup: remove temp files, close resources")


def create_dag() -> DAG:
    default_args = {
        "owner": "airflow",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="batch_processor_dag",
        start_date=datetime(2024, 1, 1),
        schedule=None,
        default_args=default_args,
        catchup=False,
        description="Batch processing DAG for Upstage APIs",
        tags=["ocr", "kie", "api"],
    ) as dag:
        t_pre = PythonOperator(task_id="preprocess", python_callable=preprocess)
        t_api = PythonOperator(task_id="api_call", python_callable=call_upstage_api)
        t_val = PythonOperator(task_id="validate", python_callable=validate)
        t_exp = PythonOperator(task_id="export", python_callable=export)
        t_clean = PythonOperator(task_id="cleanup", python_callable=cleanup)

        t_pre >> t_api >> t_val >> t_exp >> t_clean
        return dag


globals()["batch_processor_dag"] = create_dag()
