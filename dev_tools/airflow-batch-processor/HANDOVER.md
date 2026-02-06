# Session Handover: Debugging batch_processor_dag

## Current Status
- **Permission Fixed**: `chmod 777` on logs via Bridge resolved the `PermissionError`.
- **Config Fixed**: Updated `docker-compose.yml` to set `AIRFLOW__CORE__INTERNAL_API_URL`, fixing the `Connection Refused` error.
- **Bridge Active**: Windows Agent Bridge is working and `pika` is installed.

## Immediate Action Required (User)
The configuration change requires a restart of the Airflow stack.

1.  **Restart Containers**:
    Run this in your terminal (not via bridge):
    ```bash
    cd airflow-batch-processor/docker
    docker compose up -d --force-recreate
    ```

2.  **Verify**:
    Once restarted, trigger the DAG again via the API or UI.
    ```bash
    # Get Token
    export AIRFLOW_TOKEN=$(./airflow-batch-processor/scripts/get_airflow_token.sh | grep -A 1 "Token:" | tail -n 1)
    
    # Trigger
    curl -X POST "http://172.17.0.1:8080/api/v2/dags/batch_processor_dag/dagRuns" \
      -H "Authorization: Bearer $AIRFLOW_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"logical_date": "2024-01-26T10:00:00Z"}'
    ```

3.  **Check Output**:
    Wait for completion and check:
    ```bash
    ls -l airflow-batch-processor/data/output_*.json
    ```

## Troubleshooting
If it fails again, use the Bridge to fetch logs:
```bash
python airflow-batch-processor/windows_agent_bridge/bridge_client.py "docker logs --tail 100 airflow-scheduler"
```
