---
name: airflow_bridge_ops
description: Operational guide for Hybrid Airflow 3 Environment (Windows Host + WSL Agent) using RabbitMQ Bridge.
---

# Airflow Bridge Operations

This skill provides the knowledge and procedures required to operate the Airflow Batch Processor in a hybrid environment where:
- **Host**: Windows (runs Docker Desktop, RabbitMQ, and Bridge Server).
- **Agent**: WSL/Linux (runs development tools, Bridge Client).

## Why This Exists (Context)
The AI Agent operates inside WSL (Linux) but needs to control Docker containers running on the Windows Host. Direct `docker` CLI access is often restricted or complex to tunnel. We use a **message-based bridge** (RabbitMQ) to proxy commands from Agent to Host.

## Architecture
`Agent (WSL) -> [Bridge Client] -> RabbitMQ -> [Bridge Server] -> Host (Windows) -> Docker Desktop`

## Critical Dependencies
1.  **RabbitMQ**: Must be accessible via `RABBITMQ_URL`.
2.  **Bridge Server**: Must be running on the Windows Host (`python bridge_server.py`).
3.  **Bridge Client**: Used by the Agent to send commands (`python bridge_client.py`).

## How to Use (Quickstart)

### 1. Check Container Status
Use the bridge to run `docker ps`:
```bash
uv run --no-project --with pika python airflow-batch-processor/windows_agent_bridge/bridge_client.py "docker ps"
```

### 2. Fetch Logs
```bash
uv run --no-project --with pika python airflow-batch-processor/windows_agent_bridge/bridge_client.py "docker logs --tail 100 airflow-scheduler"
```

### 3. Restart Stack
When configuration changes (e.g. `docker-compose.yml`):
```bash
uv run --no-project --with pika python airflow-batch-processor/windows_agent_bridge/bridge_client.py "docker compose -f ../docker/docker-compose.yml up -d --force-recreate"
```

## Critical Configurations (Airflow 3 Specifics)

### 1. Execution API URL
*   **Variable**: `AIRFLOW__CORE__EXECUTION_API_SERVER_URL`
*   **Value**: `http://airflow-webserver:8080/execution/`
*   **Crucial**: You **MUST** include the `/execution/` suffix.
*   **Symptom of Failure**: `405 Method Not Allowed`.

### 2. Internal API URL
*   **Variable**: `AIRFLOW__CORE__INTERNAL_API_URL`
*   **Value**: `http://airflow-webserver:8080`
*   **Symptom of Failure**: `Connection Refused` (tries connecting to localhost).

### 3. Authentication (JWT)
*   **Variable**: `AIRFLOW__API_AUTH__JWT_SECRET`
*   **Value**: Must be a **shared static string** across Scheduler and App (e.g., in `docker-compose.yml`).
*   **Symptom of Failure**: `403 Forbidden` / `InvalidSignatureError`.

## Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `pika.exceptions.AMQPConnectionError` | RabbitMQ unreachable | Check `RABBITMQ_URL` and ensure RabbitMQ is running. |
| `Command not allowed` | Command not valid/whitelisted | Update `ALLOWED_COMMANDS` in `bridge_server.py` and restart server. |
| `PermissionError: [Errno 13]` | Container user cannot write to mount | Run `chmod 777 <dir>` via bridge client. |
| `HTTP 400 Bad Request` (API) | Malformed Python Request | Check `files` key name (use `document`) and Remove `Content-Type: application/json`. |

## Reference
See `examples/` for usage scripts.
