# Windows Agent Bridge Setup Summary

**Goal**: Enable the AI Agent (in Linux Docker) to execute `docker` commands on the Windows Host to debug Airflow containers.

## 1. Architecture
*   **Broker**: RabbitMQ (allows communication between Host and Container).
*   **Server**: Python script on **Windows** that listens for commands and runs them.
*   **Client**: Python script on **Linux** (Agent) that sends commands.

## 2. Prerequisites
*   **RabbitMQ**: Running on port `5672`.
    *   Management UI: [http://localhost:15672](http://localhost:15672) (guest/guest)
*   **Python**: Installed on Windows.

## 3. Installation (Windows Host)
Run these commands in PowerShell from the project root:

1.  **Navigate to Bridge Directory**:
    ```powershell
    cd airflow-batch-processor/windows_agent_bridge
    ```

2.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

3.  **Start Bridge Server**:
    ```powershell
    uv run python bridge_server.py
    ```
    *   *Keep this window open. It will show "Waiting for commands..."*

## 4. Usage (Agent / Linux)
The Agent can now use the client script to run commands on your behalf.

**Test Command**:
```bash
uv run python airflow-batch-processor/windows_agent_bridge/bridge_client.py "docker ps"
```

**Debug Airflow Task**:
```bash
uv run python airflow-batch-processor/windows_agent_bridge/bridge_client.py "docker exec airflow-scheduler airflow tasks test batch_processor_dag preprocess 2024-01-25T13:00:00Z"
```

## 5. Troubleshooting
*   **Connection Refused**: Ensure RabbitMQ is running.
*   **Host Access**: If the Agent cannot reach the host, try changing `RABBITMQ_HOST` in `bridge_client.py` to your specific internal IP string (e.g., `192.168.x.x`) if `host.docker.internal` fails.
*   **Permissions**: The Bridge only allows specific commands (defined in `bridge_server.py`).