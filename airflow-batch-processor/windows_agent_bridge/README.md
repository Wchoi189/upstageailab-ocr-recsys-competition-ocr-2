# Windows Agent Bridge

This bridge allows the AI Agent (running in a Linux Docker container) to execute specific Docker commands on the Windows Host. It uses RabbitMQ as a message broker.

## Prerequisites

1.  **RabbitMQ**: Must be running and accessible from both Windows Host and the Docker container.
    *   Default assumption: access via `localhost` on Windows and `host.docker.internal` (or gateway IP) from Linux.
2.  **Python 3.11**: Recommended version is 3.11.14 (3.11.9 also works). Use pyenv to manage Python versions.
3.  **uv Package Manager**: Used for dependency management and virtual environments.

## Setup on Windows Host (The "Legs")

### Option 1: Using uv (Recommended)
1.  Install dependencies:
    ```powershell
    uv pip install -r requirements.txt
    ```

2.  Run the server:
    ```powershell
    uv run python bridge_server.py
    ```
    *   It will print: `[*] Waiting for commands...`

### Option 2: Traditional pip
1.  Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```

2.  Run the server:
    ```powershell
    python bridge_server.py
    ```
    *   It will print: `[*] Waiting for commands...`

## Environment Variables

For clients to connect to RabbitMQ from the Windows host:
- Set environment variable: `RABBITMQ_HOST=localhost`
- Alternative: Modify the client configuration to use "localhost" instead of "host.docker.internal"

## Usage from Agent (The "Brain")

The agent can now use `bridge_client.py` to send commands.

```bash
# Example
python bridge_client.py "docker ps"
```

## Security Note

This bridge executes commands via `subprocess`. Only commands listed in `ALLOWED_COMMANDS` (in `bridge_server.py`) are permitted. Currently:
*   `docker ps`
*   `docker images`
*   `docker logs`
*   `docker exec`
*   `airflow tasks test`

## Development

For development purposes, you can use the provided Makefile to manage the bridge:

```bash
# Install dependencies
make install

# Run the server
make run

# Create a test script for client testing
make create-test-script

# Clean up virtual environment
make clean
```

## Windows-Specific Usage

On Windows, due to environment variable handling differences, the most reliable way to run the client is using a batch file:

```batch
# Create and run the test script
make create-test-script
test_env.bat
```

Or manually:

```batch
# Activate virtual environment and set environment variable
call .venv\Scripts\activate.bat
set RABBITMQ_HOST=localhost
python bridge_client.py "docker ps"
```
