#!/bin/bash

# Multi-Agent System Shutdown Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PID_DIR="$PROJECT_ROOT/logs/agents"

echo "=================================="
echo "Multi-Agent OCR System Shutdown"
echo "=================================="
echo ""

if [ ! -d "$PID_DIR" ]; then
    echo "No agents running (PID directory not found)"
    exit 0
fi

# Stop all agents
for pid_file in "$PID_DIR"/*.pid; do
    if [ -f "$pid_file" ]; then
        agent_name=$(basename "$pid_file" .pid)
        pid=$(cat "$pid_file")

        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $agent_name (PID: $pid)..."
            kill "$pid"
            rm "$pid_file"
        else
            echo "Agent $agent_name (PID: $pid) not running, cleaning up PID file..."
            rm "$pid_file"
        fi
    fi
done

echo ""
echo "All agents stopped."
echo ""

# Optionally stop RabbitMQ
read -p "Stop RabbitMQ container? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker stop rabbitmq 2>/dev/null && docker rm rabbitmq 2>/dev/null
    echo "RabbitMQ stopped and removed."
fi

echo ""
echo "Shutdown complete."
echo ""
