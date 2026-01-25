#!/bin/bash

# Multi-Agent System Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=================================="
echo "Multi-Agent OCR System Startup"
echo "=================================="
echo ""

# Check if RabbitMQ is running
echo "Checking RabbitMQ..."
if ! nc -z localhost 5672 2>/dev/null; then
    echo "❌ RabbitMQ not running on localhost:5672"
    echo ""
    echo "Starting RabbitMQ using Docker..."
    docker run -d \
        --name rabbitmq \
        -p 5672:5672 \
        -p 15672:15672 \
        -e RABBITMQ_DEFAULT_USER=admin \
        -e RABBITMQ_DEFAULT_PASS=admin123 \
        rabbitmq:3.12-management

    echo "Waiting for RabbitMQ to start..."
    sleep 10
fi

echo "✓ RabbitMQ is running"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import pika" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install pika opencv-python-headless rembg httpx openai
}

echo "✓ Dependencies installed"
echo ""

# Export environment variables
export PYTHONPATH="$PROJECT_ROOT"
export RABBITMQ_HOST="localhost"

# Agent start function
start_agent() {
    local agent_name=$1
    local agent_module=$2
    local log_file="$PROJECT_ROOT/logs/agents/${agent_name}.log"

    mkdir -p "$(dirname "$log_file")"

    echo "Starting $agent_name..."
    python3 -m "$agent_module" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$PROJECT_ROOT/logs/agents/${agent_name}.pid"
    echo "  → Started with PID $pid (log: $log_file)"
}

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs/agents"

echo "Starting agents..."
echo ""

# Start all agents
start_agent "preprocessor" "ocr.agents.ocr_preprocessing_agent"
sleep 2

start_agent "inference" "ocr.agents.ocr_inference_agent"
sleep 2

start_agent "validator" "ocr.agents.ocr_validation_agent"
sleep 2

start_agent "orchestrator" "ocr.agents.orchestrator_agent"
sleep 2

start_agent "linter" "ocr.agents.linting_agent"
sleep 2

start_agent "worker-1" "ocr.workers.ocr_worker"
sleep 2

echo ""
echo "=================================="
echo "All agents started successfully!"
echo "=================================="
echo ""
echo "RabbitMQ Management UI: http://localhost:15672"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "Agent Logs: $PROJECT_ROOT/logs/agents/"
echo ""
echo "To stop all agents, run:"
echo "  $SCRIPT_DIR/stop_agents.sh"
echo ""
echo "To view logs in real-time:"
echo "  tail -f $PROJECT_ROOT/logs/agents/*.log"
echo ""
