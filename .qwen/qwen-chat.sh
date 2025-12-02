#!/bin/bash
# @qwen Command Interpreter for AI Agent Chat Integration

PROJECT_ROOT="/workspaces/agent_qms"

# Function to show usage
show_usage() {
    cat << 'EOF'
@qwen Command Interpreter - AI Agent Integration
===============================================

USAGE:
  @qwen [command description]

EXAMPLES:
  @qwen fix the validation errors in docs/artifacts/assessments/
  @qwen rename file.md to follow AgentQMS naming conventions

IN CHAT CONVERSATIONS:
  AI Agent: "@qwen fix all validation violations in docs/artifacts/"

EOF
}

# Main execution
main() {
    local input_text="$*"

    # If no arguments, show usage
    if [ -z "$input_text" ]; then
        show_usage
        exit 0
    fi

    # Look for @qwen in input
    if echo "$input_text" | grep -q "@qwen"; then
        # Extract command after @qwen
        local command=$(echo "$input_text" | sed 's/.*@qwen *//')

        echo "Executing @qwen command: $command"
        echo ""

        # Build prompt
        local prompt="Follow AgentQMS/knowledge/agent/system.md for all operations.

TASK: $command

Execute the requested task following AgentQMS guidelines."

        # Execute Qwen
        echo "Running Qwen Coder..."
        qwen --approval-mode yolo \
             --include-directories /workspaces/agent_qms \
             --prompt "$prompt"

        echo ""
        echo "@qwen command completed"
    else
        echo "No @qwen command found in input"
        echo ""
        show_usage
        exit 1
    fi
}

main "$@"