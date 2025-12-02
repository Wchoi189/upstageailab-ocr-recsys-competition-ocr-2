#!/bin/bash
# Qwen Coder + AgentQMS Integration Script

set -e

PROJECT_ROOT="/workspaces/upstageailab-ocr-recsys-competition-ocr-2"
QWEN_CONFIG="$PROJECT_ROOT/.qwen/settings.json"
AGENTQMS_INSTRUCTIONS="$PROJECT_ROOT/AgentQMS/knowledge/agent/system.md"

# Function to run Qwen with AgentQMS context
run_qwen_agentqms() {
    local prompt="$1"
    local interactive="$2"

    echo "🤖 Starting Qwen Coder with AgentQMS integration..."
    echo "📋 Instructions: $AGENTQMS_INSTRUCTIONS"
    echo "⚙️  Config: $QWEN_CONFIG"
    echo ""
    echo "⚠️  Note: Due to Qwen checkpointing/Git detection issue, using alternative approach..."
    echo "   You can run Qwen directly with: qwen --yolo --prompt \"[your prompt]\""
    echo "   Or use the manual validation approach below."
    echo ""

    # For now, just show the prompt that would be used
    echo "📝 Generated Prompt:"
    echo "==================="
    echo "$prompt"
    echo "==================="
    echo ""
    echo "💡 To run manually:"
    echo "   1. Copy the prompt above"
    echo "   2. Run: qwen --approval-mode yolo --include-directories /workspaces/upstageailab-ocr-recsys-competition-ocr-2 --prompt \"<paste prompt here>\""
    echo "   3. Or use any other AI tool with the prompt"
}

# Function to run document validation
run_validation_agent() {
    local validation_prompt
    validation_prompt=$(cat << 'EOF'
# AUTONOMOUS DOCUMENT VALIDATION RESOLUTION AGENT

You are an autonomous AI agent specialized in resolving document validation issues for the AgentQMS Quality Management System.

## PRIMARY INSTRUCTIONS
Read and follow: AgentQMS/knowledge/agent/system.md

## TASK
Fix all document validation violations in docs/artifacts/ following AgentQMS naming conventions and frontmatter requirements.

## SCOPE
- Only modify files in docs/artifacts/
- Ignore: docs/*.md, docs/assets/**, docs/archive/**, AgentQMS/**, .agentqms/**, .copilot/**, .cursor/**, .github/**, docs_deprecated/**

## FIXES NEEDED
1. Rename files to: YYYY-MM-DD_HHMM_{ARTIFACT_TYPE}_descriptive-name.md
2. Add proper YAML frontmatter
3. Move files to correct directories

Begin by scanning docs/artifacts/ for violations.
EOF
)

    run_qwen_agentqms "$validation_prompt" "false"
}

# Function to create new artifacts
create_artifact() {
    local type="$1"
    local name="$2"
    local title="$3"

    if [ -z "$type" ] || [ -z "$name" ] || [ -z "$title" ]; then
        echo "Usage: $0 create <type> <name> <title>"
        echo "Types: plan, assessment, bug-report, design, research, template"
        exit 1
    fi

    local create_prompt
    create_prompt=$(cat << EOF
Create a new $type artifact using AgentQMS framework.

INSTRUCTIONS: Follow AgentQMS/knowledge/agent/system.md exactly.

TASK:
1. Use the appropriate make command: make create-$type NAME=$name TITLE="$title"
2. Ensure proper naming and frontmatter
3. Place in correct directory: docs/artifacts/

Execute the creation process.
EOF
)

    run_qwen_agentqms "$create_prompt" "false"
}

# Function to run manual validation
run_manual_validation() {
    echo "🔍 Running manual document validation..."
    ./.qwen/manual_validate.sh
}

# Main script logic
case "${1:-}" in
    "validate")
        run_validation_agent
        ;;
    "validate-manual")
        run_manual_validation
        ;;
    "create")
        create_artifact "$2" "$3" "$4"
        ;;
    "interactive")
        run_qwen_agentqms "${2:-}" "true"
        ;;
    "help"|*)
        echo "Qwen Coder + AgentQMS Integration Script"
        echo ""
        echo "Usage:"
        echo "  $0 validate                    - Run document validation agent (Qwen)"
        echo "  $0 validate-manual             - Run manual document validation"
        echo "  $0 create <type> <name> <title> - Create new artifact"
        echo "  $0 interactive [prompt]         - Start interactive session"
        echo "  $0 help                         - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 validate"
        echo "  $0 create plan my-feature \"New Feature Implementation\""
        echo "  $0 interactive \"Help me refactor this code\""
        ;;
esac
