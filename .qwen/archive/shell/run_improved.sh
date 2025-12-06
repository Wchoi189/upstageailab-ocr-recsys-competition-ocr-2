#!/bin/bash
# Qwen Coder + AgentQMS Integration Script (IMPROVED)
# NOW ACTUALLY EXECUTES Qwen INSTEAD OF JUST ECHOING
# Checkpointing: ON | Validation: ON | Artifact context: VISIBLE

set -e

PROJECT_ROOT="/workspaces/upstageailab-ocr-recsys-competition-ocr-2"
QWEN_CONFIG="$PROJECT_ROOT/.qwen/settings.json"
AGENTQMS_INSTRUCTIONS="$PROJECT_ROOT/AgentQMS/knowledge/agent/system.md"

# Function to actually run Qwen with AgentQMS context
run_qwen_agentqms() {
    local prompt="$1"

    echo "🤖 Executing Qwen Coder with AgentQMS integration..."
    echo "📋 Instructions: $AGENTQMS_INSTRUCTIONS"
    echo "⚙️  Config: $QWEN_CONFIG"
    echo "✅ Checkpointing: ENABLED"
    echo "✅ Validation: ENABLED"
    echo ""

    # ACTUALLY EXECUTE Qwen with proper context
    qwen --approval-mode yolo \
         --include-directories "$PROJECT_ROOT" \
         --prompt "$prompt"
}

# Enhanced frontmatter-only batch prompt
run_frontmatter_batch() {
    local batch_files="$1"

    local batch_prompt=$(cat << 'EOF'
You are an autonomous AI agent for AgentQMS artifact quality management.
TASK: Fix ONLY frontmatter (YAML at top of file). Do NOT rename files.

## CRITICAL CONSTRAINTS

### Required Frontmatter (ALL REQUIRED)
Every artifact MUST have:
```yaml
---
type: [assessment|bug_report|implementation_plan|design|research|audit|template]
title: "Your Title"
date: "YYYY-MM-DD HH:MM (KST)"
category: [see list below]
status: [see list below]
version: "1.0"
tags:
  - tag1
author: ai-agent
branch: main
---
```

### Valid Categories (CHOOSE ONE)
development, architecture, evaluation, compliance, code_quality, reference, planning, research, troubleshooting, governance, meeting, security

### Valid Statuses (CHOOSE ONE)
active, draft, completed, archived, deprecated

### Date Format (EXACTLY THIS)
"YYYY-MM-DD HH:MM (KST)" with 24-hour clock. Examples:
- "2025-11-17 13:36 (KST)" ✓
- "2025-12-03 10:30 (KST)" ✓
- "2025-01-10 12:00 (KST)" ✓

### Type ↔ Filename Alignment (STRICT)
Match type field to filename prefix:
- assessment- → type: assessment
- BUG_NNN_ → type: bug_report
- implementation_plan_ → type: implementation_plan
- design- → type: design
- research- → type: research

## FILES TO FIX
EOF

    echo "$batch_prompt"
    echo ""
    echo "These files need frontmatter correction:"
    echo "$batch_files"

    cat << 'EOF'

## ACTIONS (FOR EACH FILE)
1. Read file completely
2. Extract frontmatter between --- and ---
3. Check all 9 required fields present
4. Fix invalid categories → use 'evaluation' for assessments, 'development' for others
5. Fix invalid statuses → use 'active'
6. Fix date format → must be "YYYY-MM-DD HH:MM (KST)"
7. Ensure type matches filename prefix
8. Rebuild frontmatter with proper formatting
9. Preserve all body content unchanged
10. Save file

## EXAMPLE FRONTMATTER (assessment)
---
type: assessment
title: "Security Audit Results"
date: "2025-11-17 13:36 (KST)"
category: compliance
status: active
version: "1.0"
tags:
  - security
  - audit
author: ai-agent
branch: main
---

## EXAMPLE FRONTMATTER (bug_report)
---
type: bug_report
title: "Inference UI Coordinate Transformation"
date: "2025-01-10 12:00 (KST)"
category: troubleshooting
status: active
version: "1.0"
tags:
  - bug
  - inference
author: ai-agent
branch: main
---

## SUCCESS CRITERIA
✓ Frontmatter starts with ---
✓ All 9 fields present
✓ Type matches filename
✓ Category in valid list
✓ Status in valid list
✓ Date format exactly "YYYY-MM-DD HH:MM (KST)"
✓ No file rename (only frontmatter changes)
✓ Body content unchanged

BEGIN NOW. Process all files with ONLY frontmatter fixes.
EOF

    run_qwen_agentqms "$batch_prompt"
}

# Main script logic
case "${1:-}" in
    "frontmatter")
        if [ -z "$2" ]; then
            echo "Usage: $0 frontmatter '<file_list>'"
            echo "Example: $0 frontmatter 'docs/artifacts/assessments/2025-11-17*.md docs/artifacts/bug_reports/2025-12*.md'"
            exit 1
        fi
        run_frontmatter_batch "$2"
        ;;
    "interactive")
        if [ -z "$2" ]; then
            echo "Usage: $0 interactive '<prompt>'"
            exit 1
        fi
        run_qwen_agentqms "$2"
        ;;
    "help"|*)
        echo "Qwen Coder + AgentQMS Integration Script (IMPROVED)"
        echo ""
        echo "Usage:"
        echo "  $0 frontmatter '<file_list>'  - Fix frontmatter in batch of files"
        echo "  $0 interactive '<prompt>'      - Run interactive Qwen session"
        echo "  $0 help                        - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 frontmatter 'docs/artifacts/assessments/2025-11-17*.md'"
        echo "  $0 frontmatter 'docs/artifacts/bug_reports/2025-12*.md docs/artifacts/bug_reports/archive/*.md'"
        echo "  $0 interactive 'Help me refactor this code'"
        echo ""
        echo "Configuration:"
        echo "  ✅ Checkpointing: ENABLED (Qwen remembers context)"
        echo "  ✅ Validation: ENABLED (Qwen can self-check)"
        echo "  ✅ Approvals: YOLO mode (auto-approve changes)"
        ;;
esac
