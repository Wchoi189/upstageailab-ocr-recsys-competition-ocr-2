# @qwen Syntax Guide for AI Agents

## Overview
The `@qwen` syntax allows AI agents to execute Qwen Coder commands directly in chat conversations. This enables seamless integration between conversational AI workflows and automated code/document processing.

## Syntax
```
@qwen [command description]
```

## How It Works
1. **Chat Input**: AI agents include `@qwen` commands in conversation
2. **Command Parser**: System extracts and interprets the command
3. **Qwen Execution**: Qwen Coder executes the task with AgentQMS context
4. **Results**: Command output is returned to the conversation

## Usage Examples

### Basic Commands
```bash
# In chat conversation:
AI Agent: "@qwen fix validation errors in docs/artifacts/assessments/"

# System executes:
qwen --approval-mode yolo --include-directories /workspaces/agent_qms \
     --prompt "Follow AgentQMS/knowledge/agent/system.md... TASK: fix validation errors..."
```

### File Operations
```bash
@qwen rename 02-performance-optimization.md to 2025-11-25_1200_assessment-performance-optimization.md
@qwen add frontmatter to docs/artifacts/assessments/README.md
@qwen move file.md to correct directory based on type
```

### Validation & Compliance
```bash
@qwen run AgentQMS validation on all artifacts
@qwen fix all naming convention violations
@qwen check compliance status and generate report
```

### Artifact Creation
```bash
@qwen create implementation plan for user authentication
@qwen create assessment for security audit
@qwen create bug report for login issue
```

## Command Interpreter Script

Use the provided script for testing and integration:

```bash
# Test the interpreter
./.qwen/qwen-chat.sh "@qwen validate docs/artifacts/"

# Direct execution
./.qwen/qwen-chat.sh "Please @qwen fix the frontmatter in assessment files"
```

## AI Agent Integration Guidelines

### For AI Agents Using @qwen:

1. **Be Specific**: Provide clear, actionable commands
   ```bash
   ✅ "@qwen fix validation errors in docs/artifacts/assessments/"
   ❌ "@qwen do something with files"
   ```

2. **Include Context**: Reference specific files or directories
   ```bash
   ✅ "@qwen add frontmatter to docs/artifacts/bug_reports/BUG_2025-11-28_1409_001_qwen-cli-git-detection.md"
   ✅ "@qwen rename all files in docs/artifacts/ to follow naming conventions"
   ```

3. **Use AgentQMS Terminology**: Reference framework concepts
   ```bash
   ✅ "@qwen create implementation plan for feature X"
   ✅ "@qwen validate artifacts against AgentQMS rules"
   ```

### Response Format:
When AI agents receive @qwen command results, they should:
- Acknowledge successful execution
- Summarize changes made
- Suggest next steps if applicable

## One-Shot Examples for AI Agents

### Example 1: Validation Fix
```
Human: The artifact validation is failing
Assistant: I need to fix the validation issues. @qwen run validation and fix all violations in docs/artifacts/
System: 🤖 Executing @qwen command: run validation and fix all violations in docs/artifacts/
[...Qwen output...]
Assistant: Great! All validation violations have been fixed. The compliance rate improved from 40% to 60%.
```

### Example 2: File Creation
```
Human: We need a plan for the new feature
Assistant: I'll create an implementation plan using AgentQMS. @qwen create implementation plan for user dashboard feature
System: 🤖 Executing @qwen command: create implementation plan for user dashboard feature
[...Qwen creates the plan...]
Assistant: Perfect! I've created the implementation plan at docs/artifacts/implementation_plans/2025-11-28_1430_implementation_plan_user-dashboard.md
```

### Example 3: Code Fix
```
Human: There's a bug in the authentication code
Assistant: Let me create a bug report and then fix the issue. @qwen create bug report for authentication failure and then fix the code
System: 🤖 Executing @qwen command: create bug report for authentication failure and then fix the code
[...Qwen creates bug report and fixes code...]
```

## Best Practices

### For AI Agents:
- **Test Commands**: Use the interpreter script to test commands before production use
- **Provide Context**: Include file paths and specific requirements
- **Handle Errors**: Be prepared to refine commands if they fail
- **Follow Up**: Verify results and suggest next steps

### For System Integration:
- **Logging**: Log all @qwen command executions for audit trails
- **Validation**: Always validate results after execution
- **Fallbacks**: Have manual execution paths if @qwen fails
- **Security**: Ensure commands only affect allowed directories

## Troubleshooting

### Common Issues:
1. **Unclear Commands**: Refine with more specific instructions
2. **Path Issues**: Use absolute paths or verify current working directory
3. **Permission Errors**: Ensure Qwen has proper file access
4. **Complex Tasks**: Break down into smaller, sequential commands

### Debug Mode:
```bash
# Test parsing without execution
./.qwen/qwen-chat.sh --dry-run "@qwen validate files"

# Verbose output
./.qwen/qwen-chat.sh --verbose "@qwen fix file.md"
```

## Integration with Chat Platforms

### Discord/Slack Integration:
- Create bot commands that trigger @qwen parsing
- Use webhooks to send results back to channels
- Implement approval workflows for destructive operations

### IDE Integration:
- VS Code extension that recognizes @qwen in comments
- Execute commands directly from editor
- Show results in output panel

### CI/CD Integration:
- Parse @qwen commands from commit messages
- Execute automated fixes in pipelines
- Generate compliance reports

This @qwen syntax enables natural, conversational AI agent collaboration while maintaining structured, automated execution of complex tasks.</content>
<parameter name="filePath">/workspaces/agent_qms/.qwen/README-chat-integration.md