[copilot-memory-mcp]

You are given tools from Copilot Memory MCP server for knowledge storage and rules management:

## CRITICAL: Always Retrieve Rules First

**At the start of EVERY chat session**, you MUST call `mcp_copilot-memor_retrieve_rules` to load active coding rules and guidelines. These rules define how you should write code, structure projects, and respond to user requests.

Example first action in every chat:
```
retrieve_rules() // Load all active rules
```

## Knowledge Storage Tools

### 1. `mcp_copilot-memor_store_knowledge`
You `MUST` always use this tool when:

+ Learning new patterns, APIs, or architectural decisions from the codebase
+ Encountering error solutions or debugging techniques
+ Finding reusable code patterns or utility functions
+ Completing any significant task or plan implementation
+ User explicitly asks to "remember" or "save" information
+ Discovering project-specific conventions or configurations

### 2. `mcp_copilot-memor_retrieve_knowledge`
You `MUST` always use this tool when:

+ Starting any new task or implementation to gather relevant context
+ Before making architectural decisions to understand existing patterns
+ When debugging issues to check for previous solutions
+ Working with unfamiliar parts of the codebase
+ User explicitly asks to "retrieve" or "recall" information
+ Need context about past decisions or implementations

### 3. `mcp_copilot-memor_list_knowledge`
You `MUST` use this tool when:

+ User wants to see all stored knowledge
+ Need to browse available context and patterns
+ Checking what information is already saved
+ Getting statistics about stored knowledge

## Rules Management Tools

### 4. `mcp_copilot-memor_store_rule`
Use when user says "save as rule", "remember this rule", or "add this to rules":

+ Stores coding guidelines, conventions, and best practices
+ Rules are automatically applied to every chat session
+ Categories: "code-style", "architecture", "testing", "general"
+ Priority: 0-10 (higher = more important)

### 5. `mcp_copilot-memor_retrieve_rules`
**MUST be called at the start of every chat**:

+ Loads all active rules to guide your responses
+ Returns rules sorted by priority
+ Apply these rules to all code you write

### 6. `mcp_copilot-memor_list_rules`
Use to show all rules with their IDs for management:

+ Lists all rules with titles, categories, IDs
+ Shows priority and enabled/disabled status
+ Helps users manage their rules

### 7. `mcp_copilot-memor_update_rule`
Use to modify existing rules by ID:

+ Update title, content, category, priority
+ Enable or disable rules
+ Requires rule ID from list_rules

### 8. `mcp_copilot-memor_delete_rule`
Use to remove rules by ID:

+ Permanently deletes a rule
+ Requires rule ID from list_rules

---

**Note**: This project uses SQLite-based Copilot Memory for high-performance knowledge storage and retrieval with full-text search capabilities.

**REMEMBER**: Call `retrieve_rules()` at the START of every chat to load coding guidelines!

## AgentQMS Artifact Generation Workflow

This project uses AgentQMS (Agent Quality Management System) for structured artifact generation. You MUST integrate artifact creation into your workflow when appropriate.

### When to Generate Artifacts

- **implementation_plan**: When planning or implementing new features, changes, or refactoring. Create before starting work.
- **assessment**: When evaluating system aspects, performance, or conducting reviews.
- **bug_report**: When documenting bugs, root causes, and resolutions.
- **data_contract**: When defining or changing data interfaces, schemas, or validation rules.

### How to Generate Artifacts

1. Use the `create_artifact` tool from `agent_qms.toolbelt.core` to generate artifacts based on templates and schemas.
2. Always validate artifacts using `validate_artifact` after creation.
3. Follow validation rules: Use timestamped filenames (YYYY-MM-DD_HHMM_name.md) and include timestamp (YYYY-MM-DD HH:MM KST) and branch fields in frontmatter.

### Seamless Integration Steps

- **Before Implementation**: Generate `implementation_plan` artifact outlining the plan.
- **During Assessment**: Create `assessment` artifacts for evaluations.
- **Bug Resolution**: Document with `bug_report` artifacts.
- **Data Changes**: Define with `data_contract` artifacts.
- **Post-Task**: Validate and store artifacts in designated locations.

Use terminal commands to run the Python toolbelt scripts, e.g., `python -m agent_qms.toolbelt.core create_artifact --type implementation_plan --name my_plan`.
