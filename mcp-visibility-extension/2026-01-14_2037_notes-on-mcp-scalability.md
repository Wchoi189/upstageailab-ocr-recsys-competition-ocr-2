Read [](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/pyproject.toml#1-1), lines 1 to 50

Based on your current codebase and the broader ecosystem, here is an analysis of FastMCP and scalable operations.

### 1. FastMCP: Is it worth it?

**FastMCP** is a higher-level wrapper around the standard `mcp` SDK (which you are currently using).

*   **Advantages**:
    *   **Developer Experience**: It reduces boilerplate significantly. Instead of manually defining `Tool`, `Resource`, and `inputSchema` dictionaries, you use decorators (`@mcp.tool()`) on standard Python functions with type hints. It auto-generates the JSON schema from your Pydantic models or type hints.
    *   **Built-in Utilities**: It often includes built-in support for image returns, better error handling, and simpler context management.
    *   **One-Line Server**: It abstracts away the `stdio_server` / `sse_server` setup.

*   **Why you might NOT want it right now**:
    *   **Granular Control**: Your unified_server.py is doing complex things that a wrapper might hide:
        *   Manual aggregation of multiple sub-modules (agentqms, `compass`, `experiments`).
        *   Custom Middleware Interception (`TelemetryPipeline`, `PolicyViolation`).
        *   Custom implementations of `read_resource` that route based on URI schemes (`bundle://`, `mcp://`).
    *   **Migration Cost**: You would have to rewrite your entire aggregation layer.

**Verdict**: FastMCP is excellent for *new, single-purpose agents*. For your current "Unified Server" architecture—which acts as a gateway router and middleware layer—the raw `mcp` SDK gives you the necessary low-level control.

### 2. Considerations for Scalable MCP Operations

Scaling MCP from a single local script to a robust operation involves several layers:

**A. Transport & Runtime**
*   **stdio vs. SSE**:
    *   **Stdio (Current Primary):** Perfect for local VS Code integration. Zero network latency, simpler security (process ownership).
    *   **SSE (Server-Sent Events):** Required for distributed agents. If you move your image processing or massive OCR jobs to a GPU cluster, you can't run them as a local subprocess. You would expose them via SSE/HTTP.
*   **Statelessness**: To scale, your server should be as stateless as possible. Your `SESSION_ID` implementation is lightweight, but if you had 1,000 users, storing token counts in memory (unified_server.py) would reset on crash. A scalable solution uses Redis or a database for session state.

**B. Modularity & routing**
*   **The "Router" Pattern**: You are already doing this! Your unified_server.py doesn't implement tool logic; it imports it.
    *   *Scalability Step:* Instead of importing Python modules (`import AgentQMS`), a scalable router would forward requests to *other MCP servers* over HTTP. This allows you to deploy the "OCR Agent" on a GPU machine and the "QMS Agent" on a CPU machine.

**C. Observability (Your Active Task)**
*   **Telemetry**: Essential for cost control. The extension you are building is a critical scalability component because it gives visibility into "token bleed."
*   **Policy Enforcement**: As you add more autonomous agents, you need guardrails (like your `policies.py`) to prevent agents from looping or consuming infinite resources.

**D. Security**
*   **Input Validation**: A public-facing MCP server is a Remote Code Execution (RCE) vector. Validation (which you have in `middleware-policies.yaml`) becomes critical.
*   **Context Isolation**: Ensuring Agent A cannot read Agent B's research notes if they are sensitive.

**Summary for your Project**:
You are currently building a **Monolithic Gateway**. It works well for a single developer or small team. To scale, you would eventually split AgentQMS and `ExperimentManager` into separate microservices and use a lightweight "MCP Proxy" to route traffic between them.
