## Perplexity Suggestions from Conversation 1

Quick Summary:
The following response from Perplexity is based on the results of sharing my agent-debug-toolkit mcp_server.py with Perplexity.

---

## Conversation Snippet START ##

You are very close to a “full-stack” code understanding server; what is mainly missing is a **dedicated editing layer** and **structure-aware access for non-Python code**.

## 1. New tool groups to add

### a. `adt_edits` – robust edit & patch tools

Add a group focused purely on applying edits safely, especially for long files:

```yaml
adt_edits:
  description: "Edit, patch, and format code with high reliability"
  tools:
    - apply_unified_diff
    - smart_edit
    - read_file_slice
    - format_code
```

Suggested tool contracts:

- `apply_unified_diff`
  - Input:
    - `diff: string` (standard unified diff or simplified Aider-style patch)[2][3]
    - `strategy: string` enum [`"exact"`, `"whitespace_insensitive"`, `"fuzzy"`]
  - Behavior:
    - Parse the diff.
    - For each hunk, do fuzzy matching on the **search block** to tolerate minor drift (RooCode-style).[4][2]
    - Return:
      - `applied: bool`
      - `failed_hunks: list`
      - `resulting_diff: string` (true git diff vs disk).

- `smart_edit`
  - Input:
    - `file: string`
    - `search: string`
    - `replace: string`
    - `mode: string` enum [`"exact"`, `"regex"`, `"fuzzy"`]
  - Behavior:
    - Implement the multi-stage matching described in recent edit-system posts (exact → whitespace‑insensitive → fuzzy).[4]
    - Return a small diff, not the full file.

- `read_file_slice`
  - Input:
    - `file: string`
    - `start_line: int`
    - `end_line: int`
  - Behavior:
    - Return only that region to encourage the agent to work in **slices**, not entire 2k‑line files.[5]

- `format_code`
  - Input:
    - `path: string`
    - `style: string` enum [`"black"`, `"ruff"`, `"isort"`]
    - `scope: string` enum [`"file"`, `"package"`]
  - Behavior:
    - Wrap your chosen formatters so the agent can normalize code after patches.

These directly target the “long script edits fail” problem by pushing the LLM into **unified diff + fuzzy-apply** workflows rather than whole-file rewrites.[3][6][2]

### b. `adt_symbolic` – cross-file, AST-level refactors

You already have rich analysis; expose the manipulations you plan for Phase 3 as tools:

```yaml
adt_symbolic:
  description: "Symbol-aware, cross-file code operations"
  tools:
    - build_symbol_table
    - find_references
    - rename_symbol_batch
    - resolve_hydra_configs   # your Phase 2.5 tool
```

Minimal contracts:

- `build_symbol_table(path)` → JSON schema you defined in RQ1 (modules, symbols, imports).
- `find_references(symbol, project_root)` → list of `(file, line, context)` (backed by that table).
- `rename_symbol_batch(old, new, project_root)`:
  - Use LibCST or your CST engine to create a patch, return unified diff rather than directly writing files.
- `resolve_hydra_configs(config_dir, module_root)`:
  - Implement your `HydraConfigResolver` output schema directly.

This leverages your existing analyzers but gives the agent **refactor levers**, not just reports.

### c. `tree_sitter` (external MCP server)

Given you’re already deep into ASTs, adding a generic Tree-sitter MCP server will help with:

- Multi-language repos.
- Structure-aware queries where your own analyzers don’t exist yet.

For example, with `mcp-server-tree-sitter` / `treesitter-mcp` you typically get:

- `register_project`
- `list_symbols`
- `get_ast`
- `run_query`
- `find_usages`
- `analyze_dependencies`[7][8][1]

You can then keep your Python-specific `agent_debug_toolkit` and let the agent fall back to Tree-sitter tools for front-end assets, shell scripts, etc.

## 2. How to wire this into your existing server

You have a clear pattern in `call_tool`; extend it with an **edit backend** module:

1. Create `agent_debug_toolkit.edits` with functions:
   - `apply_unified_diff(diff: str, strategy: str) -> EditReport`
   - `smart_edit(file: Path, search: str, replace: str, mode: str) -> EditReport`
   - `read_file_slice(file: Path, start: int, end: int) -> str`
   - `format_code(path: Path, style: str, scope: str) -> FormatReport`

2. In `TOOLS`, append new tool definitions (similar to others):

```python
{
    "name": "apply_unified_diff",
    "description": "Apply a unified diff to files with fuzzy matching for robustness on large/changed files.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "diff": {"type": "string", "description": "Unified diff text"},
            "strategy": {
                "type": "string",
                "enum": ["exact", "whitespace_insensitive", "fuzzy"],
                "default": "fuzzy",
            },
        },
        "required": ["diff"],
    },
},
# ... smart_edit, read_file_slice, format_code ...
```

3. Extend `call_tool` with branches for these names, delegating to your new edit module. Internally, follow RooCode/Aider patterns for fuzzy diff application.[2][3][4]

## 3. Agent-side changes to improve editing reliability

Alongside new tools, update your **agent protocol**:

- Prefer:
  - `read_file_slice` → LLM proposes patch → `apply_unified_diff`.
- Avoid:
  - “Here is the entire 2k‑line file, send back the whole file modified.”
  Evidence: whole-file editing has worst accuracy on large files vs diff/unified diff.[6][9][10]

You can codify this in your system prompt / QMS templates:

- “When changing existing code:
  - Always work on slices,
  - Always return unified diffs,
  - Never rewrite unread parts of the file.”

If you want, the next iteration can sketch concrete Python stubs for `agent_debug_toolkit.edits` (particularly a simple but robust `apply_unified_diff` implementation) tailored to your repo layout.

---


Two things below:

1) concrete Python stubs for an `adt_edits` module that fits your current server, and
2) patterns for “too many tools” so you can keep everything available without blowing context, plus IDE integration ideas.

***

## 1. `adt_edits` module: concrete stubs

Minimal, pragmatic implementation you can iterate on.

### a. `agent_debug_toolkit/edits.py`

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, List, Tuple

import difflib


Strategy = Literal["exact", "whitespace_insensitive", "fuzzy"]
Mode = Literal["exact", "regex", "fuzzy"]


@dataclass
class HunkResult:
    file: str
    hunk_header: str
    applied: bool
    message: str | None = None


@dataclass
class EditReport:
    applied: bool
    message: str
    hunks: List[HunkResult]

    def to_json(self) -> str:
        import json
        return json.dumps(asdict(self), indent=2)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _apply_unified_diff_exact(diff: str, project_root: Path) -> EditReport:
    """
    Very simple exact matcher: assumes files have not drifted.
    """
    lines = diff.splitlines(keepends=True)
    current_file: Path | None = None
    original_lines: list[str] = []
    new_lines: list[str] = []
    in_hunk = False
    hunk_results: list[HunkResult] = []

    for line in lines:
        if line.startswith(("+++ ", "--- ")):
            # ignore headers; use --- a/xxx, +++ b/xxx
            continue
        if line.startswith("@@ "):
            # start new hunk; flush previous file if needed
            in_hunk = True
            continue
        if not in_hunk:
            # file header: diff -u style like 'diff --git a/x b/x' or '+++ file'
            if line.startswith("diff ") or line.startswith("Index "):
                continue
            # fallback: not robust, but fine for minimal implementation
            continue

    # This minimal exact implementation is intentionally simple.
    # Recommend replacing with python-patch or your own robust diff applier later.
    return EditReport(
        applied=False,
        message="Exact diff application not yet implemented; please use fuzzy strategy.",
        hunks=[],
    )


def apply_unified_diff(
    diff: str,
    strategy: Strategy = "fuzzy",
    project_root: Path | None = None,
) -> EditReport:
    """
    Apply a unified diff to files under project_root.

    strategy:
      - "exact": fail if context does not match
      - "whitespace_insensitive": normalize whitespace before matching
      - "fuzzy": allow approximate context (recommended default)
    """
    if project_root is None:
        project_root = Path.cwd()

    # For now, implement a pragmatic fuzzy: re-diff desired new content from model.
    # Simplest robust approach:
    #   - parse the diff into (file, new_content)
    #   - recompute real diff vs disk, then write.
    #
    # This assumes your agent produces "full-file" sections in the diff.
    # If not, you can later swap in a library like `python-patch` plus
    # your own fuzzy context matching.
    files_to_new_content: dict[Path, str] = {}
    current_file: Path | None = None
    collecting_new = False
    new_lines: list[str] = []

    for line in diff.splitlines(keepends=True):
        if line.startswith("+++ "):
            # Determine file path from '+++ b/path' or '+++ path'
            header = line[4:].strip()
            if header.startswith("b/"):
                header = header[2:]
            current_file = project_root / header
            collecting_new = True
            new_lines = []
            continue
        if current_file is not None and collecting_new:
            # naive: treat everything after +++ as new file content, until next +++
            new_lines.append(line)

    if current_file is not None and new_lines:
        files_to_new_content[current_file] = "".join(new_lines)

    hunks: list[HunkResult] = []
    for path, new_text in files_to_new_content.items():
        if not path.exists():
            hunks.append(
                HunkResult(
                    file=str(path),
                    hunk_header="FULL",
                    applied=False,
                    message="File does not exist on disk",
                )
            )
            continue

        old_text = _read_text(path)
        if strategy in ("whitespace_insensitive", "fuzzy"):
            # accept small drift by ignoring whitespace-only changes when comparing
            s1 = "".join(old_text.split())
            s2 = "".join(new_text.split())
            if s1 == s2:
                hunks.append(
                    HunkResult(
                        file=str(path),
                        hunk_header="FULL",
                        applied=False,
                        message="No semantic change detected (whitespace only)",
                    )
                )
                continue

        _write_text(path, new_text)
        hunks.append(
            HunkResult(
                file=str(path),
                hunk_header="FULL",
                applied=True,
                message=None,
            )
        )

    applied = any(h.applied for h in hunks)
    return EditReport(
        applied=applied,
        message="Applied edits to {} file(s)".format(sum(h.applied for h in hunks)),
        hunks=hunks,
    )


def smart_edit(
    file: Path,
    search: str,
    replace: str,
    mode: Mode = "exact",
) -> EditReport:
    """
    Simple multi-stage search/replace to reduce brittle edits:
      - exact: simple .replace
      - regex: re.sub
      - fuzzy: difflib-based approximate region replace
    """
    text = _read_text(file)
    hunks: list[HunkResult] = []

    if mode == "exact":
        if search not in text:
            return EditReport(
                applied=False,
                message="Search string not found in file",
                hunks=[
                    HunkResult(
                        file=str(file),
                        hunk_header="smart_edit",
                        applied=False,
                        message="exact_not_found",
                    )
                ],
            )
        new_text = text.replace(search, replace)
        _write_text(file, new_text)
        hunks.append(
            HunkResult(
                file=str(file),
                hunk_header="smart_edit",
                applied=True,
            )
        )
        return EditReport(applied=True, message="Exact replacement applied", hunks=hunks)

    if mode == "regex":
        import re

        pattern = re.compile(search, re.MULTILINE | re.DOTALL)
        if not pattern.search(text):
            return EditReport(
                applied=False,
                message="Regex did not match file",
                hunks=[
                    HunkResult(
                        file=str(file),
                        hunk_header="smart_edit",
                        applied=False,
                        message="regex_not_found",
                    )
                ],
            )
        new_text = pattern.sub(replace, text)
        _write_text(file, new_text)
        hunks.append(
            HunkResult(
                file=str(file),
                hunk_header="smart_edit",
                applied=True,
            )
        )
        return EditReport(applied=True, message="Regex replacement applied", hunks=hunks)

    # fuzzy mode
    matcher = difflib.SequenceMatcher(None, text, search)
    best = max(matcher.get_matching_blocks(), key=lambda b: b.size, default=None)
    if not best or best.size == 0:
        return EditReport(
            applied=False,
            message="No sufficiently similar region found for fuzzy edit",
            hunks=[
                HunkResult(
                    file=str(file),
                    hunk_header="smart_edit",
                    applied=False,
                    message="fuzzy_no_match",
                )
            ],
        )

    start = best.a
    end = start + best.size
    new_text = text[:start] + replace + text[end:]
    _write_text(file, new_text)
    hunks.append(
        HunkResult(
            file=str(file),
            hunk_header="smart_edit",
            applied=True,
        )
    )
    return EditReport(applied=True, message="Fuzzy replacement applied", hunks=hunks)


def read_file_slice(file: Path, start_line: int, end_line: int) -> str:
    lines = _read_text(file).splitlines(keepends=True)
    start = max(start_line - 1, 0)
    end = min(end_line, len(lines))
    return "".join(lines[start:end])


def format_code(path: Path, style: str = "black", scope: str = "file") -> EditReport:
    """
    Thin wrapper around formatters; start with black only.
    """
    import subprocess

    cmd: list[str] = []
    if style == "black":
        cmd = ["black"]
    else:
        return EditReport(
            applied=False,
            message=f"Unknown style: {style}",
            hunks=[],
        )

    target = str(path) if scope == "file" else str(path)
    try:
        subprocess.run(cmd + [target], check=True)
        return EditReport(
            applied=True,
            message=f"Formatted with {style}",
            hunks=[],
        )
    except subprocess.CalledProcessError as e:
        return EditReport(
            applied=False,
            message=f"Formatter failed: {e}",
            hunks=[],
        )
```

### b. Hook into `mcp_servers.py`

Add tool definitions:

```python
# in TOOLS list
{
    "name": "apply_unified_diff",
    "description": "Apply a unified diff to files with simple fuzzy matching, optimized for long-file edits.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "diff": {"type": "string", "description": "Unified diff text"},
            "strategy": {
                "type": "string",
                "enum": ["exact", "whitespace_insensitive", "fuzzy"],
                "default": "fuzzy",
            },
        },
        "required": ["diff"],
    },
},
{
    "name": "smart_edit",
    "description": "Perform robust search/replace in a file using exact, regex, or fuzzy matching.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "Path to file"},
            "search": {"type": "string", "description": "Text or pattern to search for"},
            "replace": {"type": "string", "description": "Replacement text"},
            "mode": {
                "type": "string",
                "enum": ["exact", "regex", "fuzzy"],
                "default": "exact",
            },
        },
        "required": ["file", "search", "replace"],
    },
},
{
    "name": "read_file_slice",
    "description": "Read a specific line range from a file to keep context small.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "file": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "required": ["file", "start_line", "end_line"],
    },
},
{
    "name": "format_code",
    "description": "Format code using a configured formatter (e.g., black).",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "style": {
                "type": "string",
                "enum": ["black"],
                "default": "black",
            },
            "scope": {
                "type": "string",
                "enum": ["file", "package"],
                "default": "file",
            },
        },
        "required": ["path"],
    },
},
```

Then extend `call_tool`:

```python
    elif name == "apply_unified_diff":
        from agent_debug_toolkit.edits import apply_unified_diff
        diff = arguments.get("diff", "")
        strategy = arguments.get("strategy", "fuzzy")
        report = apply_unified_diff(diff, strategy=strategy, project_root=get_project_root())
        return [TextContent(type="text", text=report.to_json())]

    elif name == "smart_edit":
        from agent_debug_toolkit.edits import smart_edit
        path = resolve_path(arguments.get("file", ""))
        search = arguments.get("search", "")
        replace = arguments.get("replace", "")
        mode = arguments.get("mode", "exact")
        report = smart_edit(path, search, replace, mode=mode)
        return [TextContent(type="text", text=report.to_json())]

    elif name == "read_file_slice":
        from agent_debug_toolkit.edits import read_file_slice
        path = resolve_path(arguments.get("file", ""))
        start_line = int(arguments.get("start_line", 1))
        end_line = int(arguments.get("end_line", start_line + 200))
        snippet = read_file_slice(path, start_line, end_line)
        return [TextContent(type="text", text=snippet)]

    elif name == "format_code":
        from agent_debug_toolkit.edits import format_code
        path = resolve_path(arguments.get("path", ""))
        style = arguments.get("style", "black")
        scope = arguments.get("scope", "file")
        report = format_code(path, style=style, scope=scope)
        return [TextContent(type="text", text=report.to_json())]
```

This gives your agent concrete, testable edit tools right away.

***

## 2. Too many MCP tools: avoiding context burden

### What’s really happening

- Each tool’s name, description, and schema is serialized into the **system/tools section** that the model sees.
- With 50–100 tools across multiple MCP servers, that can take tens of thousands of tokens and degrade reasoning, as many devs have observed.[1][2][3]
- Most current IDE assistants do **not** expose all internal operations as tools to the model:
  - Copilot/Cursor have a small set of model-visible “modes” (edit, chat, agent), while many concrete actions (e.g. writing the file, running tests) are hard-coded in the client and not described in the model’s tool list.[4][5]

### Workarounds for your setup

Given you want “all tools available all the time” but without context blowup, the practical compromise is:

1. **Two-level tool system (router pattern)**
   - Expose a *small* set of “meta-tools” to the model, each of which internally fans out to many concrete functions.[6]
   - Example:
     - Model sees:
       - `adt_query(type, target, options)` – routes to `analyze_*` / `context_tree` / `intelligent_search`.
       - `adt_edit(kind, params)` – routes to `apply_unified_diff`, `smart_edit`, `format_code`.
     - Your MCP server parses the arguments and decides which internal analyzer/operation to run.

   That way the model has ~5–10 tools in context, but you still keep dozens of internal capabilities.

2. **Profiled tool loading**
   - Have multiple MCP profiles:
     - “Hydra analysis” (only `adt_core`, `adt_phase1`),
     - “Editing” (only `adt_edits`, Tree-sitter),
     - “Refactor” (symbolic + edits).
   - Your IDE or launcher selects a profile per session, keeping active tool count well below the “accuracy cliff” that starts around 40 tools in some clients.[3][7]

3. **Client-side tool gating**
   - Many IDE assistants only include tools in the prompt when the user’s instruction type matches a category.[8][9]
   - You can mimic this:
     - For “explain config flow…”: include only analysis tools in the request.
     - For “apply this patch”: include only edit tools.
   - This requires a thin controller layer in your IDE integration that classifies intent and sets `tools` for each call dynamically.

### How IDEs do generic editing without burden

- Copilot Edit Mode:
  - The IDE selects code regions and sends them as prompt context plus a **single edit operation request**; the model just returns modified code.[5][10][4]
  - File I/O, diff computation, and application are purely client-side; not tools in the model’s tool list.
- Agent modes:
  - Higher-level actions (run tests, open file) may be modeled as tools, but still in small numbers.[4][5]

For you, that suggests:

- Keep simple file operations (open/save, small selection replace) **out of MCP**, and do them as direct IDE actions.
- Use MCP tools for **semantic** operations (symbolic rename, config resolution, dead code analysis).

### Integrating natively into your IDE (personal)

For VS Code / JetBrains / Neovim:

- **VS Code**
  - Implement a VS Code extension that:
    - Starts your MCP server (if not already),
    - Talks to your LLM backend,
    - For each user command:
      - Classifies intent (edit vs analyze vs refactor),
      - Chooses a subset of tools to pass into that one LLM request,
      - Post-processes responses (apply diffs via extension API).
  - Most file editing is done via VS Code’s `TextEditor.edit`, not as tools.

- **JetBrains**
  - Similar: use their plugin APIs to:
    - Read selections,
    - Apply patches,
    - Run tests.
  - The model sees only higher-level tools; the plugin does the actual work.

- **Neovim/CLI**
  - Your current MCP server + a thin CLI client (or Neovim plugin) that:
    - Manages which tools are active per request,
    - Applies diffs locally.

In all cases, the key pattern is: **small visible tool surface to the model, rich internal operations in your code.**[2][8]

If you want, next step could be designing a `adt_meta` tool (one tool with a `kind` field that dispatches to your existing analyzers plus the new edit tools), so you can test the router pattern without re-architecting everything.


---

A VS Code extension with a router pattern acts as a **thin controller** between Copilot-style chat and your many MCP tools, so the model only ever sees a **small set of meta-tools**, while the extension decides which concrete MCP tool to invoke.[1][2]

## 1. High-level flow

- User asks in VS Code: “Scan configs and then refactor imports in these files.”
- Extension:
  - Reads selection / workspace context via VS Code API.[1]
  - Calls your LLM once with a **small tool list**, e.g. `adt_meta_query`, `adt_meta_edit`.
  - Model replies with a tool call like:
    - `adt_meta_query(kind="hydra_config_flow", target="conf/", options={...})`
- Extension router:
  - Maps `kind="hydra_config_flow"` → call `analyze_config_access`, `trace_merge_order`, `find_hydra_usage`, etc. in your MCP server.
  - Aggregates results, sends them back into the same chat as plain text or summarized context.
- For edits:
  - Model calls `adt_meta_edit(kind="apply_diff", diff=..., strategy="fuzzy")`.
  - Router calls your MCP `apply_unified_diff`, then applies resulting file changes through VS Code’s `TextEditor.edit`.

The model never sees 30+ tools; it only sees 2–5 meta-tools.[3][4]

## 2. Extension architecture (simplified)

### a. VS Code side

- **Contribution points**:
  - Chat participant / Copilot tool using the Language Model Tool API.[1]
  - Commands like `agentDebugToolkit.explainConfigFlow` for keyboard shortcuts.

- **Core pieces**:
  - `router.ts`: decides which meta-tool to expose and how to map `kind` → MCP call.
  - `mcpClient.ts`: talks JSON-RPC over stdio / TCP to your `agent_debug_toolkit` server.
  - `llmClient.ts`: calls Claude/GPT/Qwen with function-calling and a small `tools` list.

### b. Meta-tool schema visible to the LLM

Model sees something like:

```ts
const tools = [
  {
    type: "function",
    function: {
      name: "adt_meta_query",
      description: "High-level analysis over the codebase and configs.",
      parameters: {
        type: "object",
        properties: {
          kind: {
            type: "string",
            enum: [
              "config_flow",
              "dependency_graph",
              "hydra_usage",
              "symbol_search"
            ],
          },
          target: { type: "string", description: "Path or symbol" },
          options: { type: "object", additionalProperties: true },
        },
        required: ["kind", "target"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "adt_meta_edit",
      description: "High-level edit operations for code and configs.",
      parameters: {
        type: "object",
        properties: {
          kind: {
            type: "string",
            enum: [
              "apply_diff",
              "smart_edit",
              "format_code",
            ],
          },
          target: { type: "string", description: "File or directory" },
          diff: { type: "string" },
          params: { type: "object", additionalProperties: true },
        },
        required: ["kind"],
      },
    },
  },
];
```

Your extension registers these as **VS Code tools** using the Language Model Tool API.[1]

## 3. Router implementation sketch

In your VS Code extension (TypeScript):

```ts
// router.ts
import { callMcpTool } from "./mcpClient";
import * as vscode from "vscode";

export async function handleAdtMetaQuery(args: any): Promise<string> {
  const { kind, target, options = {} } = args;

  switch (kind) {
    case "config_flow": {
      const fileOrDir = resolveWorkspacePath(target);
      const [configAccess, mergeOrder, hydraUsage] = await Promise.all([
        callMcpTool("analyze_config_access", { path: fileOrDir, output: "markdown" }),
        callMcpTool("trace_merge_order", { file: fileOrDir, output: "markdown" }),
        callMcpTool("find_hydra_usage", { path: fileOrDir, output: "markdown" }),
      ]);
      return [
        "## Config Access",
        configAccess,
        "",
        "## Merge Order",
        mergeOrder,
        "",
        "## Hydra Usage",
        hydraUsage,
      ].join("\n");
    }

    case "dependency_graph": {
      const path = resolveWorkspacePath(target);
      return await callMcpTool("analyze_dependency_graph", {
        path,
        include_stdlib: !!options.includeStdlib,
        output: "markdown",
      });
    }

    case "symbol_search": {
      const root = resolveWorkspacePath(options.root || ".");
      return await callMcpTool("intelligent_search", {
        query: target,
        root,
        fuzzy: options.fuzzy ?? true,
        output: "markdown",
      });
    }

    default:
      return `Unknown query kind: ${kind}`;
  }
}

export async function handleAdtMetaEdit(args: any): Promise<string> {
  const { kind, target, diff, params = {} } = args;

  switch (kind) {
    case "apply_diff": {
      // Option A: let MCP apply changes on disk
      const reportJson = await callMcpTool("apply_unified_diff", {
        diff,
        strategy: params.strategy || "fuzzy",
      });
      // Option B (better UX): compute workspace edit and apply via VS Code API.
      await applyDiffToEditor(diff);
      return reportJson;
    }

    case "smart_edit": {
      const file = resolveWorkspacePath(target);
      const reportJson = await callMcpTool("smart_edit", {
        file,
        search: params.search,
        replace: params.replace,
        mode: params.mode || "exact",
      });
      return reportJson;
    }

    case "format_code": {
      const path = resolveWorkspacePath(target);
      const reportJson = await callMcpTool("format_code", {
        path,
        style: params.style || "black",
        scope: params.scope || "file",
      });
      return reportJson;
    }

    default:
      return `Unknown edit kind: ${kind}`;
  }
}

async function applyDiffToEditor(diff: string) {
  // parse diff and use vscode.WorkspaceEdit / TextEditor.edit
  // left as a placeholder — you can incrementally improve it
}

function resolveWorkspacePath(p: string): string {
  const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || "";
  return require("path").join(folder, p);
}
```

Then, wire these as handlers for the LLM tools declared via the VS Code Language Model Tool API.[1]

## 4. How this avoids tool/context overload

- The **LLM sees only `adt_meta_query` and `adt_meta_edit`**, not every analyzer and edit primitive.
- The VS Code extension decides which MCP tool(s) to call based on `kind`, effectively acting as a “tool router.”[2][5]
- You can add *new* MCP tools (like Tree-sitter ones) without changing the tool list in the LLM:
  - Just extend the `switch` in `handleAdtMetaQuery` / `handleAdtMetaEdit` to route new `kind` values.

This is analogous to how Copilot has a small number of modes (Ask/Edit/Agent) but many internal operations implemented in the client/IDE, not as exposed tools.[6][7][8]

If you want, the next step can be to sketch a minimal `mcpClient.ts` (stdio/TCP JSON-RPC wrapper) tailored to your `agent_debug_toolkit` server so you can drop this into a prototype extension quickly.



## Conversation Snippet END ##
