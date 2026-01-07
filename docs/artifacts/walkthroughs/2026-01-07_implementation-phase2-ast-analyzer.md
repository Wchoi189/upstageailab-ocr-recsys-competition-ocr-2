# Phase 2 AST Analyzer Implementation

Implement 3 medium-risk analyzers: DuplicationDetector, TypeInferenceAnalyzer, and SecurityScanner. Follow existing [BaseAnalyzer](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/base.py#135-341) patterns from Phase 1.

## Proposed Changes

### Component: DuplicationDetector

#### [NEW] [duplication_detector.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/duplication_detector.py)

Finds copy-paste code patterns via AST hash comparison:
- Normalize AST by replacing variable names with placeholders
- Hash normalized function/code block subtrees
- Group blocks by hash to find duplicates
- Report with similarity score and suggested refactoring action

```python
class DuplicationDetector(BaseAnalyzer):
    name = "DuplicationDetector"
    # Config: min_lines=5, similarity_threshold=0.85
```

---

### Component: TypeInferenceAnalyzer

#### [NEW] [type_inference.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/type_inference.py)

Tracks variable and function types from assignments and annotations:
- Parse type annotations
- Infer types from literal assignments (int, str, list, dict)
- Track function return types from return statements
- Detect type conflicts (same variable assigned different types)

```python
class TypeInferenceAnalyzer(BaseAnalyzer):
    name = "TypeInferenceAnalyzer"
    # Single-file scope only (no cross-file analysis)
```

---

### Component: SecurityScanner

#### [NEW] [security_scanner.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/security_scanner.py)

Detects unsafe patterns in config usage:
- **Hydra-specific**: `instantiate()` with user input, dynamic `_target_`, `eval(cfg.x)`
- **General**: hardcoded secrets (API keys), `subprocess(shell=True)` with config
- Severity levels: critical, high, medium, low, info

```python
class SecurityScanner(BaseAnalyzer):
    name = "SecurityScanner"
    severity_levels = ["critical", "high", "medium", "low", "info"]
```

---

### Component: CLI

#### [MODIFY] [cli.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/cli.py)

Add 3 new commands following existing pattern:
- `detect-duplicates` - `--min-lines INT`, `--threshold FLOAT`
- `infer-types` - `--show-conflicts`
- `security-scan` - `--severity LEVEL`

---

### Component: MCP Server

#### [MODIFY] [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py)

Add 3 new tools to TOOLS list:
- `detect_code_duplicates`
- `infer_types`
- `security_scan`

---

### Component: Exports

#### [MODIFY] [__init__.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/__init__.py)

Export: `DuplicationDetector`, `TypeInferenceAnalyzer`, `SecurityScanner`

---

## Tests

#### [NEW] [test_duplication_detector.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/tests/test_duplication_detector.py)

Test cases:
- Exact duplicate functions
- Functions with renamed variables (should still detect)
- Similar functions above/below threshold
- Empty files
- No duplicates case

#### [NEW] [test_type_inference.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/tests/test_type_inference.py)

Test cases:
- Literal assignments (int, str, list, dict)
- Type annotations
- Function return type inference
- Type conflicts detection
- Class method types

#### [NEW] [test_security_scanner.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/tests/test_security_scanner.py)

Test cases:
- `eval(cfg.x)` detection (critical)
- Hardcoded API keys detection
- `subprocess(shell=True)` with config
- Clean code (no false positives)
- Severity filtering

---

## Verification Plan

### Automated Tests

Run test suite with existing Phase 1 tests + new Phase 2 tests:

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit
uv run pytest tests/ -v
```

Expected: All tests pass (currently 34 Phase 1 + ~30 new Phase 2 tests).

### CLI Integration Verification

```bash
# DuplicationDetector
uv run adt detect-duplicates src/agent_debug_toolkit/ --format json

# TypeInferenceAnalyzer
uv run adt infer-types src/agent_debug_toolkit/analyzers/base.py --show-conflicts

# SecurityScanner
uv run adt security-scan src/agent_debug_toolkit/ --severity critical
```

### MCP Server Verification

Start MCP server and verify tools are listed:
```bash
uv run python -m agent_debug_toolkit.mcp_server
```
