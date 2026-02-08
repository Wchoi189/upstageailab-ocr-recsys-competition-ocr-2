# implementation_plan - Context Bundle Engine 2.0

> **Goal**: Refactor [AgentQMS/tools/core/context_bundle.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/core/context_bundle.py) from a script into a robust, high-performance **Context Engine** capable of sub-second resolution and strict token budgeting.

## 1. Architecture Refactor: `ContextEngine` Class
Move away from stateless global functions to a stateful `ContextEngine` class to handle caching and configuration.

```python
class ContextEngine:
    def __init__(self, config: EngineConfig):
        self.cache = ResourceCache()  # Stats, YAML, Keywords
        self.metrics = PerformanceMetrics()
    
    def get_bundle(self, request: ContextRequest) -> ContextResult:
        ...
```

## 2. Performance Optimizations (P0)
### 2.1 Caching Layer
- **`stat_cache`**: Cache `os.stat` results during a single request lifecycle to avoid redundant filesystem calls for [is_fresh](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/core/context_bundle.py#156-179) and sorting.
- **`keyword_cache`**: Load `context-keywords.yaml` once and cache until mtime changes.
- **`yaml_cache`**: Cache parsed bundle YAMLs by (path, mtime).

### 2.2 Parallel I/O
- **Parallel Token Estimation**: Use `concurrent.futures.ThreadPoolExecutor` for [estimate_token_count](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/core/context_bundle.py#349-403). Reading 20+ files serially is the main bottleneck.
- **Lazy Globbing**: Use `iglob` and `heapq` to processing large file trees without materializing full lists.

### 2.3 Fast Loading
- Use `yaml.CSafeLoader` if available (10x faster parsing).

## 3. Advanced Features (P1)
### 3.1 Token Budgeting
Implement strict budget controls to prevent context overflow.
- **Input**: `max_tokens` (e.g., 8000).
- **Strategy**:
    1. Always include `critical` priority files.
    2. Include `high` priority files until soft limit.
    3. If over budget, drop [low](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/core/workflow_detector.py#72-133) priority, then `medium`.
    4. Truncate `structure` mode files if necessary (retain signatures, drop docstrings).

### 3.2 Rich Metadata ("Why Included")
Return detailed objects explaining the decision chain:
```json
{
  "path": "src/utils.py",
  "reason": "explicit_rule (tier1)",
  "token_cost": 450,
  "status": "included",
  "mode": "structure"
}
```

### 3.3 Enhanced Task Detection
- Replace substring matching with **Regex/Word-Boundary** matching.
- Implement **Weighted Scoring** (e.g., "debug" = 5 points, "fix" = 1 point).

## 4. Safety & Ergonomics (P2)
- **Schema Validation**: Validate bundle YAMLs against a Pydantic model or JSON Schema on load.
- **Path Hardening**: Enforce `PROJECT_ROOT` sandbox (prevent `../../`).
- **CLI Enhancements**:
    - `--json`: Machine-readable output.
    - `--explain`: Print inclusion reasons.
    - `--budget`: Set token limit override.

## 5. Execution Strategy
This refactor will be performed in the next session.
1. **Setup**: Define `ContextEngine` skeleton and Pydantic models.
2. **Core**: Implement Caching and Parallel I/O.
3. **Budgeting**: Implement the `ResourceAllocator` logic.
4. **Migration**: Update main entry point to use `ContextEngine`.
