---
ads_version: '2.0'
id: 'FW-030'
type: 'rule_set'
tier: 2
priority: 'high'
updated: '2026-02-03'
spec_version: '1.0.0'
---

# Python Core Standards

## Specification

```yaml
agent: all
type_annotation_rules:
  required_locations:
  - All public function signatures
  - All class attributes
  - All module-level constants
  prohibited_patterns:
  - Using Any without justification
  - Missing return type annotations
  - Untyped **kwargs in public APIs
  recommended_patterns:
    type_checking_guard:
      purpose: Avoid import overhead for type hints
      example: "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\n\
        \    from ocr.models import DBHead\n\ndef process(x: \"torch.Tensor\") ->\
        \ \"DBHead\":\n    import torch\n    ...\n"
    optional_types:
      pattern: X | None instead of Optional[X]
      example: 'def process(data: np.ndarray | None = None)'
    collection_types:
      pattern: list[X] instead of List[X]
      example: 'def process(items: list[str]) -> dict[str, int]'
  enforcement:
    tool: mypy --strict
    config: pyproject.toml
error_handling_rules:
  exception_hierarchy:
    base: Use domain-specific exception classes
    locations:
    - ocr/core/exceptions.py if exists, else standard Python exceptions
    - Inherit from built-in exceptions (ValueError, TypeError)
  required_patterns:
  - Always include context in error messages
  - Use 'from exc' for exception chaining
  - Log before re-raising in boundaries
  prohibited_patterns:
  - Bare except clauses
  - Catching Exception without re-raise
  - Silent failures (empty except blocks)
  - String-only error messages without context
  patterns:
    prohibited:
      example: "try:\n    process(data)\nexcept:\n    pass\n"
    required:
      example: "try:\n    process(data)\nexcept ValidationError as exc:\n    logger.error(\"\
        Processing failed for %s: %s\", filename, exc)\n    raise ProcessingError(f\"\
        Failed to process {filename}\") from exc\n"
  retry_logic:
    when: Network calls, transient failures only
    library: tenacity
    max_attempts: 3
    backoff: exponential
logging_rules:
  levels:
    DEBUG: Verbose diagnostic info (disabled in prod)
    INFO: Key operations, state changes
    WARNING: Recoverable issues, degraded performance
    ERROR: Failures requiring attention
    CRITICAL: System-wide failures
  required_patterns:
  - 'Use module-level logger: logger = logging.getLogger(__name__)'
  - 'Use %-formatting for lazy evaluation: logger.info(''x=%s'', x)'
  - Include relevant context (filename, batch_idx, etc.)
  prohibited_patterns:
  - print() for logging
  - f-strings in log calls (eager evaluation)
  - Logging secrets, passwords, API keys
  - Logging entire tensors or large arrays
  fallback_policy:
    rule: No silent fallbacks in observability paths (logging, metrics, visualization)
    required_patterns:
    - Fallbacks must emit WARNING with context and be gated by an explicit config flag
    prohibited_patterns:
    - Placeholder outputs like "?" or "(no pred)" without a warning
    - Silent try/except that suppresses logging failures
  patterns:
    prohibited:
      example: 'print(f"Processing {filename}")\nlogger.info(f"Result: {large_tensor}")\n'
    required:
      example: 'logger = logging.getLogger(__name__)\nlogger.info("Processing file:
        %s", filename)\nlogger.debug("Tensor shape: %s", tensor.shape)\n'
path_resolution_rules:
  objective: Ensure robust path resolution independent of file location
  required_patterns:
  - Use AgentQMS.tools.utils.paths.get_project_root() for root-relative paths
  - Use simple / operator for path joining
  prohibited_patterns:
  - Chained .parent calls (more than 2) to find root
  - Manual sys.path.append() to fix imports
  - Hardcoded relative paths assuming directory depth
  patterns:
    prohibited:
      description: Fragile parent traversal
      example: '# BAD: Breaks if file is moved\nROOT = Path(__file__).resolve().parent.parent.parent\n'
    required:
      description: Robust root import
      example: '# GOOD: Works anywhere\nfrom AgentQMS.tools.utils.paths import get_project_root\ndata_file
        = get_project_root() / "data" / "file.txt"\n'
  enforcement:
    detection: pre-commit hook (validate-path-usage)
    command: pre-commit run validate-path-usage --all-files
deprecation_rules:
  lifecycle_phases:
    phase_1_soft:
      action: Add @deprecated decorator or DeprecationWarning
      timeline: Immediate upon decision
      behavior: Code works, logs warning once per session
    phase_2_hard:
      action: Make warning loud (stack trace) or soft-fail
      timeline: Next minor version (e.g. v1.1 -> v1.2)
      behavior: Code works but spams warnings or requires flag to enable
    phase_3_removal:
      action: Delete code
      timeline: Next major version (e.g. v1.x -> v2.0)
      behavior: ImportError or AttributeError
  required_patterns:
    python_code:
      description: Use the warnings library
      example: "import warnings\n\ndef old_function():\n    warnings.warn(\n     \
        \   \"old_function is deprecated and will be removed in v2.0. Use new_function\
        \ instead.\",\n        DeprecationWarning,\n        stacklevel=2\n    )\n\
        \    return new_function()\n"
multiprocessing_rules:
  objective: Ensure stability and CUDA compatibility across process boundaries
  critical:
    start_method:
      value: spawn
      reason: fork is incompatible with initialized CUDA contexts, causing SegFaults
      implementation: Set at script entry point before any imports that may initialize CUDA
  constraints:
    picklability:
      rule: All objects passed to workers (Configs, Datasets) must be picklable
      action: Convert Hydra DictConfig to primitive dicts before passing to DataLoader
    lazy_loading:
      rule: File handles and Database connections (LMDB) must be initialized lazily in the worker
      reason: Open file handles/transactions cannot be pickled
  patterns:
    required_training_script:
      description: Set spawn method at entry point
      example: "import torch.multiprocessing as mp\ntry:\n    mp.set_start_method('spawn',\
        \ force=True)\nexcept RuntimeError:\n    pass  # Already set\n\nimport\
        \ torch\nfrom ocr.pipelines.orchestrator import OCRProjectOrchestrator\n"
    required_dataset:
      description: Lazy initialization of file handles
      example: "class Dataset:\n    def __init__(self):\n        self.env = None\n\
        \    def __getitem__(self, idx):\n        if self.env is None:\n            self.env\
        \ = lmdb.open(...)\n"
    dataloader_config:
      description: DataLoader settings compatible with spawn
      example: "DataLoader(\n    dataset,\n    num_workers=2,  # Safe with spawn\n\
        \    pin_memory=True,  # Safe with spawn\n    persistent_workers=True\n)\n"
