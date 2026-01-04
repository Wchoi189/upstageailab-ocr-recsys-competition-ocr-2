# AST Debugging Toolkit - Walkthrough

## Summary

Built a standalone Python package `agent-debug-toolkit` that uses AST analysis to help AI agents debug Hydra/OmegaConf configuration issues.

## Package Location

[agent-debug-toolkit/](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/)

## What Was Built

### Analyzers (4)

| Analyzer | Purpose | File |
|----------|---------|------|
| [ConfigAccessAnalyzer](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/config_access.py#34-289) | Detects `cfg.X`, `self.cfg.X`, `cfg['key']`, [getattr(cfg, 'X')](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/tests/test_config_access.py#42-49) | [config_access.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/config_access.py) |
| [MergeOrderTracker](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/merge_order.py#48-334) | Tracks `OmegaConf.merge()` precedence | [merge_order.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/merge_order.py) |
| [HydraUsageAnalyzer](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/hydra_usage.py#31-314) | Finds `@hydra.main`, [instantiate()](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/hydra_usage.py#201-209) | [hydra_usage.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/hydra_usage.py) |
| [ComponentInstantiationTracker](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/instantiation.py#44-329) | Tracks `get_*_by_cfg()` patterns | [instantiation.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/analyzers/instantiation.py) |

### CLI Commands

```bash
adt analyze-config <path>     # Config access patterns
adt trace-merges <file>       # Merge precedence analysis
adt find-hydra <path>         # Hydra framework usage
adt find-instantiations <path> # Component factory calls  
adt full-analysis <path>      # Run all analyzers
```

## Test Results

**20/20 tests passed** ✅

## Real-World Validation: BUG_003

Running `adt trace-merges` on [ocr/models/architecture.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/models/architecture.py) revealed:

```
| Priority | Line | Operation | Winner on Conflict |
|----------|------|-----------|-------------------|
| P1 | 107 | create    | {}                 |
| P2 | 113 | merge     | arch_overrides     |
| P3 | 128 | merge     | direct_overrides   |  ← architecture decoder
| P4 | 140 | merge     | top_level_overrides|  ← LEGACY FPNDecoder WINS HERE
| P5 | 144 | merge     | cfg.component_overrides |
```

> **Key Insight**: The `top_level_overrides` (P4) containing the legacy `FPNDecoder` has *higher* priority than the architecture's `direct_overrides` (P3) containing `PARSeqDecoder`.

## Installation

```bash
cd agent-debug-toolkit && uv pip install -e ".[all]"
```

## Next Steps

- [ ] MCP integration (Phase 2)
- [ ] Apply insights to fix BUG_003
