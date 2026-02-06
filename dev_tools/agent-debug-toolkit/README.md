# Agent Debug Toolkit

AST-based debugging toolkit for AI agents to analyze Hydra/OmegaConf configuration patterns.

## Overview

This toolkit helps AI agents understand and debug complex configuration systems by using Python's AST module to statically analyze code. It's particularly useful for:

- **Config Precedence Issues**: Trace `OmegaConf.merge()` call order to debug override conflicts
- **Hydra Pattern Discovery**: Find `@hydra.main` decorators and `instantiate()` calls
- **Component Tracking**: Trace factory patterns like `get_decoder_by_cfg()`
- **Configuration Flow**: Understand how config values flow through the codebase

## Installation

```bash
# Install with CLI support
uv pip install -e agent-debug-toolkit[cli]

# Install with MCP server support
uv pip install -e agent-debug-toolkit[mcp]

# Install all extras (CLI + MCP + dev)
uv pip install -e agent-debug-toolkit[all]
```

## CLI Usage

```bash
# Analyze config access patterns
adt analyze-config ocr/models/architecture.py --component decoder

# Trace OmegaConf.merge precedence order
adt trace-merges ocr/models/architecture.py --output markdown

# Find Hydra usage patterns
adt find-hydra ocr/

# Find component instantiation sites
adt find-instantiations ocr/models/ --component decoder

# Run all analyzers for comprehensive analysis
adt full-analysis ocr/models/architecture.py
```

## MCP Server

The toolkit includes an MCP (Model Context Protocol) server that exposes analysis tools to AI agents.

### Starting the MCP Server

```bash
# Using the launch script
./run_mcp.sh

# Or directly via uv
uv run python -m agent_debug_toolkit.mcp_server
```

### Available MCP Tools

| Tool                            | Description                                                |
| ------------------------------- | ---------------------------------------------------------- |
| `analyze_config_access`         | Find `cfg.X`, `self.cfg.X`, `config['key']` patterns       |
| `trace_merge_order`             | Trace `OmegaConf.merge()` call precedence                  |
| `find_hydra_usage`              | Detect `@hydra.main`, `instantiate()`, `_target_` patterns |
| `find_component_instantiations` | Track `get_*_by_cfg()` and registry patterns               |
| `explain_config_flow`           | Generate high-level config flow summary                    |

## Python API

```python
from agent_debug_toolkit.analyzers.config_access import ConfigAccessAnalyzer
from agent_debug_toolkit.analyzers.merge_order import MergeOrderTracker
from agent_debug_toolkit.analyzers.hydra_usage import HydraUsageAnalyzer
from agent_debug_toolkit.analyzers.instantiation import ComponentInstantiationTracker

# Analyze a file for config access patterns
analyzer = ConfigAccessAnalyzer()
report = analyzer.analyze_file("path/to/file.py")

# Get results
for access in report.results:
    print(f"{access.file}:{access.line} - {access.pattern}")

# Export as JSON or Markdown
print(report.to_json())
print(report.to_markdown())
```

## Analyzers

### ConfigAccessAnalyzer

Detects configuration access patterns:
- `cfg.encoder`, `self.cfg.model`
- `cfg['decoder']['name']`
- `getattr(cfg, 'encoder', None)`
- `hasattr(cfg, 'decoder')`

### MergeOrderTracker

Tracks OmegaConf merge operations:
- `OmegaConf.merge(a, b, c)` - identifies precedence order
- `OmegaConf.create({})` - tracks config creation
- Explains which config "wins" in merge conflicts

### HydraUsageAnalyzer

Finds Hydra framework patterns:
- `@hydra.main(config_path="...")` decorators
- `hydra.utils.instantiate(cfg.model)` calls
- `_target_` and `_recursive_` config patterns

### ComponentInstantiationTracker

Tracks component factory patterns:
- `get_encoder_by_cfg(cfg.encoder)`
- `registry.create_architecture_components(...)`
- Direct class instantiation with config

## Use Cases

### Debugging Config Precedence (BUG_003)

```bash
# Trace all merge operations to find why legacy defaults override architecture
adt trace-merges ocr/models/architecture.py --explain

# Output shows:
# - Merge #1: base_config (line 120)
# - Merge #2: architecture_config (line 135)
# - Merge #3: component_overrides (line 145) <- WINNER
```

### Finding Component Sources

```bash
# Find where the decoder component is created
adt find-instantiations ocr/ --component decoder --output markdown
```

## License

MIT
