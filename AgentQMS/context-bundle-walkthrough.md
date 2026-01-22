# Context Bundle Optimization Walkthrough

## Overview
We have successfully optimized the AgentQMS context management system to solve the "Context Clutter" problem. 
- **Goal**: Reduce excessive token usage (e.g., 28k tokens for debugging) and eliminate forced reading of documentation.
- **Result**: Implemented granular `mode` logic (Full, Structure, Reference) across all 15 context bundles.

## Key Changes
### 1. New Context Modes
- **Full** (`mode: full`): Loads entire file content. Reserved for valid configuration files and active editing targets.
- **Structure** (`mode: structure`): Loads only class/function signatures (outlines). Used for code dependencies and tools. (~80% saving)
- **Reference** (`mode: reference`): Loads only the file description/path. Content is **opt-in**. Used for all documentation, standards, and directories. (~99% saving)

### 2. Audit Results
We audited and optimized all 15 bundles. Common transformations included:
- **Docs/Standards**: Shifted to `reference` (e.g., `hydra-v5-rules.yaml`, `tool-catalog.yaml`).
- **Tools/Code**: Shifted to `structure` (e.g., `ocr/features/detection/*.py`, `etk.py`).
- **Implementation**: Pruned internal tool logic (e.g., removed [cli.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/cli.py), [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/mcp_server.py) from context).

## Verification Data
| Bundle Name | Status | Key Optimization |
| :--- | :--- | :--- |
| **ocr-debugging** | ✅ Optimized | **28k -> ~2.6k** tokens |
| **ast-debugging** | ✅ Optimized | Reference mode for usage guides |
| **pipeline-dev** | ✅ Optimized | Structure mode for pipeline orchestration |
| **experiment** | ✅ Optimized | Configs kept Full, Manager code Structure |
| **docs/compliance** | ✅ Optimized | migrated 100% to Reference/Structure |

## User Guide
To inspect a bundle's new lightweight footprint:
```bash
python AgentQMS/tools/core/context_bundle.py --task "debug ocr"
```
You will see output indicating the mode:
- `agent-debug-toolkit/README.md [reference]`
- `ocr/features/detection/model.py [structure]`
