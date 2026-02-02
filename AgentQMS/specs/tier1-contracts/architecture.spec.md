# System Architecture Specification

**Tier**: 1 (Contracts)
**Scope**: High-level Architecture & AI Principles.

## 1. AI-Native Architecture (Registry-Driven)
The architecture is defined by the **Registry** (`AgentQMS/governance/specifications/`).
*   **Single Source of Truth**: One file = One function.
*   **Mechanized Graph**: Dependencies are explict.

### Classification Shortcuts
| Class | Meaning | Example |
| :--- | :--- | :--- |
| **Framework** | HOW it works | Protocols, Contracts, Infra |
| **Agents** | WHO it is | Identity, Persona |
| **Workflows** | DO steps | Runbooks, Scripts |

## 2. Core Components

### Applications
| App | Status | Location | Type |
| :--- | :--- | :--- | :--- |
| **Playground** | Active | `apps/playground-console/` | NextJS |
| **Inference Console** | Active | `apps/ocr-inference-console/` | Vite React |
| **AgentQMS Dashboard** | Active | `apps/agentqms-dashboard/` | Containerized |
| **Backend API** | Active | `apps/ocr-inference-console/backend/` | FastAPI (Port 8002) |

### Inference Engine
*   **Location**: `ui/utils/inference/engine.py`
*   **Consumers**: `ocr_bridge`, `playground_api`, `legacy_streamlit`.

### Config System (Hydra)
*   **Architecture**: Domain-First.
*   **Roots**: `configs/`, `configs/domain/`, `configs/model/`.
*   **Domain Switching**: Enabled (`python runners/train.py domain=X`).

## 3. Required Actions
*   **Sync Registry**: `aqms registry sync`
*   **Gen Graph**: `python AgentQMS/tools/generate_mechanized_graph.py`
