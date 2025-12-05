---
title: "AI Agent System – Single Source of Truth"
date: "2025-12-04 00:00 (KST)"
type: "guide"
category: "ai_agent"
status: "active"
version: "1.0"
tags: ["ai_agent", "rules", "operations"]
---

AI Agent System – Single Source of Truth
=======================================

Status: active

Read this file only. Agents do not need tutorials.

Core Rules
----------
- Always use automation tools; never create files manually.
- No loose docs in project root; no ALL CAPS filenames (except README.md, CHANGELOG.md).
- Use kebab/underscore naming with timestamp prefix.
- Test in browser and check logs; do not rely on unit tests alone.

Artifact Creation (use one of these)
------------------------------------
```bash
# From agent interface (recommended)
cd AgentQMS/interface/
make create-plan NAME=my-plan TITLE="My Plan"
make create-assessment NAME=my-assessment TITLE="My Assessment"
make create-design NAME=my-design TITLE="My Design"
make create-bug-report NAME=my-bug TITLE="My Bug Report"
```

Types: implementation_plan, assessment, design, research, template, bug_report

**Implementation Plans**: Always use Blueprint Protocol Template (PROTO-GOV-003). The generator uses this template automatically. See `AgentQMS/knowledge/templates/blueprint_protocol_template.md` for structure.

Tool Discovery and Validation
-----------------------------
```bash
# Agent tools (run inside AgentQMS/interface/)
cd AgentQMS/interface/
make help
make discover
make status
make validate
make compliance
```

Documentation Organization
--------------------------
- Artifacts live under `docs/artifacts/` by type.
- Required frontmatter and naming: `YYYY-MM-DD_HHMM_{ARTIFACT_TYPE}_descriptive-name.md`.
- Frontmatter `date` must use the full timestamp format `YYYY-MM-DD HH:MM (KST)` for every commit.

**Valid Artifact Types:**

| Artifact Type | Format | Directory |
|--------------|--------|----------|
| `implementation_plan_` | `YYYY-MM-DD_HHMM_implementation_plan_name.md` | `docs/artifacts/implementation_plans/` |
| `assessment-` | `YYYY-MM-DD_HHMM_assessment-name.md` | `docs/artifacts/assessments/` |
| `audit-` | `YYYY-MM-DD_HHMM_audit-name.md` | `docs/artifacts/audits/` |
| `design-` | `YYYY-MM-DD_HHMM_design-name.md` | `docs/artifacts/design_documents/` |
| `research-` | `YYYY-MM-DD_HHMM_research-name.md` | `docs/artifacts/research/` |
| `template-` | `YYYY-MM-DD_HHMM_template-name.md` | `docs/artifacts/templates/` |
| `BUG_` | `YYYY-MM-DD_HHMM_BUG_name.md` | `docs/artifacts/bug_reports/` |
| `SESSION_` | `YYYY-MM-DD_HHMM_SESSION_name.md` | `docs/artifacts/completed_plans/completion_summaries/session_notes/` |

- Long-form guidance lives in `AgentQMS/knowledge/` (not for agents).
- Agent doc map: see `AgentQMS/knowledge/agent/` index (SST + quick references).
- Tracking domain: `AgentQMS/knowledge/agent/tracking_cli.md` and related references.
- Automation domain: `AgentQMS/knowledge/agent/tool_catalog.md` and automation references.
- Coding and development protocols: `AgentQMS/knowledge/protocols/development/`.
- **OCR Experiment domain**: `AgentQMS/knowledge/agent/ocr_experiment_agent.md` – specialized instructions for OCR experiment workflows, VLM tools, and experiment-tracker integration.
- Escalation: if any knowledge domain exceeds a manageable size, adopt capability-based indexing using `.agentqms/state/architecture.yaml` and metadata (`capabilities`, `audience`, `visibility`).
- This file is the sole authoritative agent instruction.

Documentation Style for AI Agents
----------------------------------
When creating/updating AI-oriented instructions:
- ✅ Provide concise hints/reminders (1-3 lines max per concept)
- ✅ Show minimal code examples (correct/incorrect patterns)
- ✅ Use bullet points, not paragraphs
- ❌ No tutorials or comprehensive explanations
- ❌ No multi-paragraph descriptions
- ❌ No redundant context

Agents don't need tutorials - minimal hints are sufficient. Verbose docs consume context and reduce effectiveness.

Path Management
---------------
For AgentQMS tools, use `PYTHONPATH=.` from the project root or install the package.

NEVER manually manipulate sys.path:
```python
# ❌ WRONG
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
```

Loader registry path issues: Use `LOADER_BASE_PATH` env var if needed. See `AgentQMS/knowledge/references/development/loader_path_resolution.md`.

Agent Memory & State Tracking
-----------------------------
**Tracking Database** (`data/ops/tracking.db`) maintains agent memory across sessions:
- **Feature Plans**: Implementation tracking with task hierarchies
- **Experiments**: Experimental runs with parameters, metrics, outcomes
- **Debug Sessions**: Debugging context with hypothesis and notes
- **Summaries**: Ultra-concise entity summaries (≤280 chars)

Quick commands:
```bash
# Create a plan
python AgentQMS/agent_tools/utilities/tracking/cli.py plan new --title "My Feature" --owner me

# Check plan status
python AgentQMS/agent_tools/utilities/tracking/cli.py plan status --concise

# Create an experiment
python AgentQMS/agent_tools/utilities/tracking/cli.py exp new --title "Prompt tuning" --objective "Reduce tokens"

# Add experiment run
python AgentQMS/agent_tools/utilities/tracking/cli.py exp run-add <key> 1 --params '{"k":1}' --outcome pass
```

Full reference: `AgentQMS/knowledge/references/tracking/db_api.md` and `AgentQMS/knowledge/references/tracking/cli_reference.md`

Do / Don't
----------
Do:
- Use `artifact_workflow.py` for all artifacts
- Follow naming + frontmatter
- Update indexes when prompted
- Use `PYTHONPATH=.` from project root when running AgentQMS tools

Don't:
- Create or edit artifacts manually
- Place docs in project root
- Add try/except that hide errors
- Manually manipulate sys.path
- Write verbose tutorials in AI instruction files (use concise hints)

When Stuck
----------
- Re-run discovery/validate
- Check logs and browser
- Read `.ai-context.md` for architecture context


