# Hydra Configuration Refactor - AI-Optimized Instructions

**Purpose:** Systematic refactoring of Hydra configuration from fragmented structure to "Domains First" architecture
**Target:** 123 configuration files → 60-80 files with strict domain isolation
**Status:** Ready for execution

---

## 📁 File Organization

### Protocols (Rules & Constraints)
- **01_PROTOCOL_hydra_architectural_laws.md** - The 4 fundamental laws governing all Hydra configs

### Executable Scripts
- **02_SCRIPT_migration_auditor.py** - Detect violations before refactoring
- **03_SCRIPT_hydra_guard.py** - Validate runtime configuration after refactoring

### Templates (Copy & Customize)
- **04_TEMPLATE_global_paths.yaml** - Centralized path definitions
- **06_TEMPLATE_domain_controller.yaml** - Domain isolation pattern

### AI Agent Instructions
- **05_PROMPT_ai_architect_audit.md** - Tool-first structural analysis protocol

### Execution Guides
- **07_GUIDE_refactor_execution.md** - Step-by-step implementation workflow

---

## 🚀 Quick Start

### For AI Agents
1. Read `01_PROTOCOL_hydra_architectural_laws.md` first
2. Follow `05_PROMPT_ai_architect_audit.md` for analysis
3. Execute steps in `07_GUIDE_refactor_execution.md`

### For Humans
1. Run baseline audit: `uv run python 02_SCRIPT_migration_auditor.py --config-root ../../configs`
2. Review violations in generated report
3. Follow `07_GUIDE_refactor_execution.md` step-by-step
4. Validate with `uv run python 03_SCRIPT_hydra_guard.py --domain <domain> --config-name train`

---

## 🎯 Key Objectives

1. **Domain Isolation** - Prevent cross-domain key contamination
2. **File Reduction** - Merge fragments into complete presets (35-50% reduction)
3. **Path Centralization** - Single source of truth for all `${interpolations}`
4. **Archive Separation** - Move legacy/UI configs out of production tree

---

## 📊 Success Metrics

- [ ] Zero critical violations in audit
- [ ] All domains pass Hydra Guard validation
- [ ] File count reduced by 35-50%
- [ ] Training runs successfully for all domains
- [ ] No `${interpolation}` ambiguity

---

## 🔗 Dependencies

**Required Tools:**
- Python 3.8+
- Hydra 1.3+
- OmegaConf
- PyYAML

**Optional (for verification):**
- ADT (Agent Debug Toolkit)
- AST-Grep (sg_search)

---

## 📝 File Naming Convention

- `XX_TYPE_description.ext`
  - `XX` = Sequential number
  - `TYPE` = PROTOCOL | SCRIPT | TEMPLATE | PROMPT | GUIDE
  - `description` = Lowercase with underscores
  - `ext` = .md | .py | .yaml

---

## ⚠️ Critical Warnings

1. **Always backup** before starting refactor
2. **Never skip** the migration auditor baseline
3. **Validate each domain** with Hydra Guard after changes
4. **Test training runs** before committing changes

---

## 🆘 Troubleshooting

See `07_GUIDE_refactor_execution.md` for common issues and solutions.

---

**Last Updated:** 2026-01-17
**Version:** 1.0 (V5 Refactor)
