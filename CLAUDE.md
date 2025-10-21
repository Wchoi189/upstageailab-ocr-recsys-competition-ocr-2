# Claude Code Protocol

**Purpose**: Mandatory consultation checklist for all code and documentation changes.

---

## 🔍 BEFORE Making ANY Changes

### 1. Consult Documentation First
- **Primary Index**: [`docs/ai_handbook/index.md`](docs/ai_handbook/index.md) - Your single source of truth
- **Find relevant protocol**: [`docs/ai_handbook/02_protocols/`](docs/ai_handbook/02_protocols/)
- **Check architecture**: [`docs/ai_handbook/03_references/architecture/`](docs/ai_handbook/03_references/architecture/) for system modifications
- **Review data contracts**: [`docs/pipeline/data_contracts.md`](docs/pipeline/data_contracts.md) for data structure changes

### 2. Use Context Bundles
Reference [`docs/ai_handbook/index.md`](docs/ai_handbook/index.md) Section 2 for task-specific bundles:
- **Development**: Coding Standards → Architecture → Hydra Registry
- **Debugging**: Debugging Workflow → Command Registry → Past Experiments
- **Training**: Training Protocol → Performance Guides
- **Configuration**: Hydra Reference → Troubleshooting

---

## ✅ AFTER Making Changes

### Required Updates (in order):
1. **Changelog**: [`docs/CHANGELOG.md`](docs/CHANGELOG.md)
2. **Dated entry**: `docs/ai_handbook/05_changelog/YYYY-MM/DD_description.md`
3. **Bug report** (if applicable): `docs/bug_reports/BUG_YYYY_NNN_DESCRIPTION.md`
4. **Protocol updates** (if new patterns emerged)

---

## 📁 Documentation Rules

### ❌ NEVER:
- Place files in project root
- Duplicate existing documentation
- Ignore existing protocols
- Use ALL CAPS in filenames

### ✅ ALWAYS:
- Check if documentation already exists
- Update existing docs instead of creating new ones
- Use lowercase filenames with hyphens and timestamps
- Ask where files belong if unsure
- Reference [`docs/ai_handbook/02_protocols/documentation-management.md`](docs/ai_handbook/02_protocols/documentation-management.md)

---

## 🎯 Critical References

| Task | Primary Reference | Secondary |
|------|------------------|-----------|
| **Data Changes** | [`docs/pipeline/data_contracts.md`](docs/pipeline/data_contracts.md) | Architecture docs |
| **Naming** | [`docs/ai_handbook/index.json`](docs/ai_handbook/index.json) | Documentation management |
| **Performance** | [`configs/data/performance_preset/README.md`](configs/data/performance_preset/README.md) | FP16/Cache guides |
| **File Placement** | [`docs/ai_handbook/02_protocols/documentation-management.md`](docs/ai_handbook/02_protocols/documentation-management.md) | Index.md |

---

## 🚨 Mandatory Checks

**Before ANY file creation or modification:**

1. ❓ "Does similar documentation already exist?"
2. ❓ "Should I update existing docs instead?"
3. ❓ "Where does this file belong according to protocols?"
4. ❓ "Am I following the established naming conventions?"

**If you can't answer all 4 questions, STOP and consult the documentation first.**

---

**Remember**: Documentation exists to prevent mistakes. Use it proactively, not reactively.
