# AgentQMS Toolkit → Agent Tools Migration Guide

**Version**: 1.0
**Last Updated**: 2025-12-06
**Status**: In Progress (0.3.2 → 0.4.0)
**Timeline**: Version 0.3.2 (deprecation warnings), Version 0.4.0 (removal)

---

## Quick Reference

### Migration Steps
1. Find toolkit imports: `grep -r "from AgentQMS.toolkit" . --include="*.py"`
2. Replace with agent_tools equivalent (see table below)
3. Test: `python -c "from AgentQMS.agent_tools.XXX import YYY; print('✓')"`
4. Validate: `make validate && make compliance && make boundary`

---

## Import Mapping Table

### Core Imports (Most Common)

| Old Import | New Import | Module | Status |
|---|---|---|---|
| `from AgentQMS.toolkit.core.artifact_templates import ArtifactTemplates` | `from AgentQMS.agent_tools.core.artifact_templates import ArtifactTemplates` | artifact_templates | ✅ Ready |
| `from AgentQMS.toolkit.utils.runtime import ensure_project_root_on_sys_path` | `from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path` | runtime | ✅ Ready |
| `from AgentQMS.toolkit.core.artifact_workflow import *` | `from AgentQMS.agent_tools.core.artifact_workflow import *` | artifact_workflow | ✅ Ready |
| `from AgentQMS.toolkit.core.context_bundle import *` | `from AgentQMS.agent_tools.core.context_bundle import *` | context_bundle | ✅ Ready |

### Audit Imports

| Old Import | New Import | Module | Status |
|---|---|---|---|
| `from AgentQMS.toolkit.audit.audit_generator import main` | `from AgentQMS.agent_tools.audit.audit_generator import main` | audit_generator | ✅ Ready |
| `from AgentQMS.toolkit.audit.audit_validator import main` | `from AgentQMS.agent_tools.audit.audit_validator import main` | audit_validator | ✅ Ready |
| `from AgentQMS.toolkit.audit.checklist_tool import main` | `from AgentQMS.agent_tools.audit.checklist_tool import main` | checklist_tool | ✅ Ready |

### Documentation Imports

| Old Import | New Import | Module | Status |
|---|---|---|---|
| `from AgentQMS.toolkit.documentation.validate_links import main` | `from AgentQMS.agent_tools.documentation.validate_links import main` | validate_links | ✅ Ready |
| `from AgentQMS.toolkit.documentation.validate_manifest import main` | `from AgentQMS.agent_tools.documentation.validate_manifest import main` | validate_manifest | ✅ Ready |
| `from AgentQMS.toolkit.documentation.auto_generate_index import main` | `from AgentQMS.agent_tools.documentation.auto_generate_index import main` | auto_generate_index | ✅ Ready |

### Compliance Imports

| Old Import | New Import | Module | Status |
|---|---|---|---|
| `from AgentQMS.toolkit.compliance.documentation_quality_monitor import main` | `from AgentQMS.agent_tools.compliance.monitor_artifacts import main` | monitor_artifacts | ✅ Ready (function renamed) |
| `from AgentQMS.toolkit.compliance.validate_artifacts import ArtifactValidator` | `from AgentQMS.agent_tools.compliance.validate_artifacts import ArtifactValidator` | validate_artifacts | ✅ Ready |

### Utility Imports (Less Common)

| Old Import | New Import | Module | Status |
|---|---|---|---|
| `from AgentQMS.toolkit.utils.paths import *` | `from AgentQMS.agent_tools.utils.paths import *` | paths | ✅ Ready |
| `from AgentQMS.toolkit.utilities.tracking.db import *` | `from AgentQMS.agent_tools.utilities.tracking.db import *` | tracking/db | ✅ Ready |
| `from AgentQMS.toolkit.utilities.agent_feedback import main` | `from AgentQMS.agent_tools.utilities.feedback_integration import main` | feedback_integration | ✅ Ready |
| `from AgentQMS.toolkit.utilities.adapt_project import main` | `from AgentQMS.agent_tools.utilities.adapt_project import main` | adapt_project | ✅ Ready |

---

## Migration Examples

### Example 1: Simple Runtime Import

**Before**:
```python
from AgentQMS.toolkit.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()
```

**After**:
```python
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()
```

### Example 2: Artifact Templates

**Before**:
```python
from AgentQMS.toolkit.core.artifact_templates import ArtifactTemplates

templates = ArtifactTemplates()
frontmatter = templates.create_frontmatter("assessment")
```

**After**:
```python
from AgentQMS.agent_tools.core.artifact_templates import ArtifactTemplates

templates = ArtifactTemplates()
frontmatter = templates.create_frontmatter("assessment")
```

### Example 3: Validation

**Before**:
```python
from AgentQMS.toolkit.compliance.validate_artifacts import ArtifactValidator

validator = ArtifactValidator()
results = validator.validate_all()
```

**After**:
```python
from AgentQMS.agent_tools.compliance.validate_artifacts import ArtifactValidator

validator = ArtifactValidator(strict_mode=True)  # Now with strict_mode support
results = validator.validate_all()
```

---

## Deprecated Modules with No Direct Replacements

### Toolkit-Only Code (Keep Using Toolkit Until 0.4.0)

Some toolkit code doesn't have agent_tools equivalents yet:

| Toolkit Module | Reason | Timeline |
|---|---|---|
| `AgentQMS.toolkit.migration.migrate` | Specific to batch migration; not commonly used | Scheduled for 0.4.0 |
| `AgentQMS.toolkit.maintenance.*` | Legacy maintenance scripts | Will consolidate into agent_tools |
| `AgentQMS.toolkit.utils.config` | Configuration loading; being moved to agent_tools.utils.config | Scheduled for 0.4.0 |

**For these modules**: Continue using toolkit until they're available in agent_tools.

---

## Deprecation Warning Handling

When you import from toolkit, you'll see:

```
DeprecationWarning: AgentQMS.toolkit is deprecated as of 0.3.2 and will be removed in 0.4.0.
Use AgentQMS.agent_tools instead.
See docs/artifacts/design_documents/2025-12-06_design_toolkit-deprecation-roadmap.md for migration guide.
```

### Suppress Warnings (If Needed)

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='AgentQMS.toolkit')

# Your old code here (still works)
from AgentQMS.toolkit.core.artifact_templates import ArtifactTemplates
```

**However**, it's better to migrate to agent_tools instead of suppressing warnings.

---

## Testing Your Migration

### Step 1: Run a Single Import Test

```bash
python -c "from AgentQMS.agent_tools.core.artifact_templates import ArtifactTemplates; print('✓ Success')"
```

### Step 2: Test in Your Code

```bash
python your_file.py
```

### Step 3: Run Full Validation Suite

```bash
cd AgentQMS/interface
make validate   # Strict artifact validation
make compliance # Overall compliance check
make boundary   # Framework/project boundary check
```

### Step 4: Run All Interface Commands

```bash
make help       # Should work
make discover   # Should list tools
make status     # Should show status
make audit-framework  # Should audit plugins
```

---

## Common Issues & Solutions

### Issue 1: ImportError After Migration

**Problem**: `ImportError: cannot import name 'XYZ' from 'AgentQMS.agent_tools.XXX'`

**Solution**:
1. Check the mapping table above for correct import path
2. Verify agent_tools module exists: `ls -la AgentQMS/agent_tools/XXX.py`
3. Ensure you're using v0.3.2+: `grep version AgentQMS/__init__.py`

### Issue 2: Function Not Found

**Problem**: `AttributeError: module 'AgentQMS.agent_tools.XXX' has no attribute 'YYY'`

**Solution**:
1. Check if function exists in agent_tools: `grep "def YYY" AgentQMS/agent_tools/XXX.py`
2. Check mapping table for renamed functions
3. If new functionality, use agent_tools version which may have enhancements

### Issue 3: Deprecation Warnings in Tests

**Problem**: Tests fail because of deprecation warnings

**Solution**: Update test imports to use agent_tools:
```python
# In test files
import warnings
# Option A: Migrate imports (preferred)
from AgentQMS.agent_tools.core.artifact_templates import ArtifactTemplates

# Option B: Filter warnings (temporary)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='AgentQMS.toolkit')
```

---

## FAQ

**Q: When will toolkit be removed?**
A: Version 0.4.0, planned for ~4 weeks after 0.3.2 release.

**Q: Will my code break?**
A: No, not until 0.4.0. In 0.3.2, you'll just see deprecation warnings.

**Q: Can I suppress the warnings?**
A: Yes, but migration to agent_tools is preferred. See "Deprecation Warning Handling" section.

**Q: Are there any behavioral differences?**
A: Generally no, but agent_tools may have new features (e.g., `strict_mode` in ArtifactValidator). Check release notes.

**Q: What if I find a bug in agent_tools?**
A: Report it with the mapping from toolkit to agent_tools, and we'll fix it.

**Q: Can I migrate gradually?**
A: Yes, absolutely. Migrate files one at a time. You can mix toolkit and agent_tools imports during 0.3.2.

---

## Resources

- **Detailed Deprecation Roadmap**: `docs/artifacts/design_documents/2025-12-06_design_toolkit-deprecation-roadmap.md`
- **Agent Tools README**: `AgentQMS/agent_tools/README.md`
- **Toolkit README (Legacy)**: `AgentQMS/toolkit/README.md`
- **System Documentation**: `AgentQMS/knowledge/agent/system.md`

---

## Contact & Support

If you encounter issues during migration:

1. Check this guide and the mapping table
2. Review the detailed roadmap document
3. Check `AgentQMS/agent_tools/` for available modules and functions
4. Run `make discover` to list all available tools

---

**Last Updated**: 2025-12-06
**Status**: Ready for Phase 3 implementation
**Next Step**: Complete deprecation warnings in toolkit/__init__.py
