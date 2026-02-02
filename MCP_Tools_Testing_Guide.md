# MCP Tools Testing Guide (Post-Spec-Kit Migration)

**Date**: 2026-02-02  
**Context**: Phase 7.2 Spec-Kit Migration  
**Purpose**: Test MCP tools after YAML → Markdown specs migration

---

## Executive Summary

**What Changed (Phase 7.2)**:
- Standards moved: `AgentQMS/standards/` → `AgentQMS/specs/`
- Format: YAML files → Markdown `.spec.md` files
- Registry: Now auto-generated from specs → `AgentQMS/.agentqms/registry.yaml`
- Validation: Uses `SpecParser` instead of hard-coded YAML paths

**Expected Impact**: ✅ **Minimal - MCP tools use abstracted interfaces**

Most MCP tools use high-level APIs (`SpecParser`, plugin system) that were updated during Phase 7. Direct testing will verify integration.

---

## Quick Test Summary

| Tool | Phase 7 Impact | Test Priority | Status |
|------|----------------|---------------|--------|
| `create_artifact` | Low - Uses plugin system | High | Test required |
| `validate_artifact` | Medium - Uses SpecParser | High | Test required |
| `list_artifact_templates` | Low - Plugin enumeration | Medium | Test required |
| `check_compliance` | Medium - Uses validation | High | Test required |
| `get_standard` | **High** - Direct spec access | **Critical** | Test required |
| `get_context_bundle` | Low - Plugin system | Medium | Test required |

---

## Test Environment Setup

### 1. Start MCP Server

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
uv run python AgentQMS/mcp_server.py
```

**Expected Output**:
```
MCP Server starting...
Loaded 7 artifact types
Loaded 14 context bundles
Server ready on stdio
```

### 2. Test Client Setup

For manual testing, you can use the MCP inspector or call tools directly via the Claude Desktop app with MCP integration.

**Alternative**: Direct Python testing (see commands below)

---

## Tool-by-Tool Testing

### 1. `create_artifact` 

**What it does**: Creates new artifacts using plugin templates

**Phase 7 Impact**: 
- ✅ Uses plugin system (updated in Phase 7.1)
- ✅ Validation uses SpecParser (updated in Phase 7.3)
- **Risk**: Low

**Test Command**:
```python
# Via MCP
{
  "tool": "create_artifact",
  "arguments": {
    "artifact_type": "design_document",
    "name": "test-mcp-design",
    "title": "MCP Test Design Document"
  }
}
```

**Direct CLI Test**:
```bash
uv run python -c "
from AgentQMS.tools.core.artifacts.create_artifact import create_artifact_cli
import sys
sys.argv = ['', '--type', 'design_document', '--name', 'test-mcp-design', '--title', 'MCP Test Design']
create_artifact_cli()
"
```

**Expected Output**:
```
✓ Created: docs/artifacts/design_documents/test-mcp-design.md
✓ Validated against: AgentQMS/specs/tier1-contracts/compliance.spec.md
✓ ADS version: 2.0
```

**Verification**:
```bash
cat docs/artifacts/design_documents/test-mcp-design.md | head -20
# Should show frontmatter with ads_version: "2.0"
```

---

### 2. `validate_artifact`

**What it does**: Validates artifacts against specs

**Phase 7 Impact**:
- ✅ Uses `SpecParser` (completely refactored in Phase 7.3)
- ✅ Loads rules from `AgentQMS/specs/`
- ✅ Strict constraints from `standards_db.json`
- **Risk**: Medium (validation logic changed)

**Test Command**:
```python
{
  "tool": "validate_artifact",
  "arguments": {
    "file_path": "docs/artifacts/design_documents/test-mcp-design.md"
  }
}
```

**Direct CLI Test**:
```bash
uv run python AgentQMS/tools/compliance/validate_artifacts.py \
  --file docs/artifacts/design_documents/test-mcp-design.md
```

**Expected Output**:
```
✓ Validating: test-mcp-design.md
✓ Artifact type: design_document
✓ Frontmatter: Valid (ADS 2.0)
✓ Naming: Valid
✓ Structure: Valid
✓ Strict constraints: 14 field rules applied

PASSED: 1/1 artifacts validated
```

**Failure Test** (verify error detection):
```bash
# Create invalid artifact
echo "---
ads_version: '1.0'
type: invalid_type
---
# Test" > /tmp/invalid-test.md

uv run python AgentQMS/tools/compliance/validate_artifacts.py \
  --file /tmp/invalid-test.md
```

**Expected**:
```
✗ FAILED: invalid-test.md
  - Unknown artifact type: invalid_type
  - ADS version mismatch (expected 2.0, got 1.0)
```

---

### 3. `list_artifact_templates`

**What it does**: Lists available artifact type plugins

**Phase 7 Impact**:
- ✅ Reads from plugin system (fixed in Phase 7.1)
- ✅ No direct spec references
- **Risk**: Low

**Test Command**:
```python
{
  "tool": "list_artifact_templates",
  "arguments": {}
}
```

**Direct CLI Test**:
```bash
uv run python -m AgentQMS.tools.core.plugins --list
```

**Expected Output**:
```
📦 Artifact Types:
   • assessment (v?) [framework]
   • audit (v?) [framework]
   • bug_report (v?) [framework]
   • design_document (v?) [framework]
   • implementation_plan (v?) [framework]
   • vlm_report (v?) [framework]
   • walkthrough (v?) [framework]

Total: 7 artifact types
```

**Verification**:
```bash
# Check plugin snapshot
cat AgentQMS/.agentqms/state/plugins.yaml | grep "artifact_types:" -A 10
```

---

### 4. `check_compliance`

**What it does**: Runs full compliance check across all artifacts

**Phase 7 Impact**:
- ✅ Uses SpecParser for validation
- ✅ Reads from specs directory
- **Risk**: Medium

**Test Command**:
```python
{
  "tool": "check_compliance",
  "arguments": {}
}
```

**Direct CLI Test**:
```bash
uv run python AgentQMS/tools/compliance/validate_artifacts.py --all
```

**Expected Output**:
```
🔍 Scanning: docs/artifacts/
✓ Found: 142 artifacts

Validating artifacts...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 142/142

Results:
  ✓ Passed: 140
  ✗ Failed: 2
  ⚠ Warnings: 5

Failed artifacts:
  - old-artifact.md (ADS v1.0 deprecated)
  - missing-frontmatter.md (No frontmatter)
```

**Check Report Generation**:
```bash
ls -lh compliance_report.json
# Should exist if --json flag used
```

---

### 5. `get_standard` ⚠️ **CRITICAL TEST**

**What it does**: Retrieves spec content by name (fuzzy match)

**Phase 7 Impact**:
- 🔴 **HIGH** - Directly accesses spec files
- 🔴 Path changed: `standards/tier1-sst/` → `specs/tier1-contracts/`
- 🔴 Format changed: `.yaml` → `.spec.md`
- **Risk**: **HIGH** - May need updates

**Test Command**:
```python
{
  "tool": "get_standard",
  "arguments": {
    "name": "compliance"
  }
}
```

**Direct Test**:
```bash
uv run python -c "
from AgentQMS.tools.utils.spec_parser import SpecParser
parser = SpecParser()
spec = parser.get_spec_by_name('compliance')
print(f'Found: {spec.name}')
print(f'Path: {spec.file_path}')
print(spec.content[:200])
"
```

**Expected Output**:
```
Found: compliance
Path: AgentQMS/specs/tier1-contracts/compliance.spec.md
# Compliance and Artifact Standards

This specification defines...
```

**Alternative Names to Test** (fuzzy matching):
```bash
# Test various fuzzy matches
for name in "artifact-types" "naming" "validation" "workflow"; do
  echo "Testing: $name"
  uv run python -c "
from AgentQMS.tools.utils.spec_parser import SpecParser
parser = SpecParser()
try:
    spec = parser.get_spec_by_name('$name')
    print(f'  ✓ Found: {spec.name}')
except Exception as e:
    print(f'  ✗ Error: {e}')
"
done
```

**If this fails**, the tool needs updating to:
1. Search in `AgentQMS/specs/` instead of `AgentQMS/standards/`
2. Handle `.spec.md` extension
3. Use fuzzy matching across spec files

---

### 6. `get_context_bundle`

**What it does**: Retrieves context bundles for tasks

**Phase 7 Impact**:
- ✅ Uses plugin system (fixed in Phase 7.1)
- ✅ No direct spec references
- **Risk**: Low

**Test Command**:
```python
{
  "tool": "get_context_bundle",
  "arguments": {
    "task_description": "fixing validation errors"
  }
}
```

**Direct CLI Test**:
```bash
uv run python AgentQMS/tools/core/context/get_context.py \
  --task "fixing validation errors"
```

**Expected Output**:
```
📚 Context Bundle: compliance-check
Files:
  - AgentQMS/specs/tier1-contracts/compliance.spec.md
  - AgentQMS/tools/compliance/validate_artifacts.py
  - AgentQMS/.agentqms/standards_db.json

Token count: ~8,500 tokens
```

**Test by Bundle Name**:
```bash
uv run python AgentQMS/tools/core/context/get_context.py \
  --type compliance-check
```

---

## Automated Test Script

```bash
#!/bin/bash
# mcp_tools_test.sh

echo "=== MCP Tools Integration Test ==="
echo ""

# 1. Test create_artifact
echo "1. Testing create_artifact..."
uv run python -c "
from AgentQMS.tools.core.artifacts.create_artifact import create_artifact
result = create_artifact('design_document', 'mcp-test-auto', 'Automated MCP Test')
print(f'✓ Created: {result}')
" && echo "✓ PASSED" || echo "✗ FAILED"

# 2. Test validate_artifact
echo ""
echo "2. Testing validate_artifact..."
uv run python AgentQMS/tools/compliance/validate_artifacts.py \
  --file docs/artifacts/design_documents/mcp-test-auto.md \
  && echo "✓ PASSED" || echo "✗ FAILED"

# 3. Test list_artifact_templates
echo ""
echo "3. Testing list_artifact_templates..."
uv run python -m AgentQMS.tools.core.plugins --list | grep -q "artifact_types" \
  && echo "✓ PASSED" || echo "✗ FAILED"

# 4. Test check_compliance
echo ""
echo "4. Testing check_compliance..."
uv run python AgentQMS/tools/compliance/validate_artifacts.py --all > /dev/null 2>&1 \
  && echo "✓ PASSED" || echo "✗ FAILED"

# 5. Test get_standard (CRITICAL)
echo ""
echo "5. Testing get_standard..."
uv run python -c "
from AgentQMS.tools.utils.spec_parser import SpecParser
parser = SpecParser()
spec = parser.get_spec_by_name('compliance')
assert spec is not None, 'Spec not found'
assert 'specs/' in str(spec.file_path), 'Wrong path'
print(f'✓ Found: {spec.name} at {spec.file_path}')
" && echo "✓ PASSED" || echo "✗ FAILED"

# 6. Test get_context_bundle
echo ""
echo "6. Testing get_context_bundle..."
uv run python AgentQMS/tools/core/context/get_context.py \
  --task "validation" --list > /dev/null 2>&1 \
  && echo "✓ PASSED" || echo "✗ FAILED"

echo ""
echo "=== Test Complete ==="
```

**Save as**: `scripts/test/test_mcp_tools.sh`

**Run**:
```bash
chmod +x scripts/test/test_mcp_tools.sh
bash scripts/test/test_mcp_tools.sh
```

---

## Expected Test Results

### All Tests Pass ✅
```
1. create_artifact      ✓ PASSED
2. validate_artifact    ✓ PASSED
3. list_templates       ✓ PASSED
4. check_compliance     ✓ PASSED
5. get_standard         ✓ PASSED
6. get_context_bundle   ✓ PASSED
```

### If Failures Occur

**`get_standard` fails** 🔴:
- **Likely cause**: Tool still references old `standards/` path
- **Fix**: Update to use `SpecParser` and search in `specs/`
- **Files to check**: 
  - `AgentQMS/mcp_server.py` (tool implementation)
  - Search for hard-coded `standards/` paths

**`validate_artifact` fails** 🟡:
- **Likely cause**: Spec parsing or strict constraint loading
- **Check**: `AgentQMS/.agentqms/standards_db.json` exists
- **Verify**: `SpecParser` can load specs

**`create_artifact` fails** 🟡:
- **Likely cause**: Plugin template loading
- **Check**: `AgentQMS/.agentqms/state/plugins.yaml` shows 7 types
- **Verify**: Run `make qms-plugins` to reload

---

## MCP Server Integration Test

Test all tools via MCP protocol:

```bash
# Start server
uv run python AgentQMS/mcp_server.py &
MCP_PID=$!

# Send test requests (requires MCP client)
# Or use Claude Desktop with MCP configuration

# Cleanup
kill $MCP_PID
```

**MCP Configuration** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "agentqms": {
      "command": "uv",
      "args": ["run", "python", "AgentQMS/mcp_server.py"],
      "cwd": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2"
    }
  }
}
```

---

## Summary

**Test Priority**:
1. **Critical**: `get_standard` (direct spec access)
2. **High**: `validate_artifact`, `check_compliance` (use SpecParser)
3. **Medium**: Others (use plugin system)

**Manual Test Time**: ~10 minutes  
**Automated Test Time**: ~2 minutes

**Next Steps**:
1. Run automated test script
2. If `get_standard` fails, update MCP server implementation
3. Verify all tools via actual MCP client (Claude Desktop)
4. Document any issues found

---

**Last Updated**: 2026-02-02  
**Test Status**: Pending execution  
**Phase**: 7.2 Post-Migration Validation
