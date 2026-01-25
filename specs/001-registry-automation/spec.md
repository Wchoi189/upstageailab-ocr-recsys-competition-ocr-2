# Feature Specification: Automated Registry Generation System

**Feature Branch**: `001-registry-automation`  
**Created**: 2026-01-26  
**Status**: Draft  
**Input**: Create a comprehensive specification for automating the AgentQMS registry.yaml generation system

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer Adds New Standard File (Priority: P1)

A developer creates a new coding standard file `tier2-framework/coding/security-patterns.yaml` with proper ADS header metadata (tier, triggers, description). The registry system automatically detects this new file and updates `registry.yaml` to include it in the appropriate task mappings, making it immediately discoverable by AI agents without any manual intervention.

**Why this priority**: This is the core automation use case that eliminates manual registry maintenance and prevents synchronization drift—the primary pain point driving this feature.

**Independent Test**: Create a new standard file with ADS header, run the registry sync tool, verify the registry.yaml now includes the new file in correct task mappings. Agents can discover and load the new standard using path or task-based queries.

**Acceptance Scenarios**:

1. **Given** a new YAML standard file with complete ADS header in tier2-framework/coding/, **When** registry sync is triggered, **Then** registry.yaml includes the file under code_quality task mapping with correct tier classification
2. **Given** the updated registry, **When** an agent queries standards for "code review", **Then** the new security-patterns.yaml is returned in the results alongside existing coding standards
3. **Given** a standard file with malformed ADS header, **When** registry sync is triggered, **Then** sync fails with clear validation error identifying the specific malformed fields

---

### User Story 2 - Agent Discovers Standards for Specific Task (Priority: P1)

An AI agent working on configuration file changes needs to know which standards apply. Instead of loading the entire 440-line registry.yaml, the agent calls `resolve_standards(task="config_files")` and receives only the 3 relevant standard file paths (hydra-v5-rules.yaml, configuration-standards.yaml, etc.) with their tier classifications.

**Why this priority**: This addresses the "context burden" problem—reducing agent memory footprint while maintaining accurate standard discovery. Essential for scalability.

**Independent Test**: Query the resolution tool with a task type, measure token count of response vs. full registry load, verify all returned standards are relevant to the task and none are missed.

**Acceptance Scenarios**:

1. **Given** an agent needs to modify a Hydra config file, **When** agent queries `resolve_standards(task="config_files")`, **Then** receives list of 3 configuration-related standards without loading entire registry
2. **Given** an agent working on artifact creation, **When** agent queries `resolve_standards(file_path="docs/artifacts/new-design.md")`, **Then** receives naming-conventions.yaml, artifact-types.yaml, and artifact_rules.yaml
3. **Given** a file path matching multiple task patterns, **When** resolution is requested, **Then** returns union of all applicable standards ordered by priority

---

### User Story 3 - Existing Standard File Relocated (Priority: P2)

A developer moves `tier3-governance/compliance-rules.yaml` to `tier2-framework/quality/compliance-rules.yaml` to correct tier misclassification. The ADS header in the file already contains the correct tier metadata. Upon running registry sync, the registry.yaml automatically reflects the new path without requiring manual path updates or grep-replace operations.

**Why this priority**: Eliminates a major source of registry staleness during refactoring—critical for maintaining architecture purity as the system evolves.

**Independent Test**: Move a standard file with ADS header to new location, run registry sync, verify registry.yaml uses new path and old path references are removed automatically.

**Acceptance Scenarios**:

1. **Given** a standard file relocated to different tier directory, **When** registry sync runs, **Then** registry.yaml reflects new path and no broken references remain
2. **Given** a file move that changes tier classification, **When** sync validates the file, **Then** ADS header tier matches new directory tier or sync fails with mismatch error
3. **Given** multiple files moved in single operation, **When** sync runs once, **Then** all files are correctly indexed in single registry update

---

### User Story 4 - Batch Migration of Legacy Standards (Priority: P2)

A maintainer runs a migration tool to add ADS headers to all 71 existing standard files. The tool processes each file, prompts for missing metadata (triggers, task associations), validates the header schema, and writes updated files. After migration completes, a full registry sync generates the complete registry.yaml from distributed headers.

**Why this priority**: One-time migration is required before automation can go live—high value but not blocking for incremental adoption.

**Independent Test**: Run migration on subset of files (10 standards), verify all receive valid ADS headers, run sync, verify those 10 are correctly indexed while unmigrated files are skipped or flagged.

**Acceptance Scenarios**:

1. **Given** 71 legacy standard files without ADS headers, **When** migration tool runs, **Then** all files receive valid headers with tier, name, description, and triggers populated
2. **Given** migration encounters ambiguous trigger keywords, **When** user is prompted, **Then** tool suggests default triggers based on file path and allows override
3. **Given** fully migrated standards directory, **When** registry sync runs, **Then** generates complete registry.yaml with all 71 files indexed

---

### User Story 5 - Pre-Commit Validation Prevents Invalid Standards (Priority: P3)

A developer attempts to commit a new standard file without ADS header or with invalid tier metadata. The pre-commit hook runs registry validation, detects the violation, and blocks the commit with clear error message indicating which fields are missing or invalid in which file.

**Why this priority**: Preventive quality gate—important for long-term maintenance but system can function with manual checks initially.

**Independent Test**: Attempt to commit file with missing ADS header, verify commit is blocked with helpful error. Correct the header, commit succeeds.

**Acceptance Scenarios**:

1. **Given** a new standard file without ADS header, **When** developer commits, **Then** pre-commit hook blocks commit with error "Missing ADS header in [filename]"
2. **Given** a standard with tier=5 (invalid), **When** commit is attempted, **Then** hook blocks with "Invalid tier value: must be 1-4"
3. **Given** all standard files have valid ADS headers, **When** commit is attempted, **Then** validation passes and commit proceeds

---

### Edge Cases

- **Circular Dependencies**: What happens when standard A's triggers reference standard B, and B references A? System should detect cycles during validation and fail with cycle path description.
- **No Triggers Defined**: A standard file has valid ADS header but empty triggers map. Should it be indexed? Decision: Yes, indexed but only discoverable via explicit path query, not task-based discovery. Warning logged.
- **Glob Pattern Conflicts**: Two standards have overlapping path_patterns (e.g., `configs/**/*.yaml` and `configs/hydra/*.yaml`). Resolution: Both standards are returned, ordered by priority field, agent decides which to apply.
- **Registry Generation Failure**: If sync_registry.py crashes mid-run, is registry.yaml left in broken state? Solution: Write to temporary file, validate, then atomic rename to prevent corruption.
- **Schema Evolution**: New ADS v2.0 schema adds required field. How do v1.0 headers work? Solution: schema_version field in header allows migration tools to handle multiple versions during transition.
- **Deleted Standard Files**: File removed but still referenced in old registry. Sync should detect missing files and prune dead references, logging removed entries.
- **Case Sensitivity**: Standard file paths in registry vs. filesystem on case-insensitive systems (macOS). Normalize all paths to lowercase for comparison.

## Requirements *(mandatory)*

### Functional Requirements

#### Core Automation

- **FR-001**: System MUST scan all YAML files in `AgentQMS/standards/` directory tree and extract ADS header metadata
- **FR-002**: System MUST validate each ADS header against JSON schema before including file in registry
- **FR-003**: System MUST generate registry.yaml atomically (write to temp file, validate, then rename) to prevent corruption
- **FR-004**: Registry generation MUST complete in under 5 seconds for 71 standard files to support rapid iteration
- **FR-005**: System MUST detect and report files moved/renamed between sync runs by comparing previous registry state

#### ADS Header Schema

- **FR-006**: ADS header MUST include required fields: `ads_version`, `type`, `tier` (1-4), `name`, `description`
- **FR-007**: ADS header MUST support optional fields: `triggers` (map of task_id to keywords/patterns/path_patterns), `priority` (integer), `auto_load` (boolean), `status` (draft|active|deprecated)
- **FR-008**: ADS header `triggers` field MUST allow mapping to multiple task types with different keyword/pattern sets per task
- **FR-009**: System MUST reject ADS headers with tier value outside range 1-4
- **FR-010**: System MUST validate path_patterns in triggers are valid glob expressions

#### Standard Resolution Tool

- **FR-011**: System MUST provide `resolve_standards(task: str)` function that returns list of standard file paths for given task type
- **FR-012**: System MUST provide `resolve_standards(file_path: str)` function that returns standards applicable to given file path via glob matching
- **FR-013**: Resolution function MUST return standards ordered by priority field (1 = highest)
- **FR-014**: Resolution function MUST return empty list if no standards match, not raise exception
- **FR-015**: Resolution function MUST support combining task and file_path filters (return standards matching BOTH)

#### Migration Support

- **FR-016**: System MUST provide migration tool that adds ADS headers to legacy standard files without headers
- **FR-017**: Migration tool MUST infer tier from directory path (tier1-sst/ → tier=1)
- **FR-018**: Migration tool MUST suggest triggers based on file path analysis and allow user override
- **FR-019**: Migration tool MUST preserve all existing YAML content below the header when inserting ADS metadata
- **FR-020**: Migration tool MUST operate in dry-run mode showing proposed changes before applying

#### Validation and Error Handling

- **FR-021**: System MUST provide validation command that checks all standards for ADS header compliance without regenerating registry
- **FR-022**: Validation MUST report specific line numbers and field names for schema violations
- **FR-023**: System MUST detect circular dependencies in trigger references and fail with cycle description
- **FR-024**: System MUST detect glob pattern conflicts (overlapping patterns) and log warnings with affected standards
- **FR-025**: System MUST handle missing referenced files gracefully, logging warnings but continuing registry generation

#### Integration Points

- **FR-026**: System MUST integrate with AgentQMS CLI via `aqms registry sync` command
- **FR-027**: System MUST support pre-commit hook integration that validates new/modified standard files
- **FR-028**: System MUST log all registry updates to CHANGELOG.md with timestamp, files changed, and user
- **FR-029**: Registry sync MUST be callable from `project_compass` pulse-export workflow automatically
- **FR-030**: System MUST support manual override mode where registry.yaml changes are reviewed before commit

### Key Entities

- **ADS Header**: YAML frontmatter block containing standard metadata (tier, triggers, name, description, version). Lives at top of each standard file. Parsed during registry compilation.

- **Registry Entry**: Compiled representation of a standard in registry.yaml task_mappings. Contains: task_id, description, priority, standards list, triggers. Generated from aggregating ADS headers.

- **Task Mapping**: Association between task type identifier (e.g., "config_files") and list of standard files that govern that task. Includes trigger patterns for automatic task detection.

- **Standard File**: YAML document in AgentQMS/standards/ hierarchy. Contains ADS header followed by actual standard rules/specifications. File path encodes tier via directory.

- **Resolution Query**: Request from agent to discover applicable standards. Can specify task type, file path, or both. Returns filtered list of standard paths.

- **Trigger Pattern**: Matching rule in ADS header that associates standard with task. Types: keywords (exact match), patterns (regex), path_patterns (glob). Multiple triggers can map to same standard.

- **Tier Classification**: Functional category (1-4) determining standard's scope and authority. Tier 1 = constitutional, Tier 2 = framework/infrastructure, Tier 3 = agent personas, Tier 4 = operational workflows.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Registry.yaml becomes auto-generated artifact—zero manual edits required after migration complete (100% automation coverage)

- **SC-002**: Agent context footprint for standard discovery reduced by 80% (from ~1000 tokens for full registry to ~200 tokens for resolved standards)

- **SC-003**: All 71 existing standard files successfully retrofitted with valid ADS headers passing schema validation

- **SC-004**: Registry synchronization completes in under 5 seconds for full standards directory (71 files), enabling rapid development iteration

- **SC-005**: Zero registry synchronization errors reported in production after 1 month of automated operation (indicating robust file detection and validation)

- **SC-006**: Standard file relocation operations (moves/renames) automatically reflected in registry within 1 sync cycle without manual path updates

- **SC-007**: 95% of agent queries for applicable standards return complete result set (no missing standards due to incomplete triggers)

- **SC-008**: Pre-commit validation catches 100% of invalid ADS headers before reaching main branch (zero malformed standards merged)

- **SC-009**: Migration tool processes legacy standards with 100% content preservation (no data loss when adding headers)

- **SC-010**: Agent prompt modifications require only single instruction to enforce ADS header creation (minimal onboarding friction)

## Problem Statement

The AgentQMS system relies on a centralized `registry.yaml` file (441 lines) that serves as the discovery mechanism for AI agents to find applicable standards. Currently, this file is manually maintained, creating several critical problems:

1. **Synchronization Drift**: When standard files are moved, renamed, or created, developers must manually update registry.yaml. This leads to stale references, broken paths, and agents loading incorrect or missing standards.

2. **Context Burden**: AI agents must load the entire registry (1000+ tokens) to discover which standards apply to their current task, consuming valuable context window even when only 2-3 standards are relevant.

3. **Maintenance Burden**: With 71 YAML standard files across 4 tiers, every structural change requires careful registry updates. This slows refactoring and increases risk of human error.

4. **Redundant Catalog**: The current registry contains both task mappings and a full file catalog at the bottom—duplication that increases maintenance overhead and potential for inconsistency.

5. **Trigger Management**: Keywords, patterns, and path globs that determine when standards apply are centralized in the registry, disconnected from the standards themselves. Updates require coordinated changes across multiple files.

### Current Pain Points

- Developer creates new standard: Requires manual registry edit + path entry + trigger configuration
- File relocation: Requires grep-replace across registry to update all path references  
- Agent needs artifact standards: Loads entire 440-line registry to extract 3 relevant files
- Trigger refinement: Must edit registry.yaml even when only modifying standard-specific behavior
- Schema evolution: No versioning mechanism for registry structure changes

### Impact

Without automation, the registry becomes a bottleneck for AgentQMS evolution. As the system scales beyond 100 standards, manual maintenance becomes unsustainable and error-prone. The goal is to transform registry.yaml from a manually-edited document into a compiled asset generated from distributed metadata.

## Solution Architecture

### Core Design: Distributed Metadata with Centralized Compilation

The solution replaces centralized registry maintenance with a **distributed metadata system** where each standard file contains its own discovery metadata (ADS header), and the registry becomes an auto-generated compilation artifact.

### Architecture Components

#### 1. ADS Header Standard

Every standard file must include a YAML frontmatter header (ADS = Agent Discovery Specification) with structured metadata:

```yaml
---
ads_version: "1.0"
type: standard
tier: 2
priority: 100
name: "Python Core Coding Standards"
description: "Core Python style guide and anti-patterns"
status: active
triggers:
  code_quality:
    priority: 1
    keywords:
      - python code
      - code review
      - refactoring
    patterns:
      - \bpython\b.*\bcode\b
      - \bimplement\b.*\bclass\b
    path_patterns:
      - "**/*.py"
      - "ocr/**/*.py"
  code_changes:
    priority: 2
    keywords:
      - modify code
      - update function
---

# Standard Content Below
[Actual standard rules and specifications...]
```

**Key Fields**:
- `ads_version`: Schema version for future evolution
- `tier`: Functional classification (1-4) matching directory structure
- `triggers`: Map of task_id → activation criteria (keywords, regex patterns, file globs)
- `priority`: Numeric ranking for resolution ordering (lower = higher priority)
- `status`: Lifecycle state (draft, active, deprecated)

#### 2. Registry Crawler (`sync_registry.py`)

Python tool that compiles registry.yaml from distributed ADS headers:

**Algorithm**:
1. Scan all `*.yaml` files in `AgentQMS/standards/` recursively
2. Extract and validate ADS header against JSON schema
3. Build task_mappings by aggregating triggers from all standards
4. Generate registry.yaml with metadata (generation timestamp, file count, validation status)
5. Write atomically (temp file → validate → rename) to prevent corruption

**Invocation Points**:
- Manual: `aqms registry sync`
- Pre-commit hook: Validate new/modified standards
- Post-merge: Regenerate registry after branch integration
- Pulse export: Sync before committing pulse artifacts

#### 3. Resolution Tool (`resolve_standards()`)

Lightweight query interface for agents to discover applicable standards without loading full registry:

**API**:
```python
# Query by task type
standards = resolve_standards(task="config_files")
# Returns: ['tier2-framework/hydra-v5-rules.yaml', ...]

# Query by file path
standards = resolve_standards(file_path="configs/train/optimizer.yaml")
# Returns standards matching path globs

# Combined query
standards = resolve_standards(task="code_quality", file_path="ocr/models/vgg.py")
# Returns intersection of both criteria
```

**Benefits**:
- Reduces agent context from 1000 tokens (full registry) to 200 tokens (resolved standards)
- Returns standards ordered by priority
- Handles glob pattern matching for path-based queries
- No file I/O if registry cached in memory

#### 4. JSON Schema Validator

Schema file at `AgentQMS/standards/schemas/ads-header.json` defines required/optional fields, data types, and validation rules:

**Validates**:
- Required fields present (ads_version, tier, name, description)
- Tier value in range 1-4
- Triggers map structure correct
- Path patterns are valid globs
- Priority is positive integer
- Status is enum (draft|active|deprecated)

**Used By**:
- Registry crawler during compilation
- Pre-commit hook during validation
- Migration tool when adding headers

#### 5. Migration Tool (`migrate_legacy_standards.py`)

One-time migration script to retrofit existing 71 standard files with ADS headers:

**Process**:
1. Scan standards directory for files without ADS headers
2. Infer tier from directory path (tier2-framework/ → tier=2)
3. Parse filename and first 10 lines to suggest name/description
4. Analyze current registry.yaml to extract existing triggers for this file
5. Prompt user to confirm or override inferred metadata
6. Insert ADS header at top of file, preserving all content below
7. Validate header against schema
8. Log migration for each file to CHANGELOG.md

**Modes**:
- `--dry-run`: Show proposed headers without modifying files
- `--auto`: Use all defaults without prompts (risky)
- `--interactive`: Prompt for each file (default)
- `--file <path>`: Migrate single file only

### Partial Registry Pattern (Optional Future Enhancement)

To reduce registry size further, consider tier-level INDEX.yaml files:

```
AgentQMS/standards/
├── registry.yaml (master, auto-generated)
├── tier1-sst/
│   └── INDEX.yaml (tier 1 standards only)
├── tier2-framework/
│   └── INDEX.yaml (tier 2 standards only)
...
```

Master registry becomes compilation of tier indices. Agents can load only relevant tier index for faster resolution.

### Validation Workflow

**Pre-Commit Hook**:
1. Detect modified YAML files in standards/
2. Extract ADS headers from modified files
3. Validate against schema
4. Block commit if validation fails
5. Log validation result

**CI Pipeline**:
1. On PR: Full registry regeneration from scratch
2. Compare generated registry to committed registry
3. Flag diff as error if registry out of sync
4. Require developer to run `aqms registry sync` and commit

### Rollback Procedure

If registry compilation fails or produces invalid output:

1. Registry written to `.registry.yaml.tmp` first
2. Validation runs on temp file
3. If validation fails: Keep old registry, log error, exit with error code
4. If validation passes: Atomic rename temp → registry.yaml
5. Previous registry automatically preserved by git history
6. Emergency restore: `git checkout HEAD~1 AgentQMS/standards/registry.yaml`

## Technical Design Details

### ADS Header Schema (JSON Schema)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AgentQMS ADS Header v1.0",
  "type": "object",
  "required": ["ads_version", "type", "tier", "name", "description"],
  "properties": {
    "ads_version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+$",
      "description": "Schema version (e.g., '1.0')"
    },
    "type": {
      "type": "string",
      "enum": ["standard", "principle", "workflow", "catalog"],
      "description": "Document type classification"
    },
    "tier": {
      "type": "integer",
      "minimum": 1,
      "maximum": 4,
      "description": "Functional tier (1=SST, 2=Framework, 3=Agents, 4=Workflows)"
    },
    "priority": {
      "type": "integer",
      "minimum": 1,
      "description": "Resolution ordering (lower number = higher priority)"
    },
    "name": {
      "type": "string",
      "minLength": 5,
      "maxLength": 100,
      "description": "Human-readable standard name"
    },
    "description": {
      "type": "string",
      "minLength": 10,
      "maxLength": 500,
      "description": "What this standard governs"
    },
    "status": {
      "type": "string",
      "enum": ["draft", "active", "deprecated"],
      "default": "active"
    },
    "triggers": {
      "type": "object",
      "description": "Map of task_id to activation criteria",
      "patternProperties": {
        "^[a-z_]+$": {
          "type": "object",
          "properties": {
            "priority": {
              "type": "integer",
              "minimum": 1
            },
            "keywords": {
              "type": "array",
              "items": { "type": "string" }
            },
            "patterns": {
              "type": "array",
              "items": { "type": "string", "format": "regex" }
            },
            "path_patterns": {
              "type": "array",
              "items": { "type": "string" }
            }
          }
        }
      }
    }
  }
}
```

### Registry Compilation Algorithm

**Input**: Directory tree `AgentQMS/standards/`  
**Output**: `registry.yaml` with task_mappings

**Pseudocode**:
```
registry = {
  metadata: { generated_at, file_count, ... },
  root_map: { tier1: path, tier2: path, ... },
  task_mappings: {}
}

for each yaml_file in recursive_scan("AgentQMS/standards/"):
  if yaml_file.name == "registry.yaml": continue
  
  header = extract_ads_header(yaml_file)
  if header is None:
    log_warning(f"No ADS header: {yaml_file}")
    continue
  
  if not validate_schema(header):
    raise ValidationError(f"Invalid header: {yaml_file}")
  
  # Verify tier matches directory
  expected_tier = infer_tier_from_path(yaml_file)
  if header.tier != expected_tier:
    raise TierMismatchError(f"{yaml_file}: header tier={header.tier}, path tier={expected_tier}")
  
  # Aggregate triggers into task_mappings
  for task_id, trigger in header.triggers.items():
    if task_id not in registry.task_mappings:
      registry.task_mappings[task_id] = {
        description: infer_description(task_id),
        priority: trigger.priority,
        standards: [],
        triggers: { keywords: [], patterns: [], path_patterns: [] }
      }
    
    registry.task_mappings[task_id].standards.append(yaml_file.path)
    registry.task_mappings[task_id].triggers.keywords.extend(trigger.keywords)
    registry.task_mappings[task_id].triggers.patterns.extend(trigger.patterns)
    registry.task_mappings[task_id].triggers.path_patterns.extend(trigger.path_patterns)

# Deduplicate triggers
for task in registry.task_mappings.values():
  task.triggers.keywords = unique(task.triggers.keywords)
  task.triggers.patterns = unique(task.triggers.patterns)
  task.triggers.path_patterns = unique(task.triggers.path_patterns)

# Write atomically
temp_file = ".registry.yaml.tmp"
write_yaml(temp_file, registry)
validate_registry_structure(temp_file)
atomic_rename(temp_file, "registry.yaml")
```

### Resolution Tool Implementation

```python
# AgentQMS/tools/resolution/resolver.py
import yaml
import fnmatch
import re
from pathlib import Path
from typing import List, Optional

class StandardResolver:
    def __init__(self, registry_path: str = "AgentQMS/standards/registry.yaml"):
        with open(registry_path) as f:
            self.registry = yaml.safe_load(f)
    
    def resolve_standards(
        self,
        task: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> List[dict]:
        """
        Returns list of standard file paths matching criteria.
        Each item: { path: str, tier: int, priority: int }
        """
        results = []
        
        # Task-based resolution
        if task and task in self.registry['task_mappings']:
            mapping = self.registry['task_mappings'][task]
            for std_path in mapping['standards']:
                results.append({
                    'path': std_path,
                    'tier': self._infer_tier(std_path),
                    'priority': mapping.get('priority', 100)
                })
        
        # Path-based resolution
        if file_path:
            for task_id, mapping in self.registry['task_mappings'].items():
                path_patterns = mapping['triggers'].get('path_patterns', [])
                for pattern in path_patterns:
                    if fnmatch.fnmatch(file_path, pattern):
                        for std_path in mapping['standards']:
                            if std_path not in [r['path'] for r in results]:
                                results.append({
                                    'path': std_path,
                                    'tier': self._infer_tier(std_path),
                                    'priority': mapping.get('priority', 100)
                                })
        
        # Sort by priority (lower = higher)
        results.sort(key=lambda x: x['priority'])
        return results
    
    def _infer_tier(self, path: str) -> int:
        if 'tier1' in path: return 1
        if 'tier2' in path: return 2
        if 'tier3' in path: return 3
        if 'tier4' in path: return 4
        return 0
```

### Directory Structure After Implementation

```
AgentQMS/
├── standards/
│   ├── registry.yaml (auto-generated, do not edit manually)
│   ├── schemas/
│   │   └── ads-header.json (validation schema)
│   ├── tier1-sst/
│   │   ├── naming-conventions.yaml (has ADS header)
│   │   ├── file-placement.yaml (has ADS header)
│   │   └── ...
│   ├── tier2-framework/
│   │   ├── coding/
│   │   │   ├── python-core.yaml (has ADS header)
│   │   │   └── anti-patterns.yaml (has ADS header)
│   │   └── ...
│   └── ...
├── tools/
│   ├── registry/
│   │   ├── sync_registry.py (compilation tool)
│   │   ├── migrate_legacy.py (migration tool)
│   │   └── resolver.py (resolution tool)
│   └── ...
└── ...
```

## Compliance and Validation

### Pre-Commit Hook Integration

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Validate ADS headers in modified standard files

MODIFIED_STANDARDS=$(git diff --cached --name-only | grep 'AgentQMS/standards/.*\.yaml$')

if [ -n "$MODIFIED_STANDARDS" ]; then
  echo "Validating ADS headers in modified standards..."
  
  for file in $MODIFIED_STANDARDS; do
    if ! uv run python AgentQMS/tools/registry/validate_header.py "$file"; then
      echo "ERROR: Invalid ADS header in $file"
      echo "Run: aqms registry validate $file"
      exit 1
    fi
  done
  
  echo "All ADS headers valid."
fi

exit 0
```

### CI Pipeline Checks

Add to `.github/workflows/validate-standards.yml`:

```yaml
name: Validate Standards

on: [pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Validate ADS Headers
        run: |
          uv run python AgentQMS/tools/registry/validate_all.py
      
      - name: Regenerate Registry
        run: |
          uv run python AgentQMS/tools/registry/sync_registry.py
      
      - name: Check Registry Sync
        run: |
          if git diff --exit-code AgentQMS/standards/registry.yaml; then
            echo "Registry is up to date"
          else
            echo "ERROR: Registry out of sync. Run: aqms registry sync"
            exit 1
          fi
```

### Agent Prompt Modifications

Add to AI agent system prompts:

```
When creating a new standard file in AgentQMS/standards/:

1. ALWAYS include ADS header at the top of the file
2. Required header fields:
   - ads_version: "1.0"
   - type: standard
   - tier: <1-4 based on directory>
   - name: "<human-readable name>"
   - description: "<what this standard governs>"
3. Optional but recommended:
   - triggers: Map of task_id to keywords/patterns/path_patterns
   - priority: Integer for resolution ordering
   - status: active (default)
4. After creating the file, run: aqms registry sync
5. Verify the new standard appears in registry.yaml task_mappings
```

### Backward Compatibility Strategy

**During Transition Period** (1-2 months):

1. **Dual System Operation**:
   - Keep manual registry.yaml as backup
   - Run sync_registry.py to generate registry-auto.yaml
   - Compare outputs to ensure equivalence
   - Agents can use either version during testing

2. **Gradual Migration**:
   - Week 1: Migrate Tier 1 standards (5 files)
   - Week 2: Migrate Tier 2 standards (30 files)
   - Week 3: Migrate Tier 3 standards (20 files)
   - Week 4: Migrate Tier 4 standards (16 files)
   - After each tier: Full regression testing

3. **Fallback Mechanism**:
   - If auto-generated registry fails validation
   - Fall back to manual registry
   - Log error and alert team
   - Block further automated updates until fixed

4. **Migration Validation**:
   - For each migrated file: Compare old task mappings to new
   - Ensure all triggers preserved
   - Verify agent queries return same standards pre/post migration
   - Test with real agent workflows (artifact creation, code review)

## Migration Strategy

### Phase 1: Infrastructure Setup (Week 1)

1. Create JSON schema at `AgentQMS/standards/schemas/ads-header.json`
2. Implement `sync_registry.py` tool with basic crawling
3. Implement `validate_header.py` tool for schema validation
4. Add resolution tool stub (returns empty list initially)
5. Create migration tool skeleton with dry-run mode
6. Test on isolated branch with 3 test standard files

**Success Criteria**: Tools execute without errors, validation catches malformed headers

### Phase 2: Pilot Migration (Week 2)

1. Select 5 representative standards from different tiers
2. Run migration tool in interactive mode
3. Manually review generated ADS headers
4. Run sync_registry.py to generate pilot registry
5. Compare pilot registry to current registry for these 5 files
6. Test agent queries against both registries
7. Fix discrepancies and refine migration logic

**Success Criteria**: 5 files migrated, agent behavior unchanged, registry equivalence verified

### Phase 3: Tier-by-Tier Migration (Weeks 3-4)

**Tier 1 (5 files)**:
- Run migration tool with --tier 1
- Review all headers manually (small count)
- Commit ADS headers to branch
- Run sync, verify registry section for tier 1

**Tier 2 (30 files)**:
- Run migration tool with --tier 2
- Spot-check 10 headers manually
- Review autogenerated triggers against current registry
- Commit and sync

**Tier 3 (20 files)**:
- Run migration tool with --tier 3
- Use --auto mode for files with obvious metadata
- Interactive mode for ambiguous files
- Commit and sync

**Tier 4 (16 files)**:
- Run migration tool with --tier 4
- Focus on workflow-specific trigger patterns
- Commit and sync

**After Each Tier**:
- Run full test suite
- Test artifact creation workflows
- Test agent queries for standards in this tier
- Compare agent-loaded standards pre/post migration

### Phase 4: Integration Testing (Week 5)

1. Full regression test suite across all agent types
2. Test edge cases (missing triggers, glob conflicts)
3. Simulate file relocations and verify auto-updates
4. Load test resolution tool (1000 queries)
5. Measure context reduction (tokens before/after)
6. Document any behavioral changes

### Phase 5: Deployment (Week 6)

1. Merge migration branch to main
2. Update all agent prompts to enforce ADS headers
3. Install pre-commit hook on all developer machines
4. Enable CI pipeline checks
5. Monitor for validation errors over 1 week
6. Document final migration status in CHANGELOG.md

**Rollback Plan**: Keep manual registry for 1 month. If critical issues arise, revert agents to use manual registry while fixing automated system.

## Edge Case Handling

### Standard Without Triggers

**Scenario**: ADS header has no `triggers` field or empty map.

**Resolution**:
- File still indexed in registry under "unassociated_standards" section
- Not included in any task mappings
- Only discoverable via explicit path query: `resolve_standards(file_path="tier2/file.yaml")`
- Warning logged during sync: "Standard X has no triggers, limited discoverability"

**Use Case**: Meta-standards (like registry spec itself) or deprecated standards

### Multiple Standards with Same Triggers

**Scenario**: Two standards both trigger on task="code_quality" with keywords ["python", "refactor"].

**Resolution**:
- Both standards returned by resolution query
- Ordered by priority field (lower number first)
- If priorities equal, alphabetical by path
- Agent receives both and must decide relevance
- No automatic deduplication (agent may need both)

### Glob Pattern Conflicts

**Scenario**: Standard A has path_pattern `**/*.py`, Standard B has `ocr/**/*.py`.

**Resolution**:
- Both match `ocr/models/vgg.py`
- Both returned by resolver
- Logged as INFO: "2 standards matched path X"
- Not an error—file may legitimately require multiple standards
- Agent context includes both

**Future Enhancement**: Add `exclude_patterns` field to filter out specific matches

### Registry Generation Failure Mid-Run

**Scenario**: sync_registry.py crashes after processing 40 of 71 files.

**Resolution**:
- Write to `.registry.yaml.tmp` throughout
- If process terminates, temp file remains
- Original registry.yaml untouched
- Next sync run starts fresh (idempotent)
- No partial registry corruption

**User Action**: Investigate error in logs, fix problematic standard file, re-run sync

### Schema Version Mismatch

**Scenario**: Standard has `ads_version: "2.0"`, schema validator is v1.0.

**Resolution**:
- Validation fails with: "Unsupported schema version 2.0, expected 1.0"
- Sync aborts
- Admin must update validator to support v2.0
- Migration path: Support multiple schema versions in validator

**Schema Evolution Process**:
1. Design ADS v2.0 schema with new required field
2. Update validator to accept both v1.0 and v2.0
3. Add migration tool: `upgrade_ads_version.py --from 1.0 --to 2.0`
4. Gradually upgrade standards over weeks
5. Deprecate v1.0 after 6 months

### Deleted Standard File

**Scenario**: Standard file deleted from filesystem but still referenced in registry.

**Resolution**:
- Sync detects file missing during crawl
- Removes from task mappings
- Logs: "Removed deleted standard: tier2/old-file.yaml"
- No error—registry self-heals
- Dead references automatically pruned

### Circular Trigger Dependencies

**Scenario**: Standard A triggers on task "code_review", which triggers Standard B, which triggers task "code_review".

**Resolution**:
- Cycle detection during sync:
  - Build dependency graph of triggers
  - Run cycle detection algorithm (DFS)
  - If cycle found: Fail sync with path description
- Error: "Cycle detected: code_review → standard_a → code_quality → standard_b → code_review"
- User must break cycle by removing one trigger link

**Prevention**: Triggers should reference task IDs, not other standards (no transitive triggers in v1.0)

### Case-Insensitive Filesystems (macOS/Windows)

**Scenario**: Standard path stored as `Tier2-Framework/file.yaml`, filesystem is `tier2-framework/file.yaml`.

**Resolution**:
- Normalize all paths to lowercase during sync
- Store lowercase paths in registry
- Resolver normalizes query paths before matching
- Warnings logged if case mismatches found
- Prevents duplicate entries for same file

### Priority Ties

**Scenario**: Three standards all have priority=10 for task "config_files".

**Resolution**:
- Resolver returns all three
- Secondary sort: By tier (lower tier = higher authority)
- Tertiary sort: Alphabetical by path
- Deterministic ordering guaranteed
- Agent receives consistent order across runs

## Integration Points

### 1. AgentQMS CLI Integration

Add commands to `AgentQMS/cli.py`:

```python
@cli.command()
def registry_sync():
    """Regenerate registry.yaml from ADS headers"""
    from AgentQMS.tools.registry import sync_registry
    sync_registry.run()
    click.echo("Registry synchronized successfully")

@cli.command()
@click.argument('file', required=False)
def registry_validate(file):
    """Validate ADS headers"""
    from AgentQMS.tools.registry import validate
    if file:
        validate.validate_file(file)
    else:
        validate.validate_all()

@cli.command()
def registry_migrate():
    """Migrate legacy standards to ADS headers"""
    from AgentQMS.tools.registry import migrate_legacy
    migrate_legacy.run_interactive()
```

**Usage**:
- `aqms registry sync` - Regenerate registry
- `aqms registry validate` - Validate all headers
- `aqms registry validate <file>` - Validate single file
- `aqms registry migrate` - Run migration tool

### 2. Project Compass Integration

Modify `project_compass/cli.py` pulse-export workflow:

```python
def pulse_export(pulse_id: str):
    # Existing artifact sync logic
    sync_artifacts_to_staging(pulse_id)
    
    # NEW: Sync registry if any standards changed
    if artifacts_include_standards():
        from AgentQMS.tools.registry import sync_registry
        click.echo("Standards changed, syncing registry...")
        sync_registry.run()
    
    # Existing git commit logic
    commit_pulse_artifacts(pulse_id)
```

**Trigger Condition**: If any files in `AgentQMS/standards/` modified during pulse, auto-sync registry before commit.

### 3. Artifact Creation Tool Integration

Modify `AgentQMS/tools/artifacts/create_artifact.py`:

```python
def create_standard_artifact(name: str, tier: int):
    # Existing artifact creation
    content = generate_standard_template(name)
    
    # NEW: Auto-generate ADS header
    header = {
        'ads_version': '1.0',
        'type': 'standard',
        'tier': tier,
        'name': name,
        'description': click.prompt('Description'),
        'triggers': {}
    }
    
    # Prompt for triggers
    if click.confirm('Add triggers?'):
        task_id = click.prompt('Task ID')
        keywords = click.prompt('Keywords (comma-separated)').split(',')
        header['triggers'][task_id] = {
            'keywords': [k.strip() for k in keywords],
            'priority': 100
        }
    
    # Write file with header
    file_content = f"---\n{yaml.dump(header)}---\n\n{content}"
    write_file(file_path, file_content)
    
    # Auto-sync registry
    click.echo("Syncing registry...")
    sync_registry.run()
```

**Enhancement**: Artifact creation wizard automatically includes ADS header prompts and syncs registry.

### 4. Agent Context Loading

Modify agent initialization in `ocr/core/infrastructure/agents/`:

```python
class BaseAgent:
    def load_standards_for_task(self, task: str, file_path: str = None):
        """Load applicable standards using resolver (not full registry)"""
        from AgentQMS.tools.resolution import StandardResolver
        
        resolver = StandardResolver()
        standards = resolver.resolve_standards(task=task, file_path=file_path)
        
        # Load only resolved standards
        for std in standards:
            self.context.append(self._load_standard_file(std['path']))
        
        logger.info(f"Loaded {len(standards)} standards for task {task}")
```

**Change**: Agents call resolver instead of loading full registry, reducing context by 80%.

### 5. Validation Tool Integration

Existing validation tools in `AgentQMS/tools/compliance/` should trigger registry validation:

```python
# In validate_artifacts.py
def validate_all_artifacts():
    # Existing artifact validation
    validate_artifact_names()
    validate_artifact_frontmatter()
    
    # NEW: Validate registry sync status
    if standards_modified_since_last_sync():
        raise ValidationError(
            "Standards modified but registry not synced. "
            "Run: aqms registry sync"
        )
```

**Check**: CI validation fails if registry out of sync with standard files.

## Dependencies and Assumptions

### Dependencies

- **Python 3.10+**: Type hints and pattern matching features used
- **PyYAML**: YAML parsing and generation
- **jsonschema**: ADS header validation against JSON schema
- **pathlib**: Cross-platform path handling
- **click**: CLI command interface
- **gitpython (optional)**: Git history analysis for migration

### Assumptions

1. **Single Registry File**: Assumes one master registry.yaml, not distributed tier-level indices (future enhancement)

2. **YAML Frontmatter Standard**: Assumes all standards use `---` delimited YAML frontmatter at file start (existing convention)

3. **Tier-Directory Alignment**: Assumes tier number in ADS header must match tier in directory path (enforced by validation)

4. **Git Repository**: Assumes standards are version-controlled with git for rollback capability

5. **Agent Modification Access**: Assumes ability to modify agent prompt templates and initialization code

6. **Sync Frequency**: Assumes registry sync runs after every standard change, not on a schedule (event-driven)

7. **No Runtime Registry Editing**: Assumes agents never modify registry.yaml directly, only via sync tool

8. **Glob Pattern Support**: Assumes Python fnmatch module sufficient for path matching (no advanced glob features)

9. **Single Writer**: Assumes only one sync process runs at a time (no concurrent registry writes)

10. **ASCII File Paths**: Assumes all standard file paths use ASCII characters (no Unicode path handling)

## Risk Assessment

### High Risk

**Risk**: Migration tool incorrectly infers triggers, causing agents to miss critical standards  
**Mitigation**: Manual review of all migrated triggers; comparison testing against current registry; pilot migration of 5 files first

**Risk**: Registry sync fails silently, leaving stale registry  
**Mitigation**: Atomic write with validation; exit codes for error detection; CI checks enforce sync status

### Medium Risk

**Risk**: Schema evolution breaks existing ADS headers  
**Mitigation**: Multi-version schema support in validator; gradual upgrade path; 6-month deprecation cycle

**Risk**: Glob pattern conflicts cause performance degradation  
**Mitigation**: Log warnings for overlapping patterns; document best practices; future optimization with pattern deduplication

### Low Risk

**Risk**: Case-sensitivity issues on different OSes  
**Mitigation**: Normalize all paths to lowercase; validate normalization in tests

**Risk**: Agent queries resolve too many standards  
**Mitigation**: Document best practices for specific triggers; add `exclude_patterns` in future version

## Success Metrics

### Quantitative Metrics

1. **Registry Automation**: 100% of registry.yaml generated from ADS headers (zero manual edits)
2. **Context Reduction**: Agent context for standard discovery reduced from 1000 to 200 tokens (80% reduction)
3. **Migration Coverage**: 71 of 71 standard files have valid ADS headers (100% coverage)
4. **Sync Performance**: Registry regeneration completes in <5 seconds (baseline: N/A)
5. **Validation Coverage**: 100% of invalid headers caught by pre-commit hook (zero malformed standards merged)
6. **Query Accuracy**: 95% of agent queries return complete standard sets (measured via test suite)

### Qualitative Metrics

1. **Developer Experience**: Zero reported incidents of manual registry updates after migration
2. **System Reliability**: Zero registry corruption events in 1 month of production use
3. **Refactoring Velocity**: File relocations require only ADS header updates, not registry edits
4. **Agent Performance**: No reported cases of agents using wrong standards due to stale registry

### Acceptance Criteria

- All 71 standards migrated with valid ADS headers
- Pre-commit hook blocks invalid headers
- CI pipeline enforces registry sync
- Agent context footprint reduced by 80%
- Zero manual registry edits for 30 days post-migration
- Resolution tool query time <100ms
- Documentation complete for all tools and workflows

