# Artifact Index and Navigation Guide

**Generated**: 2026-01-22
**Source**: Conversation snippets from Hydra refactoring discussions
**Purpose**: AI-optimized knowledge base for systematic refactoring

---

## Overview

This directory contains AI-optimized artifacts extracted from conversation snippets about Hydra configuration refactoring, legacy import management, and systematic migration strategies.

---

## Directory Structure

```
artifacts/
├── ai_guidance/              # AI agent instruction patterns
├── implementation_guides/    # Complete implementation scripts
├── refactoring_patterns/     # Design patterns and antipatterns
├── tool_guides/             # Tool usage and mastery guides
└── analysis_outputs/        # Generated analysis results
```

---

## Artifact Categories

### 1. AI Guidance

**Purpose**: Systematic patterns for instructing AI agents during complex refactoring tasks

- [`instruction_patterns.md`](ai_guidance/instruction_patterns.md)
  - Multi-phased execution patterns
  - Machine-parseable target lists
  - Verification loop strategies
  - Tool-specific instruction templates
  - **Use when**: Instructing AI agents on large-scale refactoring (50+ changes)

---

### 2. Implementation Guides

**Purpose**: Complete, production-ready implementations of refactoring tools

- [`migration_guard_implementation.md`](implementation_guides/migration_guard_implementation.md)
  - Pre-execution validation scripts
  - Runtime assertion patterns
  - CI/CD integration examples
  - **Use when**: Setting up validation before training/testing

- [`auto_align_hydra_script.md`](implementation_guides/auto_align_hydra_script.md)
  - Automated Hydra target fixing using runtime reflection
  - ADT integration for symbol resolution
  - Batch processing strategies
  - **Use when**: Automatically fixing broken Hydra `_target_` paths

---

### 3. Refactoring Patterns

**Purpose**: Design patterns, antipatterns, and best practices for refactoring

- [`shim_antipatterns_guide.md`](refactoring_patterns/shim_antipatterns_guide.md)
  - Why backward compatibility shims become toxic
  - Validation layer alternatives
  - The "Alias" shim pattern (the only good shim)
  - **Use when**: Deciding how to handle legacy code during migration

- [`duplicate_file_detection.md`](refactoring_patterns/duplicate_file_detection.md)
  - Split-brain scenarios and duplicate file detection
  - Shadow import testing
  - The "Kill-Duplicate" plan
  - Doc-Sync audit tooling for AgentQMS
  - **Use when**: Debugging ghost code issues or preventing duplicate utilities

---

### 4. Tool Guides

**Purpose**: Mastery-level guides for refactoring tools

- [`yq_mastery_guide.md`](tool_guides/yq_mastery_guide.md)
  - Advanced yq techniques for YAML manipulation
  - Bulk target updates
  - Interpolation resolution
  - Hydra-specific patterns
  - **Use when**: Updating Hydra configurations programmatically

- [`adt_usage_patterns.md`](tool_guides/adt_usage_patterns.md)
  - AST-Grep structural patterns
  - Agent Debug Toolkit integration
  - Dependency analysis workflows
  - Structural linting rules
  - **Use when**: Performing structural code analysis and refactoring

---

## Quick Reference by Use Case

### Use Case: Fixing Broken Hydra Targets

1. **Validate environment**: [`migration_guard_implementation.md`](implementation_guides/migration_guard_implementation.md)
2. **Auto-fix targets**: [`auto_align_hydra_script.md`](implementation_guides/auto_align_hydra_script.md)
3. **Manual fixes**: [`yq_mastery_guide.md`](tool_guides/yq_mastery_guide.md)

### Use Case: Large-Scale Import Refactoring

1. **Analyze structure**: [`adt_usage_patterns.md`](tool_guides/adt_usage_patterns.md)
2. **Generate fix manifest**: [`instruction_patterns.md`](ai_guidance/instruction_patterns.md)
3. **Apply fixes**: [`adt_usage_patterns.md`](tool_guides/adt_usage_patterns.md) (AST-Grep rewrite)

### Use Case: Managing Legacy Code

1. **Understand antipatterns**: [`shim_antipatterns_guide.md`](refactoring_patterns/shim_antipatterns_guide.md)
2. **Implement validation**: [`migration_guard_implementation.md`](implementation_guides/migration_guard_implementation.md)
3. **Use alias pattern if needed**: [`shim_antipatterns_guide.md`](refactoring_patterns/shim_antipatterns_guide.md)

### Use Case: Detecting and Fixing Duplicate Files

1. **Identify duplicates**: [`duplicate_file_detection.md`](refactoring_patterns/duplicate_file_detection.md)
2. **Run shadow import test**: [`duplicate_file_detection.md`](refactoring_patterns/duplicate_file_detection.md)
3. **Consolidate and delete**: [`duplicate_file_detection.md`](refactoring_patterns/duplicate_file_detection.md)
4. **Prevent future duplicates**: [`duplicate_file_detection.md`](refactoring_patterns/duplicate_file_detection.md)

### Use Case: Instructing AI Agents

1. **Choose instruction style**: [`instruction_patterns.md`](ai_guidance/instruction_patterns.md)
2. **Prepare structured data**: [`instruction_patterns.md`](ai_guidance/instruction_patterns.md)
3. **Set up verification loop**: [`instruction_patterns.md`](ai_guidance/instruction_patterns.md)

---

## Tool Requirements

### Essential Tools

```bash
# YAML processor
brew install yq  # or download from github.com/mikefarah/yq

# AST-Grep (structural code search)
cargo install ast-grep
# or
brew install ast-grep

# Python environment manager
pip install uv
```

### Optional Tools

```bash
# Agent Debug Toolkit (if available)
# See project-specific installation instructions

# jq for JSON processing
brew install jq
```

---

## Common Workflows

### Workflow 1: Pre-Flight Validation

```bash
# Before any training/testing
uv run python scripts/audit/migration_guard.py

# Or use pre-flight script
bash scripts/preflight.sh
```

### Workflow 2: Automated Healing

```bash
# Generate audit
uv run python scripts/audit/master_audit.py > audit_results.txt

# Auto-fix (dry run first)
uv run python scripts/audit/auto_align_hydra.py --dry-run

# Apply fixes
uv run python scripts/audit/auto_align_hydra.py
```

### Workflow 3: Manual Target Updates

```bash
# Find broken targets
find configs/ -name "*.yaml" -exec yq '.. | select(has("_target_")) | ._target_' {} +

# Update specific target
yq -i '(.. | select(. == "old.path")) = "new.path"' config.yaml

# Verify
yq '.. | select(has("_target_"))' config.yaml
```

---

## Metadata and Traceability

### Source Attribution

All artifacts in this directory are derived from:
- **Source**: `conversation_snippets_suggestions.md`
- **Date**: 2026-01-22
- **Context**: Hydra configuration refactoring and legacy import management
- **Suggestion Level**: None (direct extraction and optimization)

### Naming Conventions

- **Lowercase with underscores**: `migration_guard_implementation.md`
- **Descriptive names**: Clearly indicate content and purpose
- **Consistent suffixes**:
  - `_guide.md`: Comprehensive guides
  - `_patterns.md`: Pattern collections
  - `_implementation.md`: Complete implementations
  - `_script.md`: Executable script documentation

### Python Environment

All Python code examples use `uv` as the package manager:
- Replace `python` with `uv run python`
- Replace `pip install` with `uv pip install`

---

## Integration with Existing Analysis

### Existing Analysis Outputs

The `analysis_outputs/` directory contains previously generated artifacts:

- `backward_compatibility_shims_technical_assessment.md`
- `broken_targets.json`
- `debugging_pain_points.md`
- `hydra_interpolation_map.md`
- `master_audit.md`
- `raw_analysis_output.txt`

These complement the new AI-optimized artifacts and can be used as:
- Input data for the tools described in this collection
- Validation references
- Historical context

---

## Best Practices

### For AI Agents

1. **Start with validation**: Always run migration guard before making changes
2. **Use structured data**: Prefer JSON/YAML manifests over prose
3. **Verify incrementally**: Check after each batch of changes
4. **Follow verification loops**: Audit → Fix → Verify → Repeat

### For Human Developers

1. **Read the guides**: Understand the patterns before applying
2. **Dry run first**: Test with `--dry-run` flags
3. **Backup before bulk changes**: Create backups or use version control
4. **Validate continuously**: Run audits frequently during refactoring

---

## Contributing

When adding new artifacts:

1. **Follow naming conventions**: Lowercase, underscores, descriptive
2. **Include metadata**: Source, date, context, python manager
3. **Add to index**: Update this file with new artifact information
4. **Cross-reference**: Link to related artifacts
5. **Provide examples**: Include usage examples and code snippets

---

## See Also

### External Resources

- [Hydra Documentation](https://hydra.cc/)
- [yq Documentation](https://mikefarah.gitbook.io/yq/)
- [AST-Grep Documentation](https://ast-grep.github.io/)

### Project Documentation

- Project Compass documentation (if applicable)
- AgentQMS standards (if applicable)
- OCR pipeline documentation (if applicable)
