# AST-Grep and Agent Debug Toolkit Usage Patterns

**Source**: Conversation analysis
**Date**: 2026-01-22
**Context**: Structural code analysis and refactoring with AST tools
**Python Manager**: uv

---

## Overview

AST-Grep (`sg`) and the Agent Debug Toolkit (ADT) provide structural code analysis capabilities that go beyond text-based search. This guide covers patterns for using these tools in refactoring workflows.

---

## AST-Grep Basics

### Installation

```bash
# Install ast-grep
cargo install ast-grep
# or
brew install ast-grep
```

### Core Concepts

- **Pattern Matching**: Structural patterns, not text patterns
- **Language Aware**: Understands Python/JavaScript/etc. syntax
- **Rewriting**: Can transform code structurally

---

## Common Patterns

### 1. Finding Function Calls

```bash
# Find all calls to a specific function
sg --pattern 'get_model_by_cfg($$$)' ocr/

# Find calls with specific arguments
sg --pattern 'hydra.utils.instantiate($CONF, $$$)' ocr/

# Find calls missing a parameter
sg --pattern 'get_model_by_cfg($$$)' \
   --not 'get_model_by_cfg($$$, _recursive_=False)' \
   ocr/
```

### 2. Structural Rewrites

```bash
# Add _recursive_=False to all instantiate calls
sg --pattern 'hydra.utils.instantiate($CONF, $$$)' \
   --rewrite 'hydra.utils.instantiate($CONF, _recursive_=False, $$$)' \
   ocr/

# Update import statements
sg --pattern 'from ocr.core.metrics import $NAME' \
   --rewrite 'from ocr.domains.detection.metrics import $NAME' \
   ocr/
```

### 3. Finding Imports

```bash
# Find all imports from a module
sg --pattern 'from ocr.core.metrics import $$$' ocr/

# Find specific import patterns
sg --pattern 'import $MODULE' ocr/ | grep "ocr.detection"
```

---

## Agent Debug Toolkit (ADT) Integration

### Intelligent Search

The ADT's `intelligent-search` combines multiple search strategies:

```bash
# Find where a symbol is defined
adt intelligent-search "CLEvalMetric"

# Get qualified path for Hydra configs
adt intelligent-search "CLEvalMetric" --output json | jq '.qualified_path'

# Search with context
adt intelligent-search "correct_perspective_from_mask" --context ocr.domains
```

### Dependency Analysis

```bash
# Analyze import dependencies
adt analyze-dependencies ocr/

# Find circular dependencies
adt analyze-dependencies ocr/ --check-circular

# Export to JSON for processing
adt analyze-dependencies ocr/ --output json > deps.json
```

### Context Tree

```bash
# Generate project structure tree
adt context-tree ocr/

# Save as baseline for comparison
adt context-tree ocr/ > truth_v1.json

# After refactor, compare
adt context-tree ocr/ > truth_v2.json
diff truth_v1.json truth_v2.json
```

---

## Hydra-Specific Patterns

### 1. Detecting Recursive Instantiation Traps

Create a custom rule file:

```yaml
# rules/hydra-recursion.yaml
id: missing-recursive-false
language: python
rule:
  pattern: hydra.utils.instantiate($CONF, $$$)
  not:
    pattern: hydra.utils.instantiate($CONF, _recursive_=False, $$$)
message: "Potential Recursive Instantiation Trap! Ensure _recursive_=False for model factories."
severity: warning
```

Run the lint:

```bash
sg lint --rule rules/hydra-recursion.yaml ocr/
```

### 2. Finding Model Factory Calls

```bash
# Find all get_model_by_cfg calls
sg --pattern 'get_model_by_cfg($$$)' ocr/

# Find calls that might need _recursive_=False
sg --pattern 'get_model_by_cfg($CFG)' \
   --not 'get_model_by_cfg($CFG, _recursive_=False)' \
   ocr/
```

### 3. Validating Import Patterns

```bash
# Find legacy import patterns
sg --pattern 'from ocr.detection import $$$' ocr/

# Should be: from ocr.domains.detection import $$$
sg --pattern 'from ocr.domains.detection import $$$' ocr/
```

---

## Advanced Workflows

### 1. Path-to-Symbol Reverse Lookup

Combine ADT with yq for Hydra config updates:

```bash
# 1. Find where the function actually lives now
adt intelligent-search "correct_perspective_from_mask" --output json > result.json

# 2. Extract the qualified path
NEW_PATH=$(cat result.json | jq -r '.qualified_path')

# 3. Update all Hydra configs automatically
find configs/ -name "*.yaml" | xargs -I {} yq -i \
  "(.. | select(. | test(\".*perspective_correction.*\"))) = \"$NEW_PATH\"" {}
```

### 2. Pre-computation of Truth

Before refactoring:

```bash
# Save current state
adt context-tree ocr/ --output json > pre_refactor_truth.json
adt analyze-dependencies ocr/ --output json > pre_refactor_deps.json
```

After refactoring:

```bash
# Save new state
adt context-tree ocr/ --output json > post_refactor_truth.json
adt analyze-dependencies ocr/ --output json > post_refactor_deps.json

# Generate migration guide from delta
python scripts/generate_migration_guide.py \
  pre_refactor_truth.json \
  post_refactor_truth.json \
  > migration_guide.md
```

### 3. Automated Fix Generation

```python
# scripts/generate_fixes.py
import subprocess
import json

def find_new_location(symbol_name):
    """Use ADT to find where a symbol is now."""
    result = subprocess.run(
        ['adt', 'intelligent-search', symbol_name, '--output', 'json'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return data.get('qualified_path')
    return None

def generate_sg_rewrite(old_path, new_path):
    """Generate ast-grep rewrite command."""
    return f"sg --pattern 'from {old_path} import $NAME' --rewrite 'from {new_path} import $NAME' ocr/"

# Usage
old_module = "ocr.core.metrics"
symbol = "CLEvalMetric"

new_location = find_new_location(symbol)
if new_location:
    new_module = '.'.join(new_location.split('.')[:-1])
    cmd = generate_sg_rewrite(old_module, new_module)
    print(cmd)
```

---

## Detecting "Shadow Modules"

Find directories with `__init__.py` but no actual code:

```bash
# Find empty packages (potential zombies)
find ocr/ -name "__init__.py" -exec dirname {} \; | \
  xargs -I {} sh -c 'ls {}/*.py 2>/dev/null | wc -l | xargs -I % [ % -le 1 ] && echo "Empty Package: {}"'
```

Using ADT:

```bash
# Find modules with no exports
adt analyze-modules ocr/ --find-empty
```

---

## Structural Linting Rules

### Rule: No Cross-Domain Imports

```yaml
# rules/no-cross-domain.yaml
id: no-cross-domain-imports
language: python
rule:
  any:
    - pattern: from ocr.domains.detection import $$$
      inside:
        kind: module
        has:
          pattern: ocr.domains.recognition
    - pattern: from ocr.domains.recognition import $$$
      inside:
        kind: module
        has:
          pattern: ocr.domains.detection
message: "Cross-domain imports are not allowed. Use ocr.core for shared functionality."
severity: error
```

### Rule: Require Absolute Imports

```yaml
# rules/absolute-imports.yaml
id: require-absolute-imports
language: python
rule:
  pattern: from . import $$$
message: "Relative imports are discouraged. Use absolute imports from ocr.*"
severity: warning
```

---

## Integration with Migration Guard

```python
# scripts/audit/migration_guard.py (enhanced)
import subprocess
import json

def validate_with_ast_grep():
    """Use ast-grep to validate code patterns."""
    print("🔍 Running structural validation...")

    # Check for Hydra recursion traps
    result = subprocess.run(
        ['sg', 'lint', '--rule', 'rules/hydra-recursion.yaml', 'ocr/'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("⚠️  Found potential Hydra recursion traps")
        print(result.stdout)
        return False

    # Check for cross-domain imports
    result = subprocess.run(
        ['sg', 'lint', '--rule', 'rules/no-cross-domain.yaml', 'ocr/'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Found unauthorized cross-domain imports")
        print(result.stdout)
        return False

    print("✅ Structural validation passed")
    return True
```

---

## Bulk Refactoring Workflow

### Step 1: Analyze

```bash
# Find all broken imports
adt analyze-dependencies ocr/ --find-broken > broken_imports.json
```

### Step 2: Generate Fixes

```bash
# For each broken import, find new location
cat broken_imports.json | jq -r '.[]' | while read import; do
  symbol=$(echo $import | awk -F. '{print $NF}')
  new_path=$(adt intelligent-search "$symbol" --output json | jq -r '.qualified_path')
  echo "$import -> $new_path"
done > import_mapping.txt
```

### Step 3: Apply Fixes

```bash
# Generate and execute sg commands
cat import_mapping.txt | while IFS=' -> ' read old new; do
  old_module=$(echo $old | rev | cut -d. -f2- | rev)
  new_module=$(echo $new | rev | cut -d. -f2- | rev)

  sg --pattern "from $old_module import \$NAME" \
     --rewrite "from $new_module import \$NAME" \
     ocr/ --apply
done
```

### Step 4: Verify

```bash
# Re-run analysis
adt analyze-dependencies ocr/ --find-broken

# Should return empty or significantly reduced list
```

---

## Pro Tips

### 1. Combine with grep for Context

```bash
# Find pattern and show surrounding context
sg --pattern 'get_model_by_cfg($$$)' ocr/ | \
  while read file; do
    echo "=== $file ==="
    grep -A 3 -B 3 "get_model_by_cfg" "$file"
  done
```

### 2. JSON Output for Scripting

```bash
# Get structured output
sg --pattern 'hydra.utils.instantiate($$$)' ocr/ --json | \
  jq '.matches[] | {file: .file, line: .line}'
```

### 3. Incremental Validation

```bash
# After each fix, verify
fix_one_import() {
  sg --pattern "from ocr.core.metrics import CLEvalMetric" \
     --rewrite "from ocr.domains.detection.metrics import CLEvalMetric" \
     ocr/ --apply

  # Immediate verification
  uv run python -c "from ocr.domains.detection.metrics import CLEvalMetric" && \
    echo "✅ Import works" || \
    echo "❌ Import broken"
}
```

---

## Troubleshooting

### Pattern Doesn't Match

```bash
# Debug: Show AST structure
sg --pattern '$$$' file.py --debug-query

# Try more specific pattern
sg --pattern 'def $FUNC($$$): $$$' file.py
```

### Rewrite Produces Invalid Code

```bash
# Test rewrite without applying
sg --pattern 'old($$$)' --rewrite 'new($$$)' file.py

# Review before applying
sg --pattern 'old($$$)' --rewrite 'new($$$)' file.py --apply --interactive
```

---

## See Also

- `yq_mastery_guide.md` - YAML manipulation for Hydra configs
- `migration_guard_implementation.md` - Integration with validation
- `instruction_patterns.md` - Using in AI agent workflows
