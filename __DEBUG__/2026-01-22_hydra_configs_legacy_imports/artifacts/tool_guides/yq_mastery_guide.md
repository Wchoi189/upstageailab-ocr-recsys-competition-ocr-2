# YQ Mastery Guide for Hydra Configuration Management

**Source**: Conversation analysis
**Date**: 2026-01-22
**Context**: Advanced yq techniques for Hydra config refactoring
**Python Manager**: uv

---

## Overview

`yq` is a powerful YAML processor that enables surgical updates to Hydra configurations. This guide covers advanced techniques for bulk updates, interpolation resolution, and structural validation.

---

## Installation

```bash
# Install yq (YAML processor)
# Note: Different from python-yq
brew install yq  # macOS
# or
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq
chmod +x /usr/local/bin/yq
```

---

## Basic Operations

### Reading Values

```bash
# Read a single key
yq '.data.dataset_path' configs/data/default.yaml

# Read nested keys
yq '.model.architecture.encoder.type' configs/model/base.yaml

# List all keys at root level
yq 'keys' configs/config.yaml

# Get all string values
yq '.. | select(tag == "!!str")' configs/data/default.yaml
```

### Writing Values

```bash
# Update a single value
yq -i '.model.name = "new_model"' config.yaml

# Update nested value
yq -i '.model.architecture.layers = 12' config.yaml

# Add new key
yq -i '.new_key = "value"' config.yaml
```

---

## Advanced Techniques

### 1. Bulk Target Updates

Fix all Hydra `_target_` references that moved during refactoring:

```bash
# Update all occurrences of a specific target
find configs/ -name "*.yaml" | xargs -I {} yq -i \
  '(.[] | select(has("_target_")) | select(._target_ == "old.path.Class") | ._target_) = "new.path.Class"' {}

# More readable version with variable
OLD_TARGET="ocr.core.metrics.CLEvalMetric"
NEW_TARGET="ocr.domains.detection.metrics.CLEvalMetric"

find configs/ -name "*.yaml" | xargs -I {} yq -i \
  "(.. | select(. == \"$OLD_TARGET\")) = \"$NEW_TARGET\"" {}
```

### 2. Selective Updates (Smart Update)

Update only specific broken keys, not all matching strings:

```bash
# Update only if the value exactly matches
yq -i '.. | select(. == "${data.dataset_path}.ValidatedOCRDataset") = "ocr.data.datasets.base.ValidatedOCRDataset"' \
  configs/data/datasets/craft.yaml

# Update specific key path
yq -i '.datasets.train_dataset._target_ = "new.target.path"' config.yaml
```

### 3. Interpolation Resolution

Hydra uses `${variable}` syntax for interpolation. Here's how to resolve them:

```bash
# Find what a variable is defined as
export DATA_PATH=$(yq '.data.dataset_path' configs/data/default.yaml)

# Search for configs using that variable and show resolved value
find configs/ -name "*.yaml" | xargs -I {} yq \
  ".datasets.train_dataset._target_ | sub(\"\\${data.dataset_path}\", \"$DATA_PATH\")" {}

# Recursive search for variable definition
yq '.. | select(has("dataset_path")) | .dataset_path' configs/data/default.yaml
```

### 4. Finding All Targets

Extract all `_target_` values across your config tree:

```bash
# Find all _target_ keys
find configs/ -name "*.yaml" -exec yq '.. | select(has("_target_")) | ._target_' {} +

# Get unique targets only
find configs/ -name "*.yaml" -exec yq '.. | select(has("_target_")) | ._target_' {} + | sort -u

# Find targets with their source files
find configs/ -name "*.yaml" -exec sh -c \
  'echo "File: $1"; yq ".. | select(has(\"_target_\")) | ._target_" "$1"' _ {} \;
```

---

## Hydra-Specific Patterns

### Detecting Dynamic Targets

```bash
# Find all interpolated targets (containing ${})
find configs/ -name "*.yaml" -exec yq \
  '.. | select(has("_target_")) | select(._target_ | test("\\$\\{")) | ._target_' {} +

# Find configs with interpolation variables
yq '.. | select(. | type == "!!str") | select(test("\\$\\{"))' config.yaml
```

### Validating Target Paths

```bash
# Check if all targets follow expected pattern
find configs/ -name "*.yaml" -exec yq \
  '.. | select(has("_target_")) | select(._target_ | test("^ocr\\.")) | ._target_' {} +

# Find legacy targets (old naming convention)
find configs/ -name "*.yaml" -exec yq \
  '.. | select(has("_target_")) | select(._target_ | test("^ocr\\.detection\\.")) | ._target_' {} +
```

### Checking for Recursive Instantiation

```bash
# Find all configs that might need _recursive_=False
find configs/ -name "*.yaml" -exec sh -c \
  'yq ".. | select(has(\"_target_\")) | select(has(\"_recursive_\") | not)" "$1" && echo "File: $1"' _ {} \;
```

---

## Debugging Techniques

### 1. Root Key Discovery

When `yq '.data'` returns nothing:

```bash
# Check what keys exist at root
yq 'keys' configs/data/default.yaml

# If keys are directly at root (no .data wrapper)
yq '.dataset_path' configs/data/default.yaml  # Instead of .data.dataset_path

# Get all top-level string values
yq '. | to_entries | .[] | select(.value | type == "!!str")' config.yaml
```

### 2. Structure Inspection

```bash
# See full structure
yq '.' config.yaml

# See structure with types
yq '.. | type' config.yaml

# Find depth of nesting
yq '.. | path | length' config.yaml | sort -n | tail -1
```

### 3. Validation After Updates

```bash
# Verify a specific target was updated
yq '.. | select(has("_target_")) | ._target_' config.yaml | grep "new.path"

# Check for any remaining old paths
yq '.. | select(. == "old.path.Class")' config.yaml

# Validate YAML is still valid after update
yq '.' config.yaml > /dev/null && echo "Valid YAML" || echo "Invalid YAML"
```

---

## Integration with ADT (Agent Debug Toolkit)

### Combined Workflow

```bash
# 1. Find where a symbol actually lives
NEW_PATH=$(adt intelligent-search "correct_perspective_from_mask" --output json | yq '.qualified_path')

# 2. Update all Hydra configs automatically
find configs/ -name "*.yaml" | xargs -I {} yq -i \
  "(.. | select(. | test(\".*perspective_correction.*\"))) = \"$NEW_PATH\"" {}
```

### Automated Fix Generation

```bash
# Generate fix commands from audit results
adt analyze-dependencies ocr/ --output json | \
  jq -r '.broken_imports[] |
    "yq -i \"(.. | select(. == \"\(.old_path)\")) = \"\(.new_path)\"\" \(.config_file)"'
```

---

## Common Patterns

### Pattern 1: Fix All Datasets

```bash
# Update all dataset _target_ paths
find configs/data/datasets/ -name "*.yaml" | xargs -I {} yq -i \
  '(.datasets | .. | select(has("_target_")) | ._target_) |= sub("^ocr\\.data\\.", "ocr.data.datasets.")' {}
```

### Pattern 2: Add Missing _recursive_ Flag

```bash
# Add _recursive_=False to all model configs
find configs/model/ -name "*.yaml" | xargs -I {} yq -i \
  '(.model | select(has("_target_")) | ._recursive_) = false' {}
```

### Pattern 3: Detect Zombie Configs

```bash
# Find YAML files not referenced in main config
for f in $(find configs -name "*.yaml"); do
  filename=$(basename $f .yaml)
  grep -q "$filename" configs/config.yaml || echo "Potential Zombie Config: $f"
done
```

---

## Pro Tips

### 1. Variable Storage

Store intermediate results for complex operations:

```bash
# Capture definition
export DATA_PATH=$(yq '.data.dataset_path' configs/data/default.yaml)

# Use in multiple operations
yq ".train._target_ | sub(\"\\${data.dataset_path}\", \"$DATA_PATH\")" config1.yaml
yq ".val._target_ | sub(\"\\${data.dataset_path}\", \"$DATA_PATH\")" config2.yaml
```

### 2. Dry Run Testing

```bash
# Test without modifying (remove -i flag)
yq '(.model._target_) = "new.path"' config.yaml

# Compare before/after
yq '.' config.yaml > before.yaml
yq -i '(.model._target_) = "new.path"' config.yaml
diff before.yaml config.yaml
```

### 3. Batch Processing with Logging

```bash
# Log all changes
for file in configs/**/*.yaml; do
  echo "Processing: $file" >> yq_changes.log
  yq -i '(.[] | select(has("_target_")) | select(._target_ == "old") | ._target_) = "new"' "$file" \
    && echo "  ✓ Updated" >> yq_changes.log \
    || echo "  ✗ Failed" >> yq_changes.log
done
```

---

## Troubleshooting

### Issue: yq returns nothing

**Cause**: Key doesn't exist or structure is different
**Solution**:
```bash
# Check structure first
yq 'keys' config.yaml
yq '.' config.yaml | head -20
```

### Issue: Update doesn't work

**Cause**: Selector doesn't match
**Solution**:
```bash
# Test selector first (without -i)
yq '.. | select(. == "target_value")' config.yaml

# Verify match exists
yq '.. | select(. == "target_value")' config.yaml | wc -l
```

### Issue: Breaks YAML syntax

**Cause**: Special characters in value
**Solution**:
```bash
# Use proper quoting
yq -i '.key = "value with spaces"' config.yaml

# Escape special characters
yq -i '.key = "value with \"quotes\""' config.yaml
```

---

## Complete Example: Fixing Broken Targets

```bash
#!/bin/bash
# fix_hydra_targets.sh

# Configuration
OLD_TARGET="ocr.core.metrics.CLEvalMetric"
NEW_TARGET="ocr.domains.detection.metrics.CLEvalMetric"
CONFIG_DIR="configs"

echo "🔍 Finding configs with target: $OLD_TARGET"

# Find affected files
AFFECTED_FILES=$(find "$CONFIG_DIR" -name "*.yaml" -exec \
  grep -l "$OLD_TARGET" {} \;)

if [ -z "$AFFECTED_FILES" ]; then
  echo "✅ No files found with old target"
  exit 0
fi

echo "📝 Found $(echo "$AFFECTED_FILES" | wc -l) files to update"

# Update each file
for file in $AFFECTED_FILES; do
  echo "  Updating: $file"

  # Backup
  cp "$file" "$file.bak"

  # Update
  yq -i "(.. | select(. == \"$OLD_TARGET\")) = \"$NEW_TARGET\"" "$file"

  # Verify
  if yq '.' "$file" > /dev/null 2>&1; then
    echo "    ✓ Success"
    rm "$file.bak"
  else
    echo "    ✗ Failed - restoring backup"
    mv "$file.bak" "$file"
  fi
done

echo "✅ Update complete"
```

---

## See Also

- `adt_usage_patterns.md` - Combining yq with Agent Debug Toolkit
- `hydra_target_validation.md` - Validating Hydra configurations
- `instruction_patterns.md` - Using yq in AI agent instructions
