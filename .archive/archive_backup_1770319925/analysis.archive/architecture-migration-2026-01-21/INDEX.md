# Architecture Migration Analysis - Quick Index

## 📁 Files Generated

| File | Size | Tool | Priority |
|------|------|------|----------|
| **broken_imports_full_list.txt** | 48 lines | grep | ⚠️ HIGH - Start here |
| **dependency_graph_domains.txt** | 313 findings | AST | ⚠️ HIGH - Understand structure |
| **context_tree_domains.txt** | 113 items | AST | 📖 Reference |
| **symbol_search_DetectionHead.txt** | 0 results | AST | ❌ MISSING - Need to extract |
| **symbol_search_KIEDataset.txt** | 0 results | AST | ❌ MISSING - Need to extract |

## 🎯 Critical Findings

### 1. DetectionHead Is MISSING ❌
- Imported by: `ocr/domains/detection/models/heads/db_head.py`
- **Action**: Must extract from git history
```bash
git show 89fe577^:ocr/features/detection/interfaces.py > ocr/domains/detection/interfaces.py
```

### 2. All Missing Classes Must Be Extracted
From broken_imports_full_list.txt, these don't exist in ocr/domains/:
- **DetectionHead** (interfaces.py)
- **CraftDecoder, CraftVGGEncoder, CraftHead** (from craft components)
- **DBPPDecoder, DBHead** (from DB components)
- **CRAFTPostProcessor, DBPostProcessor** (postprocessors)
- **KIEDataItem** (validation schema)

### 3. Import Pattern Analysis
Most broken imports follow pattern:
```python
from ocr.features.X.Y.Z import Class
```

Should become:
```python
from ocr.domains.X.Y.Z import Class
```

BUT many classes don't exist in domains yet!

## 🚀 Recommended Workflow

### Step 1: Extract Missing Interface (5 min)
```bash
# Get the base interface
git show 89fe577^:ocr/features/detection/interfaces.py > /tmp/interfaces.py
cat /tmp/interfaces.py

# Create in correct location
mkdir -p ocr/domains/detection
mv /tmp/interfaces.py ocr/domains/detection/interfaces.py
```

### Step 2: Check What Exists in Domains (5 min)
```bash
# Compare old vs new structure
ls -la ocr/domains/detection/models/heads/
ls -la ocr/domains/detection/models/decoders/

# See what's already there
cat dependency_graph_domains.txt | grep "class "
```

### Step 3: Extract Missing Components (30 min)
For each missing class:
1. Check if it exists in ocr/domains (use context_tree)
2. If not, extract from git: `git show 89fe577^:ocr/features/path/to/file.py`
3. Place in ocr/domains/
4. Update imports

### Step 4: Update Imports (20 min)
Use broken_imports_full_list.txt as checklist:
```bash
# For each file, update imports
# Example:
# from ocr.features.detection.interfaces import DetectionHead
#   →
# from ocr.domains.detection.interfaces import DetectionHead
```

### Step 5: Test (10 min)
```bash
python3 -c "from ocr.domains.detection.models.heads.db_head import DBHead"
pytest tests/unit/test_head.py -v
```

## 📊 Statistics

- **Total broken imports**: 48
- **Unique files affected**: 12 (ocr/) + 18 (tests/) + 1 (scripts/)
- **Missing classes**: ~10-15 (need git extraction)
- **Estimated fix time**: 2-3 hours

## 🔧 Helper Commands

```bash
# Find where a class SHOULD be
cat context_tree_domains.txt | grep -i "head"

# Check import in a file
grep "from ocr.features" ocr/domains/detection/models/heads/db_head.py

# Extract from git
git show 89fe577^:ocr/features/detection/interfaces.py

# Test import
python3 -c "from ocr.domains.detection.interfaces import DetectionHead"

# Count remaining broken imports
grep -r "from ocr.features" ocr/ tests/ scripts/ 2>/dev/null | wc -l
```

## 📚 Related Documents

- [SESSION_HANDOVER_ARCHITECTURE_PURGE.md](../../SESSION_HANDOVER_ARCHITECTURE_PURGE.md) - Full context
- [MIGRATION_CHECKLIST.md](../../MIGRATION_CHECKLIST.md) - Task tracking
- [PIPELINE_ARCHITECTURE_TRUTH.md](../../PIPELINE_ARCHITECTURE_TRUTH.md) - Root cause analysis
- [README.md](README.md) - Tool descriptions

## ⚡ Quick Wins

1. **DetectionHead interface** - Single file extraction unblocks 2 files
2. **Postprocessors** - Already in domains, just need import path updates
3. **Test files** - Simple find/replace in 18 files
