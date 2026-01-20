# Architecture Migration Analysis Tools Output

**Generated**: 2026-01-21  
**Purpose**: Support migration from deleted ocr/features/ to ocr/domains/

## Files in This Directory

### 1. dependency_graph_domains.txt
**Tool**: `dependency_graph` (AST analysis)  
**What it shows**: Import relationships within ocr/domains/  
**Use for**: 
- Understanding which modules depend on each other
- Identifying circular dependencies
- Planning migration order (fix leaves first, roots last)

### 2. context_tree_domains.txt
**Tool**: `context_tree` (semantic directory tree)  
**What it shows**: Structure of ocr/domains/ with key symbols  
**Use for**:
- Quick navigation of new architecture
- Finding where classes/functions are defined
- Understanding domain organization

### 3. symbol_search_DetectionHead.txt
**Tool**: `symbol_search` (AST symbol finder)  
**What it shows**: All locations where DetectionHead is defined/used  
**Use for**:
- Finding the original DetectionHead definition (in deleted files)
- Locating all files that import DetectionHead
- Determining if it exists in ocr/domains/ already

### 4. symbol_search_KIEDataset.txt
**Tool**: `symbol_search` (AST symbol finder)  
**What it shows**: All locations where KIEDataset is defined/used  
**Use for**:
- Similar to DetectionHead - finding definition and usage

### 5. broken_imports_full_list.txt
**Tool**: `grep` (manual search)  
**What it shows**: Complete list of all broken imports (48 total)  
**Use for**:
- Checklist of files that need fixing
- Batch processing with sed/awk if patterns are consistent

## How to Use These Tools

### Quick Win: Find Duplicates

If a class like DetectionHead exists in BOTH ocr/features (deleted) and ocr/domains/:

```bash
# Check if DetectionHead is in domains
cat symbol_search_DetectionHead.txt | grep "ocr/domains"

# If found, migration is just updating import paths
# If not found, need to extract from git history
```

### Understanding Dependencies

```bash
# See what imports what
cat dependency_graph_domains.txt

# Find circular imports (lines with <->)
grep "<->" dependency_graph_domains.txt
```

### Systematic Migration

1. Use broken_imports_full_list.txt as checklist
2. For each broken import:
   - Check symbol_search_*.txt to see if target exists in domains
   - Check dependency_graph to understand impact
   - Fix import or extract from git
3. Mark as done in MIGRATION_CHECKLIST.md

## Additional Commands

### Generate More Analysis

```bash
# Find all class definitions in domains
uv run adt symbol-search "class " ocr/domains/ --output analysis/all_classes.txt

# Check Hydra patterns
uv run adt find-hydra ocr/domains/ --output analysis/hydra_usage.txt

# Complexity report
uv run adt complexity ocr/domains/ --threshold 10 --output analysis/complexity.txt
```

### Real-time Checks

```bash
# Test if class exists
python3 -c "from ocr.domains.detection.models.heads.db_head import DBHead"

# Find where a class is defined
grep -r "class DetectionHead" ocr/

# Recover from git history
git show 89fe577^:ocr/features/detection/interfaces.py
```

## Next Steps

1. **Read symbol_search_DetectionHead.txt first** - Critical blocker
2. **Check dependency_graph_domains.txt** - Understand relationships
3. **Use broken_imports_full_list.txt** - Work through systematically
4. **Reference context_tree_domains.txt** - Navigate as you fix

## Key Insight

If analysis shows classes exist in ocr/domains/, migration is EASY:
- Just find/replace import paths
- Test each file after fixing

If classes are MISSING from ocr/domains/, migration is HARDER:
- Extract from git: `git show 89fe577^:ocr/features/path/to/file.py > new_location.py`
- Place in appropriate domain location
- Update imports
