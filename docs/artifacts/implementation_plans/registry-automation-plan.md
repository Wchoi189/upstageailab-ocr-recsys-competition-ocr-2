---
ads_version: "1.0"
type: implementation_plan
tier: 4
priority: high
status: active
date: "2026-01-26"
feature_branch: "001-registry-automation"
spec_file: "specs/001-registry-automation/spec.md"
estimated_duration: "6 weeks"
---

# Implementation Plan: Automated Registry Generation System

## Executive Summary

Transform the 440-line manually-maintained `registry.yaml` into an auto-generated artifact compiled from distributed ADS (Agent Discovery System) headers embedded in each standard file. This eliminates synchronization errors, reduces agent context burden by 80%, and enables safe standards refactoring.

**Spec Reference**: [specs/001-registry-automation/spec.md](../../specs/001-registry-automation/spec.md)

## Success Criteria

- ✅ `registry.yaml` becomes 100% auto-generated (zero manual edits)
- ✅ Agent context footprint reduced from 1000 tokens → 200 tokens (80% reduction)
- ✅ All 71 existing standards retrofitted with valid ADS v2.0 headers
- ✅ Registry sync completes in <5 seconds
- ✅ Resolution queries complete in <100ms (with caching: <10ms)
- ✅ Pre-commit validation prevents malformed standards
- ✅ Zero registry corruption events during migration
- ✅ **SC-011**: Zero untracked standard files in standards directory (strict enforcement)
- ✅ Circular dependencies detected and blocked at compile time
- ✅ Parity verification: 100% of legacy triggers migrated correctly

## Implementation Phases

### Phase 0: Nuclear Archive (Pre-Week 1)

**Goal**: Clear the workspace and prepare for clean rebuild

**Critical Action**: Archive all existing standard files to establish clean baseline

```bash
# Archive all legacy files
mkdir -p AgentQMS/standards/_archive
mv AgentQMS/standards/tier*/*.yaml AgentQMS/standards/_archive/

# Backup current registry
cp AgentQMS/standards/registry.yaml AgentQMS/standards/_archive/registry.yaml.backup
```

**Protocol**: No file leaves `_archive/` unless it passes v2.0 strict validation

**Rationale**: This "nuclear" approach ensures:
- 100% compliance from day one
- No legacy debt carried forward
- Clean testing environment
- Safe rollback path (archive remains intact)

**Acceptance**: All 71 standards archived, workspace clean, backup created

---

### Phase 1: Foundation & Schema (Week 1-2)

**Goal**: Create the enhanced ADS v2.0 header schema and intelligent registry compiler infrastructure

#### Task 1.1: Create ADS Header JSON Schema
**Effort**: 4 hours  
**Files to Create**:
- `AgentQMS/standards/schemas/ads-header.json`

**Implementation** (Enhanced v2.0 Schema):

**Note**: This schema includes critical enhancements: `id` for dependency mapping, `dependencies` for graph building, and `fuzzy_threshold` for intelligent search.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AgentQMS ADS Header v2.0 (Enhanced)",
  "type": "object",
  "required": ["ads_version", "id", "name", "description", "tier", "triggers"],
  "properties": {
    "ads_version": {
      "type": "string",
      "const": "2.0",
      "description": "ADS schema version (v2.0 with dependencies)"
    },
    "id": {
      "type": "string",
      "pattern": "^[a-z0-9-]+$",
      "minLength": 3,
      "maxLength": 50,
      "description": "Unique kebab-case identifier for dependency mapping (e.g., 'python-core', 'naming-conventions')"
    },
    "name": {
      "type": "string",
      "minLength": 3,
      "maxLength": 100,
      "description": "Human-readable standard name"
    },
    "description": {
      "type": "string",
      "minLength": 10,
      "maxLength": 500,
      "description": "What this standard governs"
    },
    "tier": {
      "type": "integer",
      "enum": [1, 2, 3, 4],
      "description": "Functional tier (1:SST, 2:Framework, 3:Agents, 4:Workflows)"
    },
    "type": {
      "type": "string",
      "enum": ["principle", "standard", "template", "schema"],
      "description": "Document classification"
    },
    "priority": {
      "type": "string",
      "enum": ["critical", "high", "medium", "low"],
      "default": "medium"
    },
    "status": {
      "type": "string",
      "enum": ["active", "deprecated", "experimental"],
      "default": "active"
    },
    "dependencies": {
      "type": "array",
      "items": {
        "type": "string",
        "pattern": "^[a-z0-9-]+$"
      },
      "uniqueItems": true,
      "description": "List of standard IDs that MUST be loaded with this one (e.g., Tier 2 depends on Tier 1 constitutional laws)"
    },
    "fuzzy_threshold": {
      "type": "integer",
      "minimum": 0,
      "maximum": 100,
      "default": 80,
      "description": "Levenshtein similarity threshold for fuzzy keyword matching (0-100, default 80)"
    },
    "triggers": {
      "type": "object",
      "description": "Task activation criteria",
      "minProperties": 1,
      "additionalProperties": {
        "type": "object",
        "properties": {
          "priority": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "default": 2
          },
          "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
          },
          "path_patterns": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
          }
        },
        "anyOf": [
          {"required": ["keywords"]},
          {"required": ["path_patterns"]}
        ]
      }
    }
  }
}
```

**Tests**:
- Validate schema itself is valid JSON Schema Draft 7
- Test validation with valid v2.0 ADS header including id, dependencies (should pass)
- Test validation with missing required fields (should fail with clear errors)
- Test validation with invalid tier value (should reject)
- Test validation with invalid id format (spaces, uppercase, etc.) (should reject)
- Test validation with circular dependencies reference (should be caught by compiler, not schema)
- Test validation with invalid fuzzy_threshold (<0 or >100) (should reject)

**Acceptance**: Schema file validates, all test cases pass/fail correctly, v2.0 fields enforce stricter constraints

---

#### Task 1.2: Build Intelligent Registry Compiler
**Effort**: 18 hours (expanded for enhanced features)  
**Files to Create**:
- `AgentQMS/tools/sync_registry.py`

**Key Enhancements**:
- ✅ **Strict Mode**: Fail build if any non-archived .yaml lacks valid ADS header
- ✅ **Cycle Detection**: Build dependency DAG, detect circular references
- ✅ **Visual Graph**: Generate `architecture_map.dot` for visual inspection
- ✅ **Pulse Delta**: Print semantic diff showing what changed
- ✅ **Atomic Write**: Use temp file → rename to prevent corruption
- ✅ **Header Enforcement**: Write "AUTO-GENERATED - DO NOT EDIT" comment

**Implementation** (Example - adjust as needed):

```python
"""
AgentQMS Registry Compiler
Generates registry.yaml from distributed ADS headers
"""
import yaml
import jsonschema
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class RegistryCompiler:
    """Compiles registry.yaml from ADS headers in standard files"""
    
    def __init__(self, root_dir: Path = Path("AgentQMS/standards"), strict_mode: bool = True):
        self.root = root_dir
        self.registry_path = self.root / "registry.yaml"
        self.schema_path = self.root / "schemas/ads-header.json"
        self.archive_dir = "_archive"
        self.strict_mode = strict_mode
        self.schema = self._load_schema()
        self.errors = []
        self.warnings = []
        self.standards_map: Dict[str, Dict[str, Any]] = {}  # id -> standard data
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load and validate ADS header schema"""
        with open(self.schema_path) as f:
            return yaml.safe_load(f)
    
    def _extract_ads_header(self, file_path: Path) -> Dict[str, Any]:
        """\n        Extract ADS header with strict validation\n        In strict mode, missing headers cause compilation failure\n        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Parse first YAML document (frontmatter)
            docs = list(yaml.safe_load_all(content))
            if not docs:
                if self.strict_mode:
                    raise ValueError("No YAML documents found")
                return None
                
            header = docs[0]
            
            # Check if it's an ADS header (has required fields)
            if not isinstance(header, dict) or 'ads_version' not in header:
                if self.strict_mode:
                    raise ValueError("Missing or invalid ADS header - all standards must have v2.0 headers")
                return None
                
            return header
            
        except Exception as e:
            error_msg = f"[STRICT:FAIL] {file_path}: {str(e)}"
            self.errors.append(error_msg)
            if self.strict_mode:
                raise SystemExit(error_msg)
            return None
    
    def _validate_header(self, header: Dict[str, Any], file_path: Path) -> bool:
        """Validate ADS header against schema"""
        try:
            jsonschema.validate(instance=header, schema=self.schema)
            
            # Additional semantic validation
            tier = header.get('tier')
            tier_dir = f"tier{tier}-" if tier else None
            
            if tier_dir and tier_dir not in str(file_path):
                self.warnings.append(
                    f"{file_path}: Tier {tier} header in non-tier{tier} directory"
                )
            
            return True
            
        except jsonschema.exceptions.ValidationError as e:
            self.errors.append(f"{file_path}: {e.message}")
            return False
    
    def _build_task_mappings(self, headers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build task_mappings from collected headers"""
        mappings = {}
        
        for header_data in headers:
            header = header_data['header']
            file_path = header_data['path']
            
            triggers = header.get('triggers', {})
            
            for task_id, task_config in triggers.items():
                if task_id not in mappings:
                    mappings[task_id] = {
                        'description': header['description'],
                        'priority': task_config.get('priority', 2),
                        'standards': [],
                        'triggers': {
                            'keywords': [],
                            'path_patterns': []
                        }
                    }
                
                # Accumulate standards
                if str(file_path) not in mappings[task_id]['standards']:
                    mappings[task_id]['standards'].append(str(file_path))
                
                # Merge triggers
                if 'keywords' in task_config:
                    mappings[task_id]['triggers']['keywords'].extend(
                        task_config['keywords']
                    )
                if 'path_patterns' in task_config:
                    mappings[task_id]['triggers']['path_patterns'].extend(
                        task_config['path_patterns']
                    )
        
        # Deduplicate triggers
        for task_id in mappings:
            mappings[task_id]['triggers']['keywords'] = sorted(
                set(mappings[task_id]['triggers']['keywords'])
            )
            mappings[task_id]['triggers']['path_patterns'] = sorted(
                set(mappings[task_id]['triggers']['path_patterns'])
            )
        
        return mappings
    
    def compile(self, dry_run: bool = False) -> bool:
        """
        Compile registry.yaml from all ADS headers
        
        Args:
            dry_run: If True, validate but don't write registry
            
        Returns:
            True if compilation successful
        """
        print("🔍 Scanning for ADS headers...")
        
        valid_headers = []
        scanned = 0
        skipped = 0
        
        # Scan all YAML files
        for yaml_file in self.root.rglob("*.yaml"):
            # Skip registry itself and templates
            if (yaml_file.name == "registry.yaml" or 
                "schemas/" in str(yaml_file) or
                "templates/" in str(yaml_file)):
                continue
            
            scanned += 1
            
            # Extract header
            header = self._extract_ads_header(yaml_file)
            if not header:
                skipped += 1
                continue
            
            # Validate header
            if self._validate_header(header, yaml_file):
                std_id = header.get('id')
                self.standards_map[std_id] = {
                    **header,
                    'path': str(yaml_file)
                }
                valid_headers.append({
                    'header': header,
                    'path': yaml_file
                })
        
        print(f"✅ Scanned {scanned} files")
        print(f"✅ Found {len(valid_headers)} valid ADS headers")
        print(f"⚠️  Skipped {skipped} files (no ADS header)")
        
        if self.errors:
            print(f"\n❌ {len(self.errors)} validation errors:")
            for error in self.errors:
                print(f"   {error}")
            return False
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"   {warning}")
        
        # Detect circular dependencies
        print("\n🔍 Checking for circular dependencies...")
        cycles = self._detect_cycles()
        if cycles:
            print(f"\n❌ [CYCLE:FAIL] Circular dependencies detected:")
            for cycle in cycles:
                print(f"   {cycle}")
            raise SystemExit("[ADS:Sync] Compilation aborted due to circular dependencies.")
        print("✅ No circular dependencies found")
        
        # Build task mappings
        task_mappings = self._build_task_mappings(valid_headers)
        
        # Build registry structure
        registry = {
            'ads_version': '2.0',
            'type': 'unified_registry',
            'agent': 'all',
            'tier': 1,
            'priority': 'critical',
            'name': 'AgentQMS Standards Registry',
            'description': 'AUTO-GENERATED BY sync_registry.py - DO NOT EDIT MANUALLY',
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'AgentQMS Registry Compiler v1.0',
                'schema_version': '1.0',
                'total_standards': len(valid_headers),
                'total_tasks': len(task_mappings)
            },
            'root_map': {
                'schema': 'AgentQMS/standards/schemas/ads-v1.0-spec.yaml',
                'tier1': 'AgentQMS/standards/tier1-sst/',
                'tier2': 'AgentQMS/standards/tier2-framework/',
                'tier3': 'AgentQMS/standards/tier3-agents/',
                'tier4': 'AgentQMS/standards/tier4-workflows/'
            },
            'task_mappings': task_mappings
        }
        
        # Generate visual graph
        print("\n📊 Generating architecture visualization...")
        self._generate_dot_graph()
        
        # Print semantic diff
        self._print_semantic_diff(registry)
        
        if dry_run:
            print("\n🔍 DRY RUN - Registry structure:")
            print(f"   Tasks: {len(task_mappings)}")
            print(f"   Standards indexed: {len(valid_headers)}")
            return True
        
        # Write registry
        print(f"\n💾 Writing registry to {self.registry_path}...")
        
        # Write to temp file first
        temp_path = self.registry_path.with_suffix('.yaml.tmp')
        
        try:
            with open(temp_path, 'w') as f:
                yaml.dump(
                    registry,
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True
                )
            
            # Atomic rename
            temp_path.rename(self.registry_path)
            
            print(f"✅ Registry compiled successfully!")
            print(f"   📊 {len(task_mappings)} tasks mapped")
            print(f"   📄 {len(valid_headers)} standards indexed")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to write registry: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compile registry.yaml from ADS headers"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate without writing registry'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('AgentQMS/standards'),
        help='Root directory for standards'
    )
    
    args = parser.parse_args()
    
    compiler = RegistryCompiler(root_dir=args.root)
    success = compiler.compile(dry_run=args.dry_run)
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

**Tests**:
- Test with empty directory (should generate minimal registry)
- Test with 5 valid headers (should compile correct task mappings)
- Test with malformed YAML (should log error and skip)
- Test with invalid tier (should reject with validation error)
- Test dry-run mode (should validate without writing)
- Test atomic write (verify .tmp file created then renamed)

**Acceptance**: Crawler successfully compiles registry from valid headers

---

#### Task 1.3: Unit Tests for Crawler
**Effort**: 6 hours  
**Files to Create**:
- `AgentQMS/tests/test_sync_registry.py`

**Implementation**: Create pytest test suite covering:
- Schema validation (valid/invalid headers)
- Header extraction (YAML parsing edge cases)
- Task mapping compilation (single task, multiple tasks, priority ordering)
- File path pattern detection (tier validation)
- Error handling (missing files, malformed YAML)
- Atomic write behavior (temp file → rename)

**Acceptance**: 95%+ code coverage, all tests pass

---

### Phase 2: Resolution Tool & Query API (Week 2-3)

**Goal**: Build the lightweight query interface that agents use instead of loading full registry

#### Task 2.1: Create Resolution Tool API
**Effort**: 10 hours  
**Files to Create**:
- `AgentQMS/tools/resolve_standards.py`

**Implementation**:

```python
"""
AgentQMS Standard Resolution Tool
Query interface for discovering applicable standards
"""
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import fnmatch


class StandardResolver:
    """Resolves applicable standards for files/tasks"""
    
    def __init__(self, registry_path: Path = Path("AgentQMS/standards/registry.yaml")):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load compiled registry"""
        with open(self.registry_path) as f:
            return yaml.safe_load(f)
    
    def _match_path_patterns(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file path matches any glob pattern"""
        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False
    
    def _score_keyword_match(self, query: str, keywords: List[str]) -> int:
        """Score keyword relevance (simple term matching)"""
        query_lower = query.lower()
        score = 0
        for keyword in keywords:
            if keyword.lower() in query_lower:
                score += 1
        return score
    
    def resolve_by_task(self, task_id: str) -> Dict[str, Any]:
        """
        Resolve standards for a specific task type
        
        Args:
            task_id: Task identifier (e.g., "config_files", "code_quality")
            
        Returns:
            Dict with standards list and metadata
        """
        task_mappings = self.registry.get('task_mappings', {})
        
        if task_id not in task_mappings:
            return {
                'task_id': task_id,
                'found': False,
                'standards': [],
                'error': f'Task "{task_id}" not found in registry'
            }
        
        mapping = task_mappings[task_id]
        
        return {
            'task_id': task_id,
            'found': True,
            'description': mapping.get('description', ''),
            'priority': mapping.get('priority', 2),
            'standards': mapping.get('standards', []),
            'triggers': mapping.get('triggers', {})
        }
    
    def resolve_by_path(self, file_path: str) -> Dict[str, Any]:
        """
        Resolve standards applicable to a file path
        
        Args:
            file_path: Relative file path (e.g., "ocr/models/vgg.py")
            
        Returns:
            Dict with matched standards and task associations
        """
        task_mappings = self.registry.get('task_mappings', {})
        matched_tasks = []
        all_standards = set()
        
        # Normalize path
        normalized_path = file_path.replace('\\', '/')
        
        # Check each task's path patterns
        for task_id, mapping in task_mappings.items():
            path_patterns = mapping.get('triggers', {}).get('path_patterns', [])
            
            if self._match_path_patterns(normalized_path, path_patterns):
                matched_tasks.append({
                    'task_id': task_id,
                    'priority': mapping.get('priority', 2),
                    'description': mapping.get('description', '')
                })
                all_standards.update(mapping.get('standards', []))
        
        # Sort by priority
        matched_tasks.sort(key=lambda x: x['priority'])
        
        return {
            'file_path': file_path,
            'matched_tasks': matched_tasks,
            'standards': sorted(all_standards),
            'total_matches': len(matched_tasks)
        }
    
    def resolve_by_keywords(self, query: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Resolve standards based on keyword search
        
        Args:
            query: Search query (e.g., "hydra configuration")
            top_n: Maximum number of results to return
            
        Returns:
            Dict with ranked standard matches
        """
        task_mappings = self.registry.get('task_mappings', {})
        scored_tasks = []
        
        for task_id, mapping in task_mappings.items():
            keywords = mapping.get('triggers', {}).get('keywords', [])
            score = self._score_keyword_match(query, keywords)
            
            if score > 0:
                scored_tasks.append({
                    'task_id': task_id,
                    'score': score,
                    'priority': mapping.get('priority', 2),
                    'description': mapping.get('description', ''),
                    'standards': mapping.get('standards', [])
                })
        
        # Sort by score (desc) then priority (asc)
        scored_tasks.sort(key=lambda x: (-x['score'], x['priority']))
        
        return {
            'query': query,
            'results': scored_tasks[:top_n],
            'total_matches': len(scored_tasks)
        }
    
    def list_all_tasks(self) -> List[str]:
        """List all available task IDs"""
        return sorted(self.registry.get('task_mappings', {}).keys())


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Resolve standards for files and tasks"
    )
    parser.add_argument(
        '--task',
        help='Task ID to resolve'
    )
    parser.add_argument(
        '--path',
        help='File path to resolve'
    )
    parser.add_argument(
        '--keywords',
        help='Keyword search query'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tasks'
    )
    parser.add_argument(
        '--registry',
        type=Path,
        default=Path('AgentQMS/standards/registry.yaml'),
        help='Registry file path'
    )
    
    args = parser.parse_args()
    
    resolver = StandardResolver(registry_path=args.registry)
    
    if args.list:
        tasks = resolver.list_all_tasks()
        print(f"📋 Available tasks ({len(tasks)}):")
        for task in tasks:
            print(f"   - {task}")
        return
    
    result = None
    
    if args.task:
        result = resolver.resolve_by_task(args.task)
    elif args.path:
        result = resolver.resolve_by_path(args.path)
    elif args.keywords:
        result = resolver.resolve_by_keywords(args.keywords)
    else:
        parser.print_help()
        return
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

**Tests**:
- Test resolve_by_task with valid task ID
- Test resolve_by_task with invalid task ID
- Test resolve_by_path with matching patterns
- Test resolve_by_path with no matches
- Test resolve_by_keywords scoring algorithm
- Test list_all_tasks returns sorted list

**Acceptance**: Resolution tool returns correct standards for all query types

---

#### Task 2.2: Integrate with Agent Context Loading
**Effort**: 8 hours  
**Files to Modify**:
- `AgentQMS/tools/utilities/context_bundler.py` (or wherever context is loaded)

**Implementation**: Replace full registry loading with resolver calls:

```python
from AgentQMS.tools.resolve_standards import StandardResolver

def load_context_for_task(task_type: str) -> str:
    """Load only relevant standards for task"""
    resolver = StandardResolver()
    result = resolver.resolve_by_task(task_type)
    
    if not result['found']:
        return ""
    
    # Load only the identified standard files
    context_parts = []
    for std_path in result['standards']:
        with open(std_path) as f:
            context_parts.append(f.read())
    
    return "\n\n".join(context_parts)
```

**Tests**:
- Measure token count before/after (verify 80% reduction claim)
- Test with config_files task (should load 3 standards, not entire registry)
- Verify all relevant standards are loaded
- Ensure no irrelevant standards are included

**Acceptance**: Agent context loading uses resolution tool, token count reduced by 80%

---

### Phase 3: Migration Tooling (Week 3-4)

**Goal**: Create tools to retrofit existing 71 standards with ADS headers

#### Task 3.1: Build Migration Script
**Effort**: 14 hours  
**Files to Create**:
- `AgentQMS/tools/migrate_to_ads_headers.py`

**Implementation**:

```python
"""
AgentQMS ADS Header Migration Tool
Retrofits existing standards with ADS headers
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import re


class ADSMigrationTool:
    """Migrates legacy standards to ADS header format"""
    
    def __init__(self, registry_path: Path, root_dir: Path):
        self.registry_path = registry_path
        self.root_dir = root_dir
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load current registry.yaml"""
        with open(self.registry_path) as f:
            return yaml.safe_load(f)
    
    def _infer_tier_from_path(self, file_path: Path) -> Optional[int]:
        """Infer tier number from directory path"""
        path_str = str(file_path)
        
        for tier in range(1, 5):
            if f"tier{tier}-" in path_str:
                return tier
        
        return None
    
    def _find_triggers_for_file(self, file_path: Path) -> Dict[str, Any]:
        """Find task triggers that reference this file in registry"""
        file_str = str(file_path)
        triggers = {}
        
        task_mappings = self.registry.get('task_mappings', {})
        
        for task_id, mapping in task_mappings.items():
            standards = mapping.get('standards', [])
            
            if file_str in standards or str(file_path.resolve()) in standards:
                # Extract trigger data for this task
                triggers[task_id] = {
                    'priority': mapping.get('priority', 2),
                    'keywords': mapping.get('triggers', {}).get('keywords', []),
                    'path_patterns': mapping.get('triggers', {}).get('path_patterns', [])
                }
        
        return triggers
    
    def _generate_ads_header(self, file_path: Path, existing_header: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate ADS header for a file"""
        tier = self._infer_tier_from_path(file_path)
        triggers = self._find_triggers_for_file(file_path)
        
        # Try to extract name/description from existing content
        name = existing_header.get('name') if existing_header else file_path.stem.replace('-', ' ').title()
        description = existing_header.get('description') if existing_header else f"Standards for {name}"
        
        header = {
            'ads_version': '1.0',
            'name': name,
            'description': description,
            'tier': tier or 2,  # Default to tier 2 if ambiguous
            'type': existing_header.get('type', 'standard') if existing_header else 'standard',
            'priority': existing_header.get('priority', 'medium') if existing_header else 'medium',
            'status': existing_header.get('status', 'active') if existing_header else 'active',
            'triggers': triggers
        }
        
        return header
    
    def migrate_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """
        Migrate a single file to ADS header format
        
        Args:
            file_path: Path to standard file
            dry_run: If True, show what would be done without modifying
            
        Returns:
            True if migration successful
        """
        print(f"📄 Processing: {file_path}")
        
        # Read existing content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if already has ADS header
        if 'ads_version:' in content[:500]:
            print("   ✅ Already has ADS header - skipping")
            return True
        
        # Try to parse existing frontmatter
        existing_header = None
        if content.startswith('---'):
            try:
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    existing_header = yaml.safe_load(parts[1])
            except:
                pass
        
        # Generate ADS header
        ads_header = self._generate_ads_header(file_path, existing_header)
        
        # Build new content
        header_yaml = yaml.dump(ads_header, sort_keys=False, default_flow_style=False)
        
        if content.startswith('---'):
            # Replace existing frontmatter
            parts = content.split('---', 2)
            new_content = f"---\n{header_yaml}---{parts[2] if len(parts) > 2 else ''}"
        else:
            # Add frontmatter
            new_content = f"---\n{header_yaml}---\n\n{content}"
        
        if dry_run:
            print(f"   🔍 Would add ADS header:")
            print(f"      Tier: {ads_header['tier']}")
            print(f"      Triggers: {len(ads_header['triggers'])} tasks")
            return True
        
        # Write updated file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"   ✅ Migrated successfully")
        return True
    
    def migrate_all(self, dry_run: bool = False, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Migrate all standard files
        
        Args:
            dry_run: If True, preview without modifying
            limit: Max number of files to migrate (for testing)
            
        Returns:
            Dict with migration statistics
        """
        print("🔄 Starting ADS header migration...")
        
        stats = {
            'scanned': 0,
            'migrated': 0,
            'skipped': 0,
            'errors': 0
        }
        
        for yaml_file in self.root_dir.rglob("*.yaml"):
            if (yaml_file.name == "registry.yaml" or
                "schemas/" in str(yaml_file) or
                "templates/" in str(yaml_file)):
                continue
            
            stats['scanned'] += 1
            
            try:
                if self.migrate_file(yaml_file, dry_run=dry_run):
                    stats['migrated'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                stats['errors'] += 1
            
            if limit and stats['migrated'] >= limit:
                print(f"\n⚠️  Reached limit of {limit} files")
                break
        
        print(f"\n📊 Migration {'Preview' if dry_run else 'Complete'}:")
        print(f"   Scanned: {stats['scanned']}")
        print(f"   Migrated: {stats['migrated']}")
        print(f"   Skipped: {stats['skipped']}")
        print(f"   Errors: {stats['errors']}")
        
        return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate standards to ADS header format"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Migrate a single file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of files to migrate (for testing)'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('AgentQMS/standards'),
        help='Root directory for standards'
    )
    
    args = parser.parse_args()
    
    registry_path = args.root / 'registry.yaml'
    migrator = ADSMigrationTool(registry_path, args.root)
    
    if args.file:
        success = migrator.migrate_file(args.file, dry_run=args.dry_run)
        exit(0 if success else 1)
    else:
        stats = migrator.migrate_all(dry_run=args.dry_run, limit=args.limit)
        exit(0 if stats['errors'] == 0 else 1)


if __name__ == "__main__":
    main()
```

**Tests**:
- Test migration of file without frontmatter
- Test migration of file with existing frontmatter
- Test tier inference from path
- Test trigger extraction from registry
- Test dry-run mode (no file modifications)
- Test limit parameter (stops at N files)

**Acceptance**: Migration tool successfully retrofits standards with valid headers

---

#### Task 3.2: Pilot Migration (5 Files)
**Effort**: 4 hours  
**Target Files**:
- `tier1-sst/naming-conventions.yaml`
- `tier2-framework/coding/python-core.yaml`
- `tier2-framework/hydra-v5-rules.yaml`
- `tier3-agents/claude/prompts.yaml` (if exists)
- `tier4-workflows/experiment-workflow.yaml` (if exists)

**Process**:
1. Run migration tool in dry-run mode on 5 files
2. Review generated ADS headers for accuracy
3. Run actual migration
4. Run registry sync
5. Test resolution tool queries
6. Validate agent can load standards correctly

**Acceptance**: 5 pilot files migrated, registry compiles, resolution works

---

### Phase 4: Integration & Automation (Week 4-5)

**Goal**: Integrate tools into CLI, add validation hooks, update agent prompts

#### Task 4.1: Add CLI Commands
**Effort**: 6 hours  
**Files to Modify**:
- `AgentQMS/cli.py` or create `bin/aqms` script

**Implementation**: Add subcommands:

```bash
# Compile registry from ADS headers
aqms sync-registry [--dry-run]

# Resolve standards for task/path/keywords
aqms resolve-standards --task config_files
aqms resolve-standards --path ocr/models/vgg.py
aqms resolve-standards --keywords "hydra configuration"

# List all tasks
aqms list-tasks

# Migrate standards to ADS headers
aqms migrate-headers [--dry-run] [--file PATH] [--limit N]

# Validate ADS headers
aqms validate-headers [--strict]
```

**Tests**:
- Test each CLI command
- Verify help text is clear
- Test error handling (invalid arguments)

**Acceptance**: All commands work, help text complete

---

#### Task 4.2: Pre-Commit Hook Integration
**Effort**: 4 hours  
**Files to Create**:
- `.pre-commit-hooks.yaml`
- `AgentQMS/hooks/validate-ads-headers.py`

**Implementation**:

```yaml
# .pre-commit-hooks.yaml
- id: validate-ads-headers
  name: Validate ADS Headers
  entry: python AgentQMS/hooks/validate-ads-headers.py
  language: system
  files: ^AgentQMS/standards/.*\.yaml$
  exclude: (registry\.yaml|schemas/|templates/)
```

```python
# AgentQMS/hooks/validate-ads-headers.py
"""Pre-commit hook to validate ADS headers"""
import sys
from pathlib import Path
from AgentQMS.tools.sync_registry import RegistryCompiler

def main():
    """Validate changed standard files"""
    changed_files = sys.argv[1:]
    
    compiler = RegistryCompiler()
    errors = []
    
    for file_path in changed_files:
        path = Path(file_path)
        
        # Extract and validate header
        header = compiler._extract_ads_header(path)
        
        if not header:
            errors.append(f"{path}: Missing ADS header")
            continue
        
        if not compiler._validate_header(header, path):
            errors.append(f"{path}: Invalid ADS header")
    
    if errors:
        print("❌ ADS Header Validation Failed:")
        for error in errors:
            print(f"   {error}")
        return 1
    
    print("✅ ADS headers valid")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**Tests**:
- Test hook with valid files (should pass)
- Test hook with missing header (should fail)
- Test hook with invalid tier (should fail)

**Acceptance**: Pre-commit hook blocks invalid standards

---

#### Task 4.3: CI Validation Pipeline
**Effort**: 6 hours  
**Files to Create/Modify**:
- `.github/workflows/validate-standards.yml`

**Implementation**:

```yaml
name: Validate Standards

on:
  push:
    paths:
      - 'AgentQMS/standards/**/*.yaml'
  pull_request:
    paths:
      - 'AgentQMS/standards/**/*.yaml'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install pyyaml jsonschema
      
      - name: Validate ADS Headers
        run: |
          python AgentQMS/tools/sync_registry.py --dry-run
      
      - name: Check Registry Compilation
        run: |
          python AgentQMS/tools/sync_registry.py
          git diff --exit-code AgentQMS/standards/registry.yaml
```

**Tests**:
- Test workflow with valid changes (should pass)
- Test workflow with invalid header (should fail)
- Test workflow with malformed YAML (should fail)

**Acceptance**: CI validates all standard changes

---

#### Task 4.4: Update Agent System Prompts
**Effort**: 4 hours  
**Files to Modify**:
- `.github/copilot-instructions.md` or agent prompt files

**Add to Prompts**:

```markdown
## ADS Header Compliance

When creating or modifying standard files in `AgentQMS/standards/`:

1. **REQUIRED**: Every standard MUST have an ADS header
2. **Header Structure**:
   - `ads_version: "1.0"`
   - `name`: Human-readable name (3-100 chars)
   - `description`: What it governs (10-500 chars)
   - `tier`: Integer 1-4 (SST/Framework/Agents/Workflows)
   - `type`: principle/standard/template/schema
   - `triggers`: Map of task_id → {priority, keywords, path_patterns}

3. **Tier Classification**:
   - Tier 1 (SST): Global laws (naming, placement)
   - Tier 2 (Framework): Technical infrastructure
   - Tier 3 (Agents): AI persona configs
   - Tier 4 (Workflows): Procedural sequences

4. **After Changes**: Run `aqms sync-registry` to update registry.yaml

5. **FORBIDDEN**: Never manually edit `registry.yaml` - it is auto-generated
```

**Acceptance**: Agents create standards with valid ADS headers

---

### Phase 5: Full Migration & Validation (Week 5-6)

**Goal**: Migrate all 71 standards, validate system, measure performance

#### Task 5.1: Tier-by-Tier Migration
**Effort**: 16 hours (spread across tiers)

**Process**:
1. **Week 5 Day 1**: Migrate Tier 1 (SST) - ~8 files
2. **Week 5 Day 2**: Migrate Tier 2 (Framework) - ~40 files
3. **Week 5 Day 3**: Migrate Tier 3 (Agents) - ~15 files
4. **Week 5 Day 4**: Migrate Tier 4 (Workflows) - ~8 files

**For Each Tier**:
1. Run migration tool with dry-run
2. Review generated headers manually
3. Adjust triggers if needed
4. Run actual migration
5. Sync registry
6. Run validation
7. Test agent queries
8. Commit tier changes

**Rollback Procedure**:
- If errors occur, revert tier and fix issues
- Each tier is a separate commit for easy rollback
- Keep old registry.yaml backed up until full validation

**Acceptance**: All 71 files migrated with valid headers, registry compiles

---

#### Task 5.2: Comprehensive Testing
**Effort**: 10 hours

**Test Suite**:

1. **Unit Tests**:
   - All compiler functions (95%+ coverage)
   - All resolver functions
   - All migration functions

2. **Integration Tests**:
   - End-to-end: create standard → sync → resolve → load
   - Agent context loading with resolution tool
   - CLI commands with real files

3. **Performance Tests**:
   - Registry compilation time (<5s target)
   - Resolution query time (<100ms target)
   - Agent context load time (measure reduction)

4. **Validation Tests**:
   - Pre-commit hook blocks invalid files
   - CI fails on malformed standards
   - Atomic write prevents corruption

**Acceptance**: All tests pass, performance targets met

---

#### Task 5.3: Performance Benchmarking
**Effort**: 4 hours

**Metrics to Measure**:
- Registry compilation time (full 71 files)
- Resolution query latency (by-task, by-path, by-keywords)
- Agent context token count (before/after)
- Memory footprint during compilation

**Benchmarking Script**:

```python
import time
from AgentQMS.tools.sync_registry import RegistryCompiler
from AgentQMS.tools.resolve_standards import StandardResolver

def benchmark():
    # Registry compilation
    start = time.time()
    compiler = RegistryCompiler()
    compiler.compile()
    compile_time = time.time() - start
    
    # Resolution queries
    resolver = StandardResolver()
    
    start = time.time()
    result = resolver.resolve_by_task("config_files")
    task_query_time = time.time() - start
    
    start = time.time()
    result = resolver.resolve_by_path("ocr/models/vgg.py")
    path_query_time = time.time() - start
    
    start = time.time()
    result = resolver.resolve_by_keywords("hydra configuration")
    keyword_query_time = time.time() - start
    
    # Context token count
    # (Measure before/after agent context loading)
    
    print(f"Registry compilation: {compile_time:.3f}s")
    print(f"Task query: {task_query_time*1000:.1f}ms")
    print(f"Path query: {path_query_time*1000:.1f}ms")
    print(f"Keyword query: {keyword_query_time*1000:.1f}ms")

if __name__ == "__main__":
    benchmark()
```

**Acceptance**: All performance targets met (see success criteria)

---

#### Task 5.4: Documentation & Training
**Effort**: 8 hours

**Documentation to Create/Update**:

1. **Developer Guide**: [AgentQMS/docs/ads-header-guide.md](../../AgentQMS/docs/ads-header-guide.md)
   - ADS header structure explained
   - How to choose tier
   - How to define triggers
   - Examples for each tier

2. **CLI Reference**: Update [AgentQMS/README.md](../../AgentQMS/README.md)
   - Document all new commands
   - Usage examples
   - Common workflows

3. **Migration Guide**: [AgentQMS/docs/migration-to-ads.md](../../AgentQMS/docs/migration-to-ads.md)
   - Why we migrated
   - Before/after comparison
   - How to maintain going forward

4. **Architecture Doc**: Update [ai-native-architecture.md](../../AgentQMS/standards/tier1-sst/ai-native-architecture.md)
   - Explain auto-generation system
   - Resolution tool usage
   - Best practices

**Acceptance**: Complete documentation, examples for each tier

---

#### Task 5.5: Rollout & Freeze
**Effort**: 4 hours

**Rollout Process**:

1. **Final Validation**:
   - Run full test suite
   - Run benchmarks
   - Verify all 71 files have valid headers
   - Verify registry compiles without errors

2. **Backup**:
   - Tag current state: `v1.0-pre-ads-automation`
   - Backup old registry.yaml
   - Document rollback procedure

3. **Deploy**:
   - Merge migration branch to main
   - Update all agent prompts
   - Enable pre-commit hooks
   - Enable CI validation

4. **Freeze**:
   - Make registry.yaml read-only (git attributes)
   - Add comment in registry: "AUTO-GENERATED - DO NOT EDIT"
   - Update CONTRIBUTING.md with new workflow

5. **Monitoring**:
   - Watch for agent errors related to standard discovery
   - Monitor CI for validation failures
   - Track registry sync performance

**Rollback Procedure** (if critical issues):
1. Revert to `v1.0-pre-ads-automation` tag
2. Restore old registry.yaml
3. Disable pre-commit hooks
4. Document issues for post-mortem

**Acceptance**: System deployed, registry frozen, monitoring active

---

## Dependencies & Prerequisites

### Technical Dependencies
- Python 3.11+
- PyYAML 6.0+
- jsonschema 4.0+
- pytest (for testing)
- pre-commit (for hooks)

### Knowledge Prerequisites
- Understanding of AgentQMS 4-tier architecture
- Familiarity with YAML frontmatter
- JSON Schema validation concepts
- Git workflow and branching

### System Prerequisites
- All 71 existing standards identified
- Current registry.yaml backed up
- Agent context loading code identified
- CI/CD pipeline access

---

## Risk Assessment & Mitigation

### Risk 1: Migration Breaks Agents
**Impact**: High  
**Probability**: Medium  
**Mitigation**:
- Pilot migration with 5 files first
- Tier-by-tier rollout with validation between tiers
- Keep old registry until full validation
- Easy rollback with git tags

### Risk 2: Registry Compilation Fails
**Impact**: Critical  
**Probability**: Low  
**Mitigation**:
- Extensive unit tests for compiler
- Atomic write (temp file → rename)
- Validation before write
- Pre-commit hooks prevent bad inputs

### Risk 3: Context Token Reduction Insufficient
**Impact**: Medium  
**Probability**: Low  
**Mitigation**:
- Benchmark before/after
- Adjust resolution algorithms if needed
- Query optimization (caching, indexing)

### Risk 4: Developer Adoption Resistance
**Impact**: Medium  
**Probability**: Medium  
**Mitigation**:
- Clear documentation and examples
- Migration tool handles most work automatically
- Pre-commit hooks enforce compliance
- Benefits visible immediately (no manual registry edits)

---

## Success Metrics

### Quantitative Metrics
- ✅ Registry.yaml 100% auto-generated (0 manual edits after rollout)
- ✅ Agent context token count reduced 80% (1000 → 200 tokens)
- ✅ Registry compilation time <5 seconds
- ✅ Resolution query latency <100ms
- ✅ All 71 standards migrated with valid headers
- ✅ Pre-commit validation blocks 100% of invalid standards
- ✅ Zero registry corruption events during 6-week rollout

### Qualitative Metrics
- Developer feedback: "Easier to maintain standards"
- Agent performance: "Faster context loading"
- Code reviews: "No more registry merge conflicts"
- Documentation: "Clear examples for new standards"

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|-----------------|
| Phase 1 | Week 1-2 | ADS schema, registry compiler, tests |
| Phase 2 | Week 2-3 | Resolution tool, agent integration |
| Phase 3 | Week 3-4 | Migration tool, pilot migration (5 files) |
| Phase 4 | Week 4-5 | CLI commands, hooks, CI, agent prompts |
| Phase 5 | Week 5-6 | Full migration (71 files), testing, rollout |

**Total Duration**: 6 weeks

---

## Related Artifacts

- **Specification**: [specs/001-registry-automation/spec.md](../../specs/001-registry-automation/spec.md)
- **Draft Research**: [__DEBUG__/2026-01-25_standards-update/additional-updates/draft-research.md](../../__DEBUG__/2026-01-25_standards-update/additional-updates/draft-research.md)
- **Architecture Philosophy**: [AgentQMS/standards/tier1-sst/ai-native-architecture.md](../../AgentQMS/standards/tier1-sst/ai-native-architecture.md)
- **Current Registry**: [AgentQMS/standards/registry.yaml](../../AgentQMS/standards/registry.yaml)

---

## Maintenance & Long-Term Operation

### Post-Rollout Workflow

**When Creating New Standard**:
1. Create file with ADS header (use template)
2. Define tier, triggers, description
3. Run `aqms sync-registry` (or pre-commit hook does it)
4. Commit both new standard and updated registry

**When Modifying Standard**:
1. Update ADS header if triggers change
2. Run `aqms sync-registry`
3. Commit changes

**When Relocating Standard**:
1. Move file to new directory
2. Update `tier` in ADS header if needed
3. Run `aqms sync-registry` (auto-detects new path)
4. Commit move and updated registry

### Quarterly Maintenance
- Review trigger accuracy (are standards being discovered correctly?)
- Check for orphaned standards (files with no triggers)
- Audit resolution query performance
- Update documentation with new examples

### Future Enhancements
- Caching layer for resolution queries
- Fuzzy keyword matching (Levenshtein distance)
- ML-based trigger suggestion during migration
- Visual dependency graph of standards
- Registry diff tool (what changed between syncs)

---

## Appendix: Example ADS Headers

### Tier 1 Example (SST - Constitutional Law)
```yaml
---
ads_version: "1.0"
name: Naming Conventions
description: Global naming standards for files, functions, variables
tier: 1
type: principle
priority: critical
status: active
triggers:
  code_changes:
    priority: 1
    keywords:
      - naming
      - convention
      - identifier
      - variable name
      - function name
    path_patterns:
      - "**/*.py"
      - "**/*.ts"
  artifact_creation:
    priority: 1
    keywords:
      - create artifact
      - new file
    path_patterns:
      - "docs/artifacts/**/*.md"
---
```

### Tier 2 Example (Framework - Technical Infrastructure)
```yaml
---
ads_version: "1.0"
name: Hydra v5 Configuration Rules
description: Hydra configuration architecture - domains-first, atomic composition
tier: 2
type: standard
priority: high
status: active
triggers:
  config_files:
    priority: 1
    keywords:
      - hydra
      - config
      - omegaconf
      - domains first
      - atomic architecture
    path_patterns:
      - "configs/**/*.yaml"
      - "**/conf/**/*.yaml"
  code_changes:
    priority: 2
    keywords:
      - instantiate
      - OmegaConf
      - DictConfig
    path_patterns:
      - "ocr/**/*.py"
---
```

### Tier 3 Example (Agents - Persona Configuration)
```yaml
---
ads_version: "1.0"
name: Claude Opus Persona
description: Claude 3 Opus identity, behavior, and prompt configuration
tier: 3
type: agent_config
priority: medium
status: active
triggers:
  agent_development:
    priority: 2
    keywords:
      - claude
      - opus
      - agent prompt
      - persona
    path_patterns:
      - "AgentQMS/standards/tier3-agents/claude/**/*.yaml"
---
```

### Tier 4 Example (Workflows - Procedural Sequences)
```yaml
---
ads_version: "1.0"
name: Experiment Workflow
description: Step-by-step experiment execution and result tracking
tier: 4
type: workflow
priority: medium
status: active
triggers:
  experiment_execution:
    priority: 1
    keywords:
      - experiment
      - run experiment
      - etk
      - experiment workflow
    path_patterns:
      - "experiment_manager/**/*.py"
      - "experiments/**/*.yaml"
---
```

---

**Plan Status**: Ready for Execution  
**Next Step**: Begin Phase 1 - Create ADS header schema  
**Questions**: Contact AgentQMS maintainers

