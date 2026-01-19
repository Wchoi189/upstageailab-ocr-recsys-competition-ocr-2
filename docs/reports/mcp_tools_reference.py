"""
MCP Tools Quick Reference for Configuration Audit

This script demonstrates how to call MCP tools programmatically
for configuration standards compliance auditing.

Available via: mcp_unified_proje_adt_meta_query
"""

# ============================================================================
# EXAMPLE 1: Find all config accesses in ocr/
# ============================================================================

# MCP Tool Call:
{
    "kind": "config_access",
    "target": "ocr/",
    "options": {
        "output": "json"
    }
}

# Returns: JSON with all cfg.X, config['X'], config.get() patterns
# Output location: Can be saved to file or processed in memory


# ============================================================================
# EXAMPLE 2: Find Hydra usage patterns
# ============================================================================

# MCP Tool Call:
{
    "kind": "hydra_usage",
    "target": "ocr/",
    "options": {}
}

# Returns: All @hydra.main, instantiate(), compose() calls


# ============================================================================
# EXAMPLE 3: Trace OmegaConf.merge() precedence
# ============================================================================

# MCP Tool Call:
{
    "kind": "merge_order",
    "target": "ocr/core/lightning/lightning_module.py",
    "options": {}
}

# Returns: Merge operation analysis with precedence order


# ============================================================================
# EXAMPLE 4: Build dependency graph for core/
# ============================================================================

# MCP Tool Call:
{
    "kind": "dependency_graph",
    "target": "ocr/core/",
    "options": {
        "output": "json"
    }
}

# Returns: Import and call dependency graph


# ============================================================================
# EXAMPLE 5: Analyze code complexity
# ============================================================================

# MCP Tool Call:
{
    "kind": "complexity",
    "target": "ocr/",
    "options": {
        "threshold": 10
    }
}

# Returns: Functions/methods exceeding complexity threshold


# ============================================================================
# EXAMPLE 6: Generate semantic context tree
# ============================================================================

# MCP Tool Call:
{
    "kind": "context_tree",
    "target": "ocr/recognition/",
    "options": {
        "depth": 2,
        "output": "markdown"
    }
}

# Returns: Annotated directory tree with docstrings and semantic labels


# ============================================================================
# EXAMPLE 7: Search for specific symbols
# ============================================================================

# MCP Tool Call:
{
    "kind": "symbol_search",
    "target": "ensure_dict",
    "options": {
        "fuzzy": True,
        "threshold": 0.8
    }
}

# Returns: All locations where ensure_dict is defined or used


# ============================================================================
# EXAMPLE 8: Find component instantiation patterns
# ============================================================================

# MCP Tool Call:
{
    "kind": "component_instantiations",
    "target": "ocr/core/models/",
    "options": {
        "component": "encoder"
    }
}

# Returns: All factory patterns for encoder creation


# ============================================================================
# USAGE IN COPILOT CHAT
# ============================================================================

"""
Example prompts:

1. "Use mcp_unified_proje_adt_meta_query to find all config accesses in ocr/recognition/"

2. "Search for hydra_usage patterns in ocr/ using the MCP tool"

3. "Generate a context tree for ocr/core/ with depth 3"

4. "Find all uses of ensure_dict using symbol_search"
"""


# ============================================================================
# CLI EQUIVALENTS (for comparison)
# ============================================================================

"""
MCP: {"kind": "config_access", "target": "ocr/"}
CLI: uv run adt analyze-config ocr/

MCP: {"kind": "hydra_usage", "target": "ocr/"}
CLI: uv run adt find-hydra ocr/

MCP: {"kind": "merge_order", "target": "file.py"}
CLI: uv run adt trace-merges file.py

MCP: {"kind": "dependency_graph", "target": "ocr/"}
CLI: uv run adt analyze-dependencies ocr/

MCP: {"kind": "complexity", "target": "ocr/", "options": {"threshold": 10}}
CLI: uv run adt analyze-complexity ocr/ -t 10

MCP: {"kind": "context_tree", "target": "ocr/", "options": {"depth": 2}}
CLI: uv run adt context-tree ocr/ --depth 2

MCP: {"kind": "symbol_search", "target": "ensure_dict"}
CLI: uv run adt intelligent-search "ensure_dict"

MCP: {"kind": "component_instantiations", "target": "ocr/", "options": {"component": "encoder"}}
CLI: uv run adt find-instantiations ocr/ --component encoder
"""


# ============================================================================
# COMPLIANCE-SPECIFIC QUERIES
# ============================================================================

def audit_isinstance_violations():
    """Find isinstance(x, dict) without DictConfig check."""
    return {
        "kind": "config_access",
        "target": "ocr/",
        "options": {
            "output": "json",
            "filter_pattern": "isinstance.*dict"
        }
    }


def audit_to_container_usage():
    """Find OmegaConf.to_container() instead of ensure_dict()."""
    return {
        "kind": "config_access",
        "target": "ocr/",
        "options": {
            "output": "json",
            "filter_pattern": "to_container"
        }
    }


def find_ensure_dict_usage():
    """Find all files using ensure_dict() (good pattern)."""
    return {
        "kind": "symbol_search",
        "target": "ensure_dict",
        "options": {
            "fuzzy": False
        }
    }


def analyze_config_utils_structure():
    """Understand config_utils module structure."""
    return {
        "kind": "context_tree",
        "target": "ocr/core/utils/config_utils.py",
        "options": {
            "depth": 1,
            "output": "markdown"
        }
    }


# ============================================================================
# WORKFLOW EXAMPLE
# ============================================================================

"""
Systematic audit workflow using MCP tools:

1. Get overview:
   {"kind": "context_tree", "target": "ocr/", "options": {"depth": 2}}

2. Find all config accesses:
   {"kind": "config_access", "target": "ocr/"}

3. Check specific violations:
   - grep results for "isinstance.*dict"
   - grep results for "to_container"

4. Find correct usage patterns:
   {"kind": "symbol_search", "target": "is_config"}
   {"kind": "symbol_search", "target": "ensure_dict"}

5. Analyze dependencies:
   {"kind": "dependency_graph", "target": "ocr/core/"}

6. Check merge precedence (if needed):
   {"kind": "merge_order", "target": "ocr/core/lightning/lightning_module.py"}
"""
