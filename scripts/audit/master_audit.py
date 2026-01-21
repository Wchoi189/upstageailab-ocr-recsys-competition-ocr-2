
import ast
import importlib
import inspect
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple
import yaml
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("MasterAudit")

# Add workspace root to sys.path
WORKSPACE_ROOT = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2")
sys.path.append(str(WORKSPACE_ROOT))

class AuditResult:
    def __init__(self):
        self.broken_imports: List[Dict] = []
        self.broken_targets: List[Dict] = []
        self.missing_paths: List[Dict] = []
        self.checked_files = 0
        self.scanned_configs = 0

def resolve_import(module_name: str, from_list: List[str] = None) -> bool:
    """Try to import a module and optional attributes."""
    try:
        module = importlib.import_module(module_name)
        if from_list:
            for item in from_list:
                if item == '*': continue
                if not hasattr(module, item):
                    # It might be a submodule (e.g. from os import path)
                    try:
                        importlib.import_module(f"{module_name}.{item}")
                    except ImportError:
                         return False
        return True
    except ImportError:
        return False
    except Exception as e:
        # Some side effects might cause other errors, but we care mainly about existence
        return False

def check_python_imports(result: AuditResult):
    """Scan all .py files for broken imports using AST."""
    logger.info("Scanning Python files for broken imports...")

    # Directories to scan
    dirs_to_scan = [WORKSPACE_ROOT / "ocr", WORKSPACE_ROOT / "runners", WORKSPACE_ROOT / "scripts"]

    for root_dir in dirs_to_scan:
        if not root_dir.exists(): continue

        for py_file in root_dir.rglob("*.py"):
            result.checked_files += 1
            try:
                with open(py_file, "r") as f:
                    tree = ast.parse(f.read(), filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if not resolve_import(alias.name):
                                result.broken_imports.append({
                                    "file": str(py_file.relative_to(WORKSPACE_ROOT)),
                                    "line": node.lineno,
                                    "module": alias.name,
                                    "error": "Module not found"
                                })
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            # Handle relative imports logic broadly or just skip check for very local ones if complex
                            # For now, let's assume absolute imports or simple relative ones resolved by importlib
                            module_name = node.module
                            if node.level > 0:
                                # Skip relative imports verification for now in this simple scanner
                                # as it requires context of the package structure
                                continue

                            if not resolve_import(module_name, [n.name for n in node.names]):
                                result.broken_imports.append({
                                    "file": str(py_file.relative_to(WORKSPACE_ROOT)),
                                    "line": node.lineno,
                                    "module": module_name,
                                    "names": [n.name for n in node.names],
                                    "error": "Module or attribute not found"
                                })
            except SyntaxError:
                pass # Skip files with syntax errors (legacy scripts etc)
            except Exception as e:
                logger.error(f"Failed to scan {py_file}: {e}")

def check_hydra_targets(result: AuditResult):
    """Scan all .yaml files for broken _target_ definitions."""
    logger.info("Scanning Hydra configs for broken targets...")

    config_dir = WORKSPACE_ROOT / "configs"

    for yaml_file in config_dir.rglob("*.yaml"):
        result.scanned_configs += 1
        try:
            # Load raw yaml to avoid hydra resolution errors masking the target errors
            with open(yaml_file, 'r') as f:
                # Use safe_load to just get structure
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError:
                    continue # Skip invalid yaml

            if not isinstance(data, (dict, list)):
                continue

            # Recursive search for _target_
            def search_target(obj, path=""):
                if isinstance(obj, dict):
                    if "_target_" in obj:
                        target = obj["_target_"]
                        verify_target(target, yaml_file, path, result)
                    if "_partial_" in obj: # _partial_ implies _target_ usually, but checking anyway
                         pass

                    for k, v in obj.items():
                        search_target(v, f"{path}.{k}" if path else k)
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        search_target(v, f"{path}[{i}]")

            search_target(data)

        except Exception as e:
            logger.error(f"Failed to scan config {yaml_file}: {e}")

def verify_target(target_path: str, file_path: Path, dict_path: str, result: AuditResult):
    """Verify if a python path string points to a valid object."""
    parts = target_path.split('.')
    module_name = ".".join(parts[:-1])
    object_name = parts[-1]

    try:
        module = importlib.import_module(module_name)
        if not hasattr(module, object_name):
             result.broken_targets.append({
                "config": str(file_path.relative_to(WORKSPACE_ROOT)),
                "key": dict_path,
                "target": target_path,
                "error": f"Module '{module_name}' has no attribute '{object_name}'"
            })
    except ImportError:
         result.broken_targets.append({
            "config": str(file_path.relative_to(WORKSPACE_ROOT)),
            "key": dict_path,
            "target": target_path,
            "error": f"Module '{module_name}' not found"
        })
    except Exception as e:
         result.broken_targets.append({
            "config": str(file_path.relative_to(WORKSPACE_ROOT)),
            "key": dict_path,
            "target": target_path,
            "error": str(e)
        })

def check_hardcoded_paths(result: AuditResult):
    """Check for commonly hardcoded paths that might verify as broken."""
    # This is heuristic based on the user report
    logger.info("Checking for known problematic paths...")
    # Just a placeholder for strict path checking if we were parsing 'path' keys in configs
    pass

def print_report(result: AuditResult):
    print("\n" + "="*80)
    print(f"AUDIT REPORT | Scanned {result.checked_files} .py files, {result.scanned_configs} .yaml files")
    print("="*80)

    if result.broken_imports:
        print(f"\n🚨 BROKEN IMPORTS ({len(result.broken_imports)}):")
        for err in result.broken_imports:
            print(f"  [File] {err['file']}:{err.get('line', '?')}")
            print(f"    --> Import: {err['module']} {err.get('names', '')}")
            print(f"    --> Error: {err['error']}")

    if result.broken_targets:
        print(f"\n🚨 BROKEN HYDRA TARGETS ({len(result.broken_targets)}):")
        for err in result.broken_targets:
            print(f"  [Config] {err['config']}")
            print(f"    --> Key: {err['key']}")
            print(f"    --> Target: {err['target']}")
            print(f"    --> Error: {err['error']}")

    if not result.broken_imports and not result.broken_targets:
        print("\n✅ No static anomalies found.")
    else:
        print("\n❌ Anomalies detected. See kill list above.")

if __name__ == "__main__":
    audit_res = AuditResult()
    check_python_imports(audit_res)
    check_hydra_targets(audit_res)
    check_hardcoded_paths(audit_res)
    print_report(audit_res)
