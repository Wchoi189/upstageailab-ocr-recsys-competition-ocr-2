
import glob
import yaml
import json
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(".").resolve()
BUNDLES_DIR = PROJECT_ROOT / "AgentQMS/.agentqms/plugins/context_bundles"
DB_PATH = PROJECT_ROOT / "AgentQMS/.agentqms/standards_db.json"

def load_db():
    if not DB_PATH.exists():
        print(f"Error: DB not found at {DB_PATH}")
        return {}
    with open(DB_PATH) as f:
        return json.load(f)

def audit_bundles():
    db = load_db()
    # Assume db is a dict where keys are standard IDs or paths?
    # Based on view_file, we'll see the structure.
    # For now assuming simple key-value or list.
    # If db is a dict of standards, we check if the file path matches a standard ID or property.

    # Flatten db keys - assume it's a dict where keys are file paths or IDs
    # Based on file view, the root keys seem to be paths like "tier1-sst/naming-conventions.yaml"
    db_keys = set(db.keys())

    # Also index by 'file_path' value if available
    for k, v in db.items():
        if isinstance(v, dict) and 'file_path' in v:
             db_keys.add(v['file_path'])

    print(f"{'BUNDLE':<30} | {'TOTAL':<5} | {'BROKEN':<6} | {'IN_DB':<5} | {'STATUS'}")
    print("-" * 80)

    total_broken_refs = 0
    affected_bundles = 0

    for bundle_path in glob.glob(str(BUNDLES_DIR / "*.yaml")):
        bundle_name = Path(bundle_path).stem
        try:
            with open(bundle_path) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"{bundle_name:<30} | ERROR PARSING: {e}")
            continue

        files = []
        if 'tiers' in data:
            for tier in data['tiers'].values():
                if 'files' in tier:
                    files.extend(tier['files'])

        broken_count = 0
        in_db_count = 0

        for file_def in files:
            path_str = ""
            if isinstance(file_def, str):
                path_str = file_def
            elif isinstance(file_def, dict):
                path_str = file_def.get('path')

            if not path_str: continue

            # Skip globs for existence check, but maybe check DB?
            if '*' in path_str:
                continue

            full_path = PROJECT_ROOT / path_str
            if not full_path.exists():
                broken_count += 1
                # Check DB - Try multiple variations
                # DB keys are relative paths like "tier2-framework/..."
                # Paths in bundles are "AgentQMS/standards/tier2-framework/..."

                found_in_db = False

                # Check 1: Exact Match (unlikely if missing)
                if path_str in db_keys: found_in_db = True

                # Check 2: Relative to AgentQMS/standards/
                if not found_in_db and "AgentQMS/standards/" in path_str:
                    rel_path = path_str.split("AgentQMS/standards/")[-1]
                    if rel_path in db_keys: found_in_db = True

                # Check 3: Just filename (loose match)
                if not found_in_db:
                     filename = Path(path_str).name
                     # This is expensive but let's check
                     for k in db_keys:
                         if k.endswith(filename):
                             found_in_db = True
                             break

                if found_in_db:
                    in_db_count += 1
                # else:
                #    print(f"DEBUG: Broken & Not in DB: {path_str}")

        status = "OK"
        if broken_count > 0:
            status = "FAIL"
            affected_bundles += 1
            total_broken_refs += broken_count

        print(f"{bundle_name:<30} | {len(files):<5} | {broken_count:<6} | {in_db_count:<5} | {status}")

    print("-" * 80)
    print(f"Total Affected Bundles: {affected_bundles}")
    print(f"Total Broken References: {total_broken_refs}")

if __name__ == "__main__":
    audit_bundles()
