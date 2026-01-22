import subprocess
import importlib
import sys
import os

# Ensure the local 'ocr' package is in the path
sys.path.append(os.getcwd())

def get_true_path(class_name):
    """Uses Python reflection to find the current module of a class."""
    # This is the "magic" part. We search for the class across the ocr namespace.
    # For simplicity, we can use your ADT 'intelligent-search' logic here.
    cmd = f"adt intelligent-search '{class_name}' --output json"
    try:
        result = subprocess.check_output(cmd, shell=True).decode()
        # Parse the JSON and return the qualified_path (e.g., ocr.core.models.X)
        import json
        data = json.loads(result)
        return data.get("qualified_path")
    except:
        return None

def heal_configs(audit_log):
    """Parses audit log and applies yq fixes."""
    with open(audit_log, 'r') as f:
        lines = f.readlines()

    current_config = None
    for line in lines:
        if "[Config]" in line:
            current_config = line.split("]")[1].strip()
        if "--> Target:" in line:
            old_target = line.split(":")[1].strip()
            class_name = old_target.split(".")[-1] # Extract 'DBLoss' from '...db_loss.DBLoss'

            print(f"🛠️ Attempting to heal: {class_name} in {current_config}")
            new_path = get_true_path(class_name)

            if new_path:
                print(f"✅ Found new home: {new_path}")
                # Use yq to update ONLY that specific broken string
                yq_cmd = f"yq -i '.. | select(. == \"{old_target}\") = \"{new_path}\"' {current_config}"
                subprocess.run(yq_cmd, shell=True)
            else:
                print(f"❌ Could not locate {class_name}. Manual intervention required.")

if __name__ == "__main__":
    heal_configs("audit_results.txt")
