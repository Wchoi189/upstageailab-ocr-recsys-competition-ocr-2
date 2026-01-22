import json
import re
from pathlib import Path

def parse_audit_report(report_path):
    """
    Parses your 'master_audit.py' output into a machine-parseable dictionary.
    """
    with open(report_path, 'r') as f:
        content = f.read()

    # Regex to capture file path, current import, and the error
    import_errors = re.findall(
        r"\[File\] (.*?):\d+\s+--> Import: (.*?) \[(.*?)\]\s+--> Error: (.*)",
        content
    )

    manifest = []
    for file_path, module, symbols, error in import_errors:
        manifest.append({
            "action": "ALIGN_IMPORT",
            "file": file_path.strip(),
            "faulty_module": module.strip(),
            "symbols": [s.strip().replace("'", "") for s in symbols.split(',')],
            "status": "PENDING"
        })
    return manifest

def generate_ai_instructions(manifest):
    """
    Wraps the manifest in high-precision guardrails for the AI agent.
    """
    instruction_header = """
### 🛠️ PHASE 1: IMPORT ALIGNMENT EXECUTION
**Guardrails:**
1. ONLY modify the lines identified in the manifest.
2. Use 'adt intelligent-search' to find the NEW location of each 'faulty_module'.
3. If 'intelligent-search' returns 'NOT_FOUND', flag for human review.
4. Ensure 'pip install -e .' is run after every 5 fixes to refresh the symbol table.
    """

    print(instruction_header)
    print("```json")
    print(json.dumps(manifest[:10], indent=2)) # Send in batches of 10
    print("```")

if __name__ == "__main__":
    # Assuming you pipe your audit report to a text file
    # python scripts/audit/master_audit.py > audit_results.txt
    report = parse_audit_report("audit_results.txt")
    generate_ai_instructions(report)
