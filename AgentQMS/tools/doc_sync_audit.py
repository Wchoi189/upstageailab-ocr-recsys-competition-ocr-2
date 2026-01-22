import yaml
import subprocess

def run_compliance_audit(standards_path: str, target_dir: str):
    with open(standards_path) as f:
        standards = yaml.safe_load(f)

    for rule in standards.get('rules', []):
        print(f"Checking Rule: {rule['id']}...")
        for pattern in rule.get('bad_patterns', []):
            # Escape the pattern for shell execution
            escaped_pattern = pattern.replace('"', '\\"')
            cmd = f"adt sg-search --pattern \"{escaped_pattern}\" --path {target_dir}"

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if "Matches: 0" not in result.stdout and result.returncode == 0:
                print(f"🚨 VIOLATION FOUND for rule {rule['id']} in {target_dir}")
                print(result.stdout)
                # In CI, we would sys.exit(1) here

if __name__ == "__main__":
    run_compliance_audit(
        "AgentQMS/standards/tier2-framework/configuration-standards.yaml",
        "ocr/"
    )
