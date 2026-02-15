import yaml
import subprocess
from pathlib import Path

def _extract_yaml_block(content: str) -> str | None:
    start_token = "```yaml"
    end_token = "```"
    start_idx = content.find(start_token)
    if start_idx == -1:
        return None
    start_idx += len(start_token)
    end_idx = content.find(end_token, start_idx)
    if end_idx == -1:
        return None
    return content[start_idx:end_idx].strip()


def run_compliance_audit(spec_path: str, target_dir: str):
    spec_file = Path(spec_path)
    if spec_file.suffix == ".md":
        raw_text = spec_file.read_text(encoding="utf-8") if spec_file.exists() else ""
        yaml_block = _extract_yaml_block(raw_text)
        standards = yaml.safe_load(yaml_block) if yaml_block else {}
    else:
        with open(spec_file, encoding="utf-8") as f:
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
        "AgentQMS/specs/tier2-framework/configuration.spec.md",
        "ocr/"
    )
