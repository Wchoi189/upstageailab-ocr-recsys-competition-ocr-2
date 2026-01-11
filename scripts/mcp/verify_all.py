
import sys
import subprocess
from pathlib import Path

def run_verification():
    project_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = project_root / "scripts" / "mcp"

    scripts = [
        "verify_server.py",
        "verify_feedback.py"
    ]

    print(f"Running verification scripts from {scripts_dir}...")

    for script in scripts:
        print(f"\n--- Running {script} ---")
        try:
            result = subprocess.run(
                [sys.executable, str(scripts_dir / script)],
                cwd=str(project_root),
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"❌ {script} Failed:")
                print(result.stderr)
            else:
                print(f"✅ {script} Passed")
        except Exception as e:
            print(f"❌ Error running {script}: {e}")

if __name__ == "__main__":
    run_verification()
