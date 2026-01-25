import subprocess
import json
import logging
from typing import Any

from ocr.core.infrastructure.agents.base_agent import BaseAgent

logger = logging.getLogger("LintingAgent")

class LintingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent.linter",
            binding_keys=["cmd.lint_code.#"]
        )

    def process_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Process a linting request.
        Expected payload: {"files": ["path/to/file.py"], "linter": "ruff"}
        """
        files = payload.get('files', [])
        linter = payload.get('linter', 'ruff')

        self.logger.info(f"Processing lint request for {len(files)} files using {linter}")

        if not files:
            return {"status": "error", "message": "No files provided"}

        # Resolve paths relative to project root using base helper
        resolved_files = []
        for f in files:
            path = self.resolve_path(f)
            if path.exists():
                resolved_files.append(str(path))
            else:
                self.logger.warning(f"File not found: {path} (requested: {f})")

        if not resolved_files:
             return {"status": "error", "message": "No valid files found"}

        violations = []

        if linter == 'ruff':
            # Run ruff on the files
            # We use --json to get structured output
            cmd = ["ruff", "check", "--output-format=json"] + resolved_files
            try:
                # Run from project root to keep paths clean in output
                result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=self.project_root)

                if result.stdout:
                    try:
                        lint_data = json.loads(result.stdout)
                        violations = lint_data
                    except json.JSONDecodeError:
                        violations = [{"error": "Failed to parse ruff output", "raw": result.stdout}]

                # Check for other errors or warnings
                if result.stderr:
                    self.logger.warning(f"Ruff stderr: {result.stderr}")

            except Exception as e:
                self.logger.error(f"Failed to run ruff: {e}")
                return {"status": "error", "message": str(e)}

        return {
            "status": "success",
            "linter": linter,
            "checked_files": files, # Return original requested paths
            "violations": violations
        }

if __name__ == "__main__":
    agent = LintingAgent()
    agent.run()
