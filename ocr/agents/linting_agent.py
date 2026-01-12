import subprocess
import os
import logging
import json
import sys
from typing import Any
from pathlib import Path

# Add project root to sys.path if not present
# This allows running the script directly from anywhere
try:
    from ocr.core.utils.path_utils import PROJECT_ROOT
except ImportError:
    # Fallback/Bootstrap if ocr package is not yet importable
    current_file = Path(__file__).resolve()
    # Path: ocr/agents/linting_agent.py
    # agents(0)->ocr(1)->root(2)
    project_root = current_file.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from ocr.core.utils.path_utils import PROJECT_ROOT

from ocr.communication.rabbitmq_transport import RabbitMQTransport

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LintingAgent")

class LintingAgent:
    def __init__(self):
        host = os.getenv("RABBITMQ_HOST", "rabbitmq")
        self.transport = RabbitMQTransport(host=host, agent_id="agent.linter")
        self.project_root = PROJECT_ROOT

    def run(self):
        """Starts the agent loop."""
        logger.info(f"Starting LintingAgent (Root: {self.project_root})...")
        try:
            self.transport.start_listening(
                binding_keys=["cmd.lint_code.#"],
                handler=self.handle_lint_request
            )
        except KeyboardInterrupt:
            logger.info("Stopping agent...")
            self.transport.close()

    def handle_lint_request(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """
        Process a linting request.
        Expected payload: {"files": ["path/to/file.py"], "linter": "ruff"}
        """
        payload = envelope.get('payload', {})
        files = payload.get('files', [])
        linter = payload.get('linter', 'ruff')

        logger.info(f"Processing lint request for {len(files)} files using {linter}")

        if not files:
            return {"status": "error", "message": "No files provided"}

        # Resolve paths relative to project root
        resolved_files = []
        for f in files:
            # Join path with project root to ensure we are working with absolute paths
            # path_utils PROJECT_ROOT is absolute
            path = self.project_root / f
            if path.exists():
                resolved_files.append(str(path))
            else:
                logger.warning(f"File not found: {path} (requested: {f})")

        if not resolved_files:
             return {"status": "error", "message": "No valid files found"}

        violations = []

        if linter == 'ruff':
            # Run ruff on the files
            # We use --json to get structured output
            cmd = ["ruff", "check", "--output-format=json"] + resolved_files
            try:
                # Run from project root to keep paths clean in output (if ruff supports relative output)
                # or just run normally. Using cwd=project_root matches expectations.
                result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=self.project_root)

                if result.stdout:
                    try:
                        lint_data = json.loads(result.stdout)
                        violations = lint_data
                    except json.JSONDecodeError:
                        violations = [{"error": "Failed to parse ruff output", "raw": result.stdout}]

                # Check for other errors or warnings
                if result.stderr:
                    logger.warning(f"Ruff stderr: {result.stderr}")

            except Exception as e:
                logger.error(f"Failed to run ruff: {e}")
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
