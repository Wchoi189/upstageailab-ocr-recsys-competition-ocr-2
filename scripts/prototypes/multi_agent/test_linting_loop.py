import os
import sys
import json
import logging
from pathlib import Path

# Bootstrap path
try:
    from AgentQMS.tools.utils.system.paths import get_project_root
    project_root = get_project_root()
except ImportError:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from ocr.core.utils.path_utils import PROJECT_ROOT
from ocr.core.infrastructure.communication.rabbitmq_transport import RabbitMQTransport

logging.basicConfig(level=logging.INFO)

def main():
    transport = RabbitMQTransport(host=os.getenv("RABBITMQ_HOST", "rabbitmq"), agent_id="agent.test_client")
    transport.connect()

    # Create a temporary file relative to PROJECT_ROOT for testing
    test_file_path = PROJECT_ROOT / "temp_lint_test.py"
    try:
        with open(test_file_path, "w") as f:
            f.write("import os\nimport sys\n\ndef foo():\n    x = 1\n    y = 2\n")

        # Send command to agent.linter
        response = transport.send_command(
            target="agent.linter",
            type_suffix="lint_code",
            payload={
                "files": ["temp_lint_test.py"], # Send relative path
                "linter": "ruff"
            },
            timeout=15
        )

        print("\n=== RESPONSE RECEIVED ===")
        print(json.dumps(response, indent=2))

        payload = response['payload']
        if payload['status'] == 'success' and len(payload['violations']) > 0:
            print("\nSUCCESS: Violations found as expected.")
        else:
            print("\nFAILURE: Unexpected response structure or no violations found.")

    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        transport.close()
        if test_file_path.exists():
            test_file_path.unlink()

if __name__ == "__main__":
    main()
