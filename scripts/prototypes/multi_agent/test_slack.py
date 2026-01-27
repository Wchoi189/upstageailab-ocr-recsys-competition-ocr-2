import os
import sys
import asyncio
from pathlib import Path

# Bootstrap path
try:
    from AgentQMS.tools.utils.system.paths import get_project_root
    project_root = get_project_root()
except ImportError:
    current_file = Path(__file__).resolve()
    # scripts(1)/prototypes(2)/multi_agent(3)/file
    project_root = current_file.parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from ocr.core.infrastructure.communication.slack_service import SlackNotificationService

# Use the webhook URL from environment variable
TEST_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "REPLACE_WITH_YOUR_WEBHOOK_URL")

def test_sync():
    print("Testing Sync Message...")
    service = SlackNotificationService(webhook_url=TEST_WEBHOOK_URL)
    success = service.send_message(text="Test Message from Multi-Agent Infrastructure (Sync)")
    print(f"Sync Result: {success}")

async def test_async():
    print("Testing Async Message...")
    service = SlackNotificationService(webhook_url=TEST_WEBHOOK_URL)
    success = await service.send_message_async(text="Test Message from Multi-Agent Infrastructure (Async)")
    print(f"Async Result: {success}")

if __name__ == "__main__":
    test_sync()
    asyncio.run(test_async())
