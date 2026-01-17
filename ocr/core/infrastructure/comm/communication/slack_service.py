import os
import logging
import httpx

logger = logging.getLogger(__name__)

class SlackNotificationService:
    """
    Service for sending notifications to Slack via Webhooks.
    """
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        if not self.webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not set. Slack notifications will be disabled.")

    def send_message(self, text: str, blocks: list | None = None) -> bool:
        """
        Sends a message to the configured Slack webhook.

        Args:
            text: The main text of the message.
            blocks: Optional standard Slack Block Kit blocks.

        Returns:
            True if successful, False otherwise.
        """
        if not self.webhook_url:
            logger.info("Skipping Slack notification (no webhook URL configured).")
            return False

        payload = {"text": text}
        if blocks:
            payload["blocks"] = blocks

        try:
            response = httpx.post(self.webhook_url, json=payload, timeout=10.0)
            response.raise_for_status()
            logger.debug("Slack notification sent successfully.")
            return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    async def send_message_async(self, text: str, blocks: list | None = None) -> bool:
        """
        Asynchronously sends a message to Slack.
        """
        if not self.webhook_url:
            return False

        payload = {"text": text}
        if blocks:
            payload["blocks"] = blocks

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.webhook_url, json=payload, timeout=10.0)
                response.raise_for_status()
            logger.debug("Slack notification sent successfully (async).")
            return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to send Slack notification (async): {e}")
            return False
