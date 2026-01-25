"""
Grok4 API client wrapper for multi-agent collaboration.

Integrates with xAI's Grok API for high-quality reasoning tasks.
"""

import os
import logging
from typing import Optional

from ocr.core.infrastructure.agents.llm.base_client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class Grok4Client(BaseLLMClient):
    """
    Client for Grok4 API (xAI).

    Uses the OpenAI-compatible API format.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "grok-beta"
    ):
        """
        Initialize Grok4 client.

        Args:
            api_key: xAI API key (defaults to XAI_API_KEY env var)
            base_url: API base URL
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            logger.warning("No Grok API key provided. Set XAI_API_KEY environment variable.")

        self.base_url = base_url or os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        self.model = model

        logger.info(f"Initialized Grok4Client with model {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text using Grok4.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Grok-specific arguments

        Returns:
            Generated text
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Grok4 generation failed: {e}")
            raise

    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Generate text with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Yields:
            Text chunks
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Grok4 streaming generation failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count using tiktoken.

        Args:
            text: Input text

        Returns:
            Token count
        """
        try:
            import tiktoken

            # Use cl100k_base encoding (similar to GPT-4)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

        except ImportError:
            logger.warning("tiktoken not available, using approximation")
            # Fallback: approximate 4 chars per token
            return len(text) // 4
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return len(text) // 4
