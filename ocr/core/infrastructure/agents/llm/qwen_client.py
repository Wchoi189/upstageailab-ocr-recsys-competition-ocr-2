"""
QwenCLI client wrapper for local LLM inference.

This wrapper integrates with Qwen models running via CLI/API for
cost-effective local inference in multi-agent workflows.
"""

import subprocess
import json
import logging
import os
from typing import Optional
from pathlib import Path

from .base_client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class QwenClient(BaseLLMClient):
    """
    Client for Qwen local LLM inference.

    Supports both CLI-based and API-based interaction with Qwen models.
    """

    def __init__(
        self,
        model_name: str = "qwen3:4b-instruct",
        api_endpoint: Optional[str] = None,
        mode: str = "api"  # 'cli' or 'api'
    ):
        """
        Initialize Qwen client (Ollama compatible).

        Args:
            model_name: Model tag to use (e.g. 'qwen3:4b-instruct')
            api_endpoint: API endpoint URL (default: http://host.docker.internal:11434)
            mode: Execution mode ('cli' or 'api')
        """
        self.mode = mode
        self.model_name = model_name
        # Default to Ollama endpoint
        self.api_endpoint = api_endpoint or os.getenv("QWEN_API_ENDPOINT", "http://host.docker.internal:11434")

        # For CLI mode, we use model prompt/path logic, but we focus on API for now
        self.model_path = os.getenv("QWEN_MODEL_PATH", "/models/qwen") 

        logger.info(f"Initialized QwenClient (Model: {self.model_name}) in {mode} mode at {self.api_endpoint}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text using Qwen model via Ollama API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        if self.mode == "api":
            return self._generate_api(prompt, system_prompt, temperature, max_tokens, **kwargs)
        else:
            return self._generate_cli(prompt, system_prompt, temperature, max_tokens, **kwargs)

    def _generate_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using API endpoint (Ollama compatible)."""
        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.api_endpoint}/api/chat", # Ollama native API or OpenAI compat
                    json=payload
                )
                
                # Check if we should use /v1/chat/completions (OpenAI compat) or /api/chat (Ollama native)
                # The user endpoint http://host.docker.internal:11434 serves typical Ollama API.
                # OpenAI compat is usually at /v1. Let's try to be smart or stick to one.
                # Defaulting to OpenAI compatible endpoint as originally implemented, assuming Ollama is running valid wrapper or native compat.
                # Actually, standard Ollama port 11434 exposes /api/chat and /v1/chat/completions (recent versions).
                # Let's start with /v1/chat/completions as it was the original code's intent (standardized).
                
                if response.status_code == 404:
                     # Fallback to Ollama native /api/chat if /v1 is missing
                     logger.info("Fallback to /api/chat")
                     payload.pop("max_tokens", None) # Ollama might use options
                     payload["options"] = {"num_predict": max_tokens, "temperature": temperature}
                     response = client.post(f"{self.api_endpoint}/api/chat", json=payload)

                response.raise_for_status()
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                elif "message" in data: # Ollama native format
                    return data["message"]["content"]
                else:
                    raise ValueError(f"Invalid response format from API: {data.keys()}")

        except Exception as e:
            logger.error(f"Qwen API generation failed: {e}")
            raise

    def _generate_cli(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using CLI interface."""
        # Construct CLI command
        # This is a placeholder - adjust based on actual Qwen CLI interface
        cmd = [
            "qwen-cli",
            "--model", self.model_path,
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
            "--prompt", prompt
        ]

        if system_prompt:
            cmd.extend(["--system", system_prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )

            # Parse output (adjust based on actual CLI output format)
            output = result.stdout.strip()

            # If output is JSON
            try:
                data = json.loads(output)
                return data.get("response", output)
            except json.JSONDecodeError:
                return output

        except subprocess.TimeoutExpired:
            logger.error("Qwen CLI generation timed out")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Qwen CLI generation failed: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Qwen CLI: {e}")
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
        Generate with streaming (API mode only).

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Yields:
            Text chunks
        """
        if self.mode != "api":
            raise NotImplementedError("Streaming only supported in API mode")

        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream(
                    "POST",
                    f"{self.api_endpoint}/v1/chat/completions",
                    json=payload
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Qwen streaming generation failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.

        Args:
            text: Input text

        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 chars per token for Chinese/English mixed
        # For production, use actual tokenizer
        return len(text) // 4 + len(text.split())
