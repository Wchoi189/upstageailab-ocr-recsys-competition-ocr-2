"""LLM client wrappers for multi-agent collaboration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_client import BaseLLMClient
    from .qwen_client import QwenClient
    from .grok_client import Grok4Client
    from .openai_client import OpenAIClient

def __getattr__(name: str):
    if name == "BaseLLMClient":
        from .base_client import BaseLLMClient
        return BaseLLMClient
    elif name == "QwenClient":
        from .qwen_client import QwenClient
        return QwenClient
    elif name == "Grok4Client":
        from .grok_client import Grok4Client
        return Grok4Client
    elif name == "OpenAIClient":
        from .openai_client import OpenAIClient
        return OpenAIClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseLLMClient",
    "QwenClient",
    "Grok4Client",
    "OpenAIClient",
]
