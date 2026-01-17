{
  "analyzer_name": "DependencyGraphAnalyzer",
  "target_path": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr",
  "results": [
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/base_agent.py",
      "line": 10,
      "column": 0,
      "pattern": "import signal",
      "context": "Imports module: signal",
      "category": "import",
      "code_snippet": "import sys\nimport signal\nfrom typing import Any",
      "metadata": {
        "module": "signal",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/base_agent.py",
      "line": 18,
      "column": 0,
      "pattern": "from ocr.communication.rabbitmq_transport import RabbitMQTransport",
      "context": "Imports RabbitMQTransport from ocr.communication.rabbitmq_transport",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.communication.rabbitmq_transport import RabbitMQTransport\nfrom ocr.core.utils.path_utils import PROJECT_ROOT",
      "metadata": {
        "module": "ocr.communication.rabbitmq_transport",
        "name": "RabbitMQTransport",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/base_agent.py",
      "line": 19,
      "column": 0,
      "pattern": "from ocr.core.utils.path_utils import PROJECT_ROOT",
      "context": "Imports PROJECT_ROOT from ocr.core.utils.path_utils",
      "category": "from_import",
      "code_snippet": "from ocr.communication.rabbitmq_transport import RabbitMQTransport\nfrom ocr.core.utils.path_utils import PROJECT_ROOT\n",
      "metadata": {
        "module": "ocr.core.utils.path_utils",
        "name": "PROJECT_ROOT",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/base_agent.py",
      "line": 302,
      "column": 16,
      "pattern": "from ocr.agents.llm.qwen_client import QwenClient",
      "context": "Imports QwenClient from ocr.agents.llm.qwen_client",
      "category": "from_import",
      "code_snippet": "            if self.llm_provider == \"qwen\":\n                from ocr.agents.llm.qwen_client import QwenClient\n                self._llm_client = QwenClient()",
      "metadata": {
        "module": "ocr.agents.llm.qwen_client",
        "name": "QwenClient",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/base_agent.py",
      "line": 305,
      "column": 16,
      "pattern": "from ocr.agents.llm.grok_client import Grok4Client",
      "context": "Imports Grok4Client from ocr.agents.llm.grok_client",
      "category": "from_import",
      "code_snippet": "            elif self.llm_provider == \"grok4\":\n                from ocr.agents.llm.grok_client import Grok4Client\n                self._llm_client = Grok4Client()",
      "metadata": {
        "module": "ocr.agents.llm.grok_client",
        "name": "Grok4Client",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/base_agent.py",
      "line": 308,
      "column": 16,
      "pattern": "from ocr.agents.llm.openai_client import OpenAIClient",
      "context": "Imports OpenAIClient from ocr.agents.llm.openai_client",
      "category": "from_import",
      "code_snippet": "            elif self.llm_provider == \"openai\":\n                from ocr.agents.llm.openai_client import OpenAIClient\n                self._llm_client = OpenAIClient()",
      "metadata": {
        "module": "ocr.agents.llm.openai_client",
        "name": "OpenAIClient",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/coordinator_agent.py",
      "line": 20,
      "column": 0,
      "pattern": "from ocr.agents.base_agent import LLMAgent",
      "context": "Imports LLMAgent from ocr.agents.base_agent",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.base_agent import LLMAgent, AgentCapability\n",
      "metadata": {
        "module": "ocr.agents.base_agent",
        "name": "LLMAgent",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/coordinator_agent.py",
      "line": 20,
      "column": 0,
      "pattern": "from ocr.agents.base_agent import AgentCapability",
      "context": "Imports AgentCapability from ocr.agents.base_agent",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.base_agent import LLMAgent, AgentCapability\n",
      "metadata": {
        "module": "ocr.agents.base_agent",
        "name": "AgentCapability",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/linting_agent.py",
      "line": 6,
      "column": 0,
      "pattern": "from ocr.agents.base_agent import BaseAgent",
      "context": "Imports BaseAgent from ocr.agents.base_agent",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.base_agent import BaseAgent\n",
      "metadata": {
        "module": "ocr.agents.base_agent",
        "name": "BaseAgent",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from ocr.agents.llm.base_client import BaseLLMClient",
      "context": "Imports BaseLLMClient from ocr.agents.llm.base_client",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.llm.base_client import BaseLLMClient\nfrom ocr.agents.llm.qwen_client import QwenClient",
      "metadata": {
        "module": "ocr.agents.llm.base_client",
        "name": "BaseLLMClient",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from ocr.agents.llm.qwen_client import QwenClient",
      "context": "Imports QwenClient from ocr.agents.llm.qwen_client",
      "category": "from_import",
      "code_snippet": "from ocr.agents.llm.base_client import BaseLLMClient\nfrom ocr.agents.llm.qwen_client import QwenClient\nfrom ocr.agents.llm.grok_client import Grok4Client",
      "metadata": {
        "module": "ocr.agents.llm.qwen_client",
        "name": "QwenClient",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/__init__.py",
      "line": 5,
      "column": 0,
      "pattern": "from ocr.agents.llm.grok_client import Grok4Client",
      "context": "Imports Grok4Client from ocr.agents.llm.grok_client",
      "category": "from_import",
      "code_snippet": "from ocr.agents.llm.qwen_client import QwenClient\nfrom ocr.agents.llm.grok_client import Grok4Client\nfrom ocr.agents.llm.openai_client import OpenAIClient",
      "metadata": {
        "module": "ocr.agents.llm.grok_client",
        "name": "Grok4Client",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from ocr.agents.llm.openai_client import OpenAIClient",
      "context": "Imports OpenAIClient from ocr.agents.llm.openai_client",
      "category": "from_import",
      "code_snippet": "from ocr.agents.llm.grok_client import Grok4Client\nfrom ocr.agents.llm.openai_client import OpenAIClient\n",
      "metadata": {
        "module": "ocr.agents.llm.openai_client",
        "name": "OpenAIClient",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/grok_client.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.agents.llm.base_client import BaseLLMClient",
      "context": "Imports BaseLLMClient from ocr.agents.llm.base_client",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.llm.base_client import BaseLLMClient, LLMResponse\n",
      "metadata": {
        "module": "ocr.agents.llm.base_client",
        "name": "BaseLLMClient",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/grok_client.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.agents.llm.base_client import LLMResponse",
      "context": "Imports LLMResponse from ocr.agents.llm.base_client",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.llm.base_client import BaseLLMClient, LLMResponse\n",
      "metadata": {
        "module": "ocr.agents.llm.base_client",
        "name": "LLMResponse",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/grok_client.py",
      "line": 68,
      "column": 12,
      "pattern": "from openai import OpenAI",
      "context": "Imports OpenAI from openai",
      "category": "from_import",
      "code_snippet": "        try:\n            from openai import OpenAI\n",
      "metadata": {
        "module": "openai",
        "name": "OpenAI",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/grok_client.py",
      "line": 116,
      "column": 12,
      "pattern": "from openai import OpenAI",
      "context": "Imports OpenAI from openai",
      "category": "from_import",
      "code_snippet": "        try:\n            from openai import OpenAI\n",
      "metadata": {
        "module": "openai",
        "name": "OpenAI",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/grok_client.py",
      "line": 156,
      "column": 12,
      "pattern": "import tiktoken",
      "context": "Imports module: tiktoken",
      "category": "import",
      "code_snippet": "        try:\n            import tiktoken\n",
      "metadata": {
        "module": "tiktoken",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/openai_client.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.agents.llm.base_client import BaseLLMClient",
      "context": "Imports BaseLLMClient from ocr.agents.llm.base_client",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.llm.base_client import BaseLLMClient, LLMResponse\n",
      "metadata": {
        "module": "ocr.agents.llm.base_client",
        "name": "BaseLLMClient",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/openai_client.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.agents.llm.base_client import LLMResponse",
      "context": "Imports LLMResponse from ocr.agents.llm.base_client",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.llm.base_client import BaseLLMClient, LLMResponse\n",
      "metadata": {
        "module": "ocr.agents.llm.base_client",
        "name": "LLMResponse",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/openai_client.py",
      "line": 60,
      "column": 12,
      "pattern": "from openai import OpenAI",
      "context": "Imports OpenAI from openai",
      "category": "from_import",
      "code_snippet": "        try:\n            from openai import OpenAI\n",
      "metadata": {
        "module": "openai",
        "name": "OpenAI",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/openai_client.py",
      "line": 105,
      "column": 12,
      "pattern": "from openai import OpenAI",
      "context": "Imports OpenAI from openai",
      "category": "from_import",
      "code_snippet": "        try:\n            from openai import OpenAI\n",
      "metadata": {
        "module": "openai",
        "name": "OpenAI",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/openai_client.py",
      "line": 142,
      "column": 12,
      "pattern": "import tiktoken",
      "context": "Imports module: tiktoken",
      "category": "import",
      "code_snippet": "        try:\n            import tiktoken\n",
      "metadata": {
        "module": "tiktoken",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/qwen_client.py",
      "line": 15,
      "column": 0,
      "pattern": "from ocr.agents.llm.base_client import BaseLLMClient",
      "context": "Imports BaseLLMClient from ocr.agents.llm.base_client",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.llm.base_client import BaseLLMClient, LLMResponse\n",
      "metadata": {
        "module": "ocr.agents.llm.base_client",
        "name": "BaseLLMClient",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/qwen_client.py",
      "line": 15,
      "column": 0,
      "pattern": "from ocr.agents.llm.base_client import LLMResponse",
      "context": "Imports LLMResponse from ocr.agents.llm.base_client",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.llm.base_client import BaseLLMClient, LLMResponse\n",
      "metadata": {
        "module": "ocr.agents.llm.base_client",
        "name": "LLMResponse",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/qwen_client.py",
      "line": 85,
      "column": 8,
      "pattern": "import httpx",
      "context": "Imports module: httpx",
      "category": "import",
      "code_snippet": "        \"\"\"Generate using API endpoint.\"\"\"\n        import httpx\n",
      "metadata": {
        "module": "httpx",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/llm/qwen_client.py",
      "line": 192,
      "column": 8,
      "pattern": "import httpx",
      "context": "Imports module: httpx",
      "category": "import",
      "code_snippet": "\n        import httpx\n",
      "metadata": {
        "module": "httpx",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/ocr_agent.py",
      "line": 1,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "import cv2\nimport logging",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/ocr_agent.py",
      "line": 3,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import logging\nimport numpy as np\nfrom typing import Any",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/ocr_agent.py",
      "line": 6,
      "column": 0,
      "pattern": "from ocr.agents.base_agent import BaseAgent",
      "context": "Imports BaseAgent from ocr.agents.base_agent",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.base_agent import BaseAgent\nfrom ocr.core.inference.orchestrator import InferenceOrchestrator",
      "metadata": {
        "module": "ocr.agents.base_agent",
        "name": "BaseAgent",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/ocr_agent.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.core.inference.orchestrator import InferenceOrchestrator",
      "context": "Imports InferenceOrchestrator from ocr.core.inference.orchestrator",
      "category": "from_import",
      "code_snippet": "from ocr.agents.base_agent import BaseAgent\nfrom ocr.core.inference.orchestrator import InferenceOrchestrator\n",
      "metadata": {
        "module": "ocr.core.inference.orchestrator",
        "name": "InferenceOrchestrator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/validation_agent.py",
      "line": 15,
      "column": 0,
      "pattern": "from ocr.agents.base_agent import LLMAgent",
      "context": "Imports LLMAgent from ocr.agents.base_agent",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.base_agent import LLMAgent, AgentCapability\n",
      "metadata": {
        "module": "ocr.agents.base_agent",
        "name": "LLMAgent",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/validation_agent.py",
      "line": 15,
      "column": 0,
      "pattern": "from ocr.agents.base_agent import AgentCapability",
      "context": "Imports AgentCapability from ocr.agents.base_agent",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.agents.base_agent import LLMAgent, AgentCapability\n",
      "metadata": {
        "module": "ocr.agents.base_agent",
        "name": "AgentCapability",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/agents/validation_agent.py",
      "line": 266,
      "column": 16,
      "pattern": "from difflib import SequenceMatcher",
      "context": "Imports SequenceMatcher from difflib",
      "category": "from_import",
      "code_snippet": "            if ground_truth:\n                from difflib import SequenceMatcher\n                similarity = SequenceMatcher(None, full_text, ground_truth).ratio()",
      "metadata": {
        "module": "difflib",
        "name": "SequenceMatcher",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/command_builder/compute.py",
      "line": 8,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/command_builder/models.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/command_builder/overrides.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/command_builder/overrides.py",
      "line": 5,
      "column": 0,
      "pattern": "from ocr.core.utils.config import ConfigParser",
      "context": "Imports ConfigParser from ocr.core.utils.config",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config import ConfigParser\n",
      "metadata": {
        "module": "ocr.core.utils.config",
        "name": "ConfigParser",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/command_builder/recommendations.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/command_builder/recommendations.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.core.utils.config import ConfigParser",
      "context": "Imports ConfigParser from ocr.core.utils.config",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config import ConfigParser\n",
      "metadata": {
        "module": "ocr.core.utils.config",
        "name": "ConfigParser",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/command_builder/recommendations.py",
      "line": 9,
      "column": 0,
      "pattern": "from models import UseCaseRecommendation",
      "context": "Imports UseCaseRecommendation from models",
      "category": "from_import",
      "code_snippet": "\nfrom .models import UseCaseRecommendation\n",
      "metadata": {
        "module": "models",
        "name": "UseCaseRecommendation",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/communication/rabbitmq_transport.py",
      "line": 4,
      "column": 0,
      "pattern": "import pika",
      "context": "Imports module: pika",
      "category": "import",
      "code_snippet": "import time\nimport pika\nimport logging",
      "metadata": {
        "module": "pika",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/communication/slack_service.py",
      "line": 3,
      "column": 0,
      "pattern": "import httpx",
      "context": "Imports module: httpx",
      "category": "import",
      "code_snippet": "import logging\nimport httpx\n",
      "metadata": {
        "module": "httpx",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from interfaces.losses import BaseLoss",
      "context": "Imports BaseLoss from interfaces.losses",
      "category": "from_import",
      "code_snippet": "\nfrom .interfaces.losses import BaseLoss\nfrom .interfaces.metrics import BaseMetric",
      "metadata": {
        "module": "interfaces.losses",
        "name": "BaseLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from interfaces.metrics import BaseMetric",
      "context": "Imports BaseMetric from interfaces.metrics",
      "category": "from_import",
      "code_snippet": "from .interfaces.losses import BaseLoss\nfrom .interfaces.metrics import BaseMetric\nfrom .interfaces.models import BaseDecoder, BaseEncoder, BaseHead",
      "metadata": {
        "module": "interfaces.metrics",
        "name": "BaseMetric",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/__init__.py",
      "line": 5,
      "column": 0,
      "pattern": "from interfaces.models import BaseDecoder",
      "context": "Imports BaseDecoder from interfaces.models",
      "category": "from_import",
      "code_snippet": "from .interfaces.metrics import BaseMetric\nfrom .interfaces.models import BaseDecoder, BaseEncoder, BaseHead\nfrom .utils.registry import ComponentRegistry, get_registry, registry",
      "metadata": {
        "module": "interfaces.models",
        "name": "BaseDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/__init__.py",
      "line": 5,
      "column": 0,
      "pattern": "from interfaces.models import BaseEncoder",
      "context": "Imports BaseEncoder from interfaces.models",
      "category": "from_import",
      "code_snippet": "from .interfaces.metrics import BaseMetric\nfrom .interfaces.models import BaseDecoder, BaseEncoder, BaseHead\nfrom .utils.registry import ComponentRegistry, get_registry, registry",
      "metadata": {
        "module": "interfaces.models",
        "name": "BaseEncoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/__init__.py",
      "line": 5,
      "column": 0,
      "pattern": "from interfaces.models import BaseHead",
      "context": "Imports BaseHead from interfaces.models",
      "category": "from_import",
      "code_snippet": "from .interfaces.metrics import BaseMetric\nfrom .interfaces.models import BaseDecoder, BaseEncoder, BaseHead\nfrom .utils.registry import ComponentRegistry, get_registry, registry",
      "metadata": {
        "module": "interfaces.models",
        "name": "BaseHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from utils.registry import ComponentRegistry",
      "context": "Imports ComponentRegistry from utils.registry",
      "category": "from_import",
      "code_snippet": "from .interfaces.models import BaseDecoder, BaseEncoder, BaseHead\nfrom .utils.registry import ComponentRegistry, get_registry, registry\n",
      "metadata": {
        "module": "utils.registry",
        "name": "ComponentRegistry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from utils.registry import get_registry",
      "context": "Imports get_registry from utils.registry",
      "category": "from_import",
      "code_snippet": "from .interfaces.models import BaseDecoder, BaseEncoder, BaseHead\nfrom .utils.registry import ComponentRegistry, get_registry, registry\n",
      "metadata": {
        "module": "utils.registry",
        "name": "get_registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from utils.registry import registry",
      "context": "Imports registry from utils.registry",
      "category": "from_import",
      "code_snippet": "from .interfaces.models import BaseDecoder, BaseEncoder, BaseHead\nfrom .utils.registry import ComponentRegistry, get_registry, registry\n",
      "metadata": {
        "module": "utils.registry",
        "name": "registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from  import data",
      "context": "Imports data from ",
      "category": "from_import",
      "code_snippet": "\nfrom . import data, debugging, validation  # noqa: F401",
      "metadata": {
        "module": "",
        "name": "data",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from  import debugging",
      "context": "Imports debugging from ",
      "category": "from_import",
      "code_snippet": "\nfrom . import data, debugging, validation  # noqa: F401",
      "metadata": {
        "module": "",
        "name": "debugging",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from  import validation",
      "context": "Imports validation from ",
      "category": "from_import",
      "code_snippet": "\nfrom . import data, debugging, validation  # noqa: F401",
      "metadata": {
        "module": "",
        "name": "validation",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/calculate_normalization.py",
      "line": 12,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nimport torch",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/calculate_normalization.py",
      "line": 13,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import numpy as np\nimport torch\nfrom hydra import compose, initialize",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/calculate_normalization.py",
      "line": 14,
      "column": 0,
      "pattern": "from hydra import compose",
      "context": "Imports compose from hydra",
      "category": "from_import",
      "code_snippet": "import torch\nfrom hydra import compose, initialize\nfrom omegaconf import DictConfig",
      "metadata": {
        "module": "hydra",
        "name": "compose",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/calculate_normalization.py",
      "line": 14,
      "column": 0,
      "pattern": "from hydra import initialize",
      "context": "Imports initialize from hydra",
      "category": "from_import",
      "code_snippet": "import torch\nfrom hydra import compose, initialize\nfrom omegaconf import DictConfig",
      "metadata": {
        "module": "hydra",
        "name": "initialize",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/calculate_normalization.py",
      "line": 15,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "from hydra import compose, initialize\nfrom omegaconf import DictConfig\nfrom tqdm import tqdm",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/calculate_normalization.py",
      "line": 16,
      "column": 0,
      "pattern": "from tqdm import tqdm",
      "context": "Imports tqdm from tqdm",
      "category": "from_import",
      "code_snippet": "from omegaconf import DictConfig\nfrom tqdm import tqdm\n",
      "metadata": {
        "module": "tqdm",
        "name": "tqdm",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/calculate_normalization.py",
      "line": 18,
      "column": 0,
      "pattern": "from datasets import ValidatedOCRDataset",
      "context": "Imports ValidatedOCRDataset from datasets",
      "category": "from_import",
      "code_snippet": "\nfrom ...datasets import ValidatedOCRDataset\nfrom ...datasets.schemas import DatasetConfig",
      "metadata": {
        "module": "datasets",
        "name": "ValidatedOCRDataset",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/calculate_normalization.py",
      "line": 19,
      "column": 0,
      "pattern": "from datasets.schemas import DatasetConfig",
      "context": "Imports DatasetConfig from datasets.schemas",
      "category": "from_import",
      "code_snippet": "from ...datasets import ValidatedOCRDataset\nfrom ...datasets.schemas import DatasetConfig\nfrom ...datasets.transforms import DBTransforms",
      "metadata": {
        "module": "datasets.schemas",
        "name": "DatasetConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/calculate_normalization.py",
      "line": 20,
      "column": 0,
      "pattern": "from datasets.transforms import DBTransforms",
      "context": "Imports DBTransforms from datasets.transforms",
      "category": "from_import",
      "code_snippet": "from ...datasets.schemas import DatasetConfig\nfrom ...datasets.transforms import DBTransforms\n",
      "metadata": {
        "module": "datasets.transforms",
        "name": "DBTransforms",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/gen_image_metadata.py",
      "line": 19,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/gen_image_metadata.py",
      "line": 25,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "\nfrom PIL import Image, UnidentifiedImageError\nfrom tqdm import tqdm",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/gen_image_metadata.py",
      "line": 25,
      "column": 0,
      "pattern": "from PIL import UnidentifiedImageError",
      "context": "Imports UnidentifiedImageError from PIL",
      "category": "from_import",
      "code_snippet": "\nfrom PIL import Image, UnidentifiedImageError\nfrom tqdm import tqdm",
      "metadata": {
        "module": "PIL",
        "name": "UnidentifiedImageError",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/gen_image_metadata.py",
      "line": 26,
      "column": 0,
      "pattern": "from tqdm import tqdm",
      "context": "Imports tqdm from tqdm",
      "category": "from_import",
      "code_snippet": "from PIL import Image, UnidentifiedImageError\nfrom tqdm import tqdm\n",
      "metadata": {
        "module": "tqdm",
        "name": "tqdm",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/recommend_buckets.py",
      "line": 20,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/data/recommend_buckets.py",
      "line": 27,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/validation/analyze_worst_images.py",
      "line": 8,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport matplotlib.pyplot as plt",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/validation/analyze_worst_images.py",
      "line": 9,
      "column": 0,
      "pattern": "import matplotlib.pyplot",
      "context": "Imports module: matplotlib.pyplot",
      "category": "import",
      "code_snippet": "import cv2\nimport matplotlib.pyplot as plt\nimport numpy as np",
      "metadata": {
        "module": "matplotlib.pyplot",
        "alias": "plt"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/validation/analyze_worst_images.py",
      "line": 10,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/validation/analyze_worst_images.py",
      "line": 11,
      "column": 0,
      "pattern": "import pandas",
      "context": "Imports module: pandas",
      "category": "import",
      "code_snippet": "import numpy as np\nimport pandas as pd\n",
      "metadata": {
        "module": "pandas",
        "alias": "pd"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/validation/render_underperforming.py",
      "line": 13,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from evaluator import CLEvalEvaluator",
      "context": "Imports CLEvalEvaluator from evaluator",
      "category": "from_import",
      "code_snippet": "\nfrom .evaluator import CLEvalEvaluator\n",
      "metadata": {
        "module": "evaluator",
        "name": "CLEvalEvaluator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 11,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom PIL import Image",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 12,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image\nfrom tqdm import tqdm",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 13,
      "column": 0,
      "pattern": "from tqdm import tqdm",
      "context": "Imports tqdm from tqdm",
      "category": "from_import",
      "code_snippet": "from PIL import Image\nfrom tqdm import tqdm\n",
      "metadata": {
        "module": "tqdm",
        "name": "tqdm",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 15,
      "column": 0,
      "pattern": "from ocr.core.validation import LightningStepPrediction",
      "context": "Imports LightningStepPrediction from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import LightningStepPrediction, validate_predictions\nfrom ocr.core.metrics import CLEvalMetric",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "LightningStepPrediction",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 15,
      "column": 0,
      "pattern": "from ocr.core.validation import validate_predictions",
      "context": "Imports validate_predictions from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import LightningStepPrediction, validate_predictions\nfrom ocr.core.metrics import CLEvalMetric",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "validate_predictions",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core.metrics import CLEvalMetric",
      "context": "Imports CLEvalMetric from ocr.core.metrics",
      "category": "from_import",
      "code_snippet": "from ocr.core.validation import LightningStepPrediction, validate_predictions\nfrom ocr.core.metrics import CLEvalMetric\nfrom ocr.core.utils.logging import get_rich_console",
      "metadata": {
        "module": "ocr.core.metrics",
        "name": "CLEvalMetric",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 17,
      "column": 0,
      "pattern": "from ocr.core.utils.logging import get_rich_console",
      "context": "Imports get_rich_console from ocr.core.utils.logging",
      "category": "from_import",
      "code_snippet": "from ocr.core.metrics import CLEvalMetric\nfrom ocr.core.utils.logging import get_rich_console\nfrom ocr.core.utils.orientation import remap_polygons",
      "metadata": {
        "module": "ocr.core.utils.logging",
        "name": "get_rich_console",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 18,
      "column": 0,
      "pattern": "from ocr.core.utils.orientation import remap_polygons",
      "context": "Imports remap_polygons from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.logging import get_rich_console\nfrom ocr.core.utils.orientation import remap_polygons\n",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "remap_polygons",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 21,
      "column": 4,
      "pattern": "from rich.progress import BarColumn",
      "context": "Imports BarColumn from rich.progress",
      "category": "from_import",
      "code_snippet": "try:  # Rich is optional \u2013 fall back to tqdm when unavailable\n    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn\n",
      "metadata": {
        "module": "rich.progress",
        "name": "BarColumn",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 21,
      "column": 4,
      "pattern": "from rich.progress import Progress",
      "context": "Imports Progress from rich.progress",
      "category": "from_import",
      "code_snippet": "try:  # Rich is optional \u2013 fall back to tqdm when unavailable\n    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn\n",
      "metadata": {
        "module": "rich.progress",
        "name": "Progress",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 21,
      "column": 4,
      "pattern": "from rich.progress import TextColumn",
      "context": "Imports TextColumn from rich.progress",
      "category": "from_import",
      "code_snippet": "try:  # Rich is optional \u2013 fall back to tqdm when unavailable\n    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn\n",
      "metadata": {
        "module": "rich.progress",
        "name": "TextColumn",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py",
      "line": 21,
      "column": 4,
      "pattern": "from rich.progress import TimeElapsedColumn",
      "context": "Imports TimeElapsedColumn from rich.progress",
      "category": "from_import",
      "code_snippet": "try:  # Rich is optional \u2013 fall back to tqdm when unavailable\n    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn\n",
      "metadata": {
        "module": "rich.progress",
        "name": "TimeElapsedColumn",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from config_loader import ModelConfigBundle",
      "context": "Imports ModelConfigBundle from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import ModelConfigBundle, PostprocessSettings, PreprocessSettings\nfrom .engine import InferenceEngine, get_available_checkpoints, run_inference_on_image",
      "metadata": {
        "module": "config_loader",
        "name": "ModelConfigBundle",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from config_loader import PostprocessSettings",
      "context": "Imports PostprocessSettings from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import ModelConfigBundle, PostprocessSettings, PreprocessSettings\nfrom .engine import InferenceEngine, get_available_checkpoints, run_inference_on_image",
      "metadata": {
        "module": "config_loader",
        "name": "PostprocessSettings",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from config_loader import PreprocessSettings",
      "context": "Imports PreprocessSettings from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import ModelConfigBundle, PostprocessSettings, PreprocessSettings\nfrom .engine import InferenceEngine, get_available_checkpoints, run_inference_on_image",
      "metadata": {
        "module": "config_loader",
        "name": "PreprocessSettings",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from engine import InferenceEngine",
      "context": "Imports InferenceEngine from engine",
      "category": "from_import",
      "code_snippet": "from .config_loader import ModelConfigBundle, PostprocessSettings, PreprocessSettings\nfrom .engine import InferenceEngine, get_available_checkpoints, run_inference_on_image\n",
      "metadata": {
        "module": "engine",
        "name": "InferenceEngine",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from engine import get_available_checkpoints",
      "context": "Imports get_available_checkpoints from engine",
      "category": "from_import",
      "code_snippet": "from .config_loader import ModelConfigBundle, PostprocessSettings, PreprocessSettings\nfrom .engine import InferenceEngine, get_available_checkpoints, run_inference_on_image\n",
      "metadata": {
        "module": "engine",
        "name": "get_available_checkpoints",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from engine import run_inference_on_image",
      "context": "Imports run_inference_on_image from engine",
      "category": "from_import",
      "code_snippet": "from .config_loader import ModelConfigBundle, PostprocessSettings, PreprocessSettings\nfrom .engine import InferenceEngine, get_available_checkpoints, run_inference_on_image\n",
      "metadata": {
        "module": "engine",
        "name": "run_inference_on_image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/config_loader.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/config_loader.py",
      "line": 13,
      "column": 0,
      "pattern": "from dependencies import OCR_MODULES_AVAILABLE",
      "context": "Imports OCR_MODULES_AVAILABLE from dependencies",
      "category": "from_import",
      "code_snippet": "\nfrom .dependencies import OCR_MODULES_AVAILABLE, PROJECT_ROOT\n",
      "metadata": {
        "module": "dependencies",
        "name": "OCR_MODULES_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/config_loader.py",
      "line": 13,
      "column": 0,
      "pattern": "from dependencies import PROJECT_ROOT",
      "context": "Imports PROJECT_ROOT from dependencies",
      "category": "from_import",
      "code_snippet": "\nfrom .dependencies import OCR_MODULES_AVAILABLE, PROJECT_ROOT\n",
      "metadata": {
        "module": "dependencies",
        "name": "PROJECT_ROOT",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/config_loader.py",
      "line": 111,
      "column": 12,
      "pattern": "from ocr.core.utils.config_utils import load_config",
      "context": "Imports load_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "        try:\n            from ocr.core.utils.config_utils import load_config as load_hydra_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "load_config",
        "alias": "load_hydra_config"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/config_loader.py",
      "line": 125,
      "column": 8,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "            config_dict = json.load(handle)\n        from omegaconf import DictConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/config_loader.py",
      "line": 133,
      "column": 16,
      "pattern": "import yaml",
      "context": "Imports module: yaml",
      "category": "import",
      "code_snippet": "            if path.suffix in {\".yaml\", \".yml\"}:\n                import yaml\n",
      "metadata": {
        "module": "yaml",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/config_loader.py",
      "line": 140,
      "column": 8,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "                config_dict = json.load(handle)\n        from omegaconf import DictConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/config_loader.py",
      "line": 243,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/coordinate_manager.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/coordinate_manager.py",
      "line": 19,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/crop_extractor.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/crop_extractor.py",
      "line": 13,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/crop_extractor.py",
      "line": 14,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/dependencies.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/dependencies.py",
      "line": 9,
      "column": 0,
      "pattern": "from ocr.core.utils.path_utils import PROJECT_ROOT",
      "context": "Imports PROJECT_ROOT from ocr.core.utils.path_utils",
      "category": "from_import",
      "code_snippet": "# Import PROJECT_ROOT from central path utility (stable, works from any location)\nfrom ocr.core.utils.path_utils import PROJECT_ROOT\n",
      "metadata": {
        "module": "ocr.core.utils.path_utils",
        "name": "PROJECT_ROOT",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/engine.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/engine.py",
      "line": 14,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/engine.py",
      "line": 20,
      "column": 0,
      "pattern": "from dependencies import OCR_MODULES_AVAILABLE",
      "context": "Imports OCR_MODULES_AVAILABLE from dependencies",
      "category": "from_import",
      "code_snippet": "\nfrom .dependencies import OCR_MODULES_AVAILABLE\nfrom .image_loader import ImageLoader",
      "metadata": {
        "module": "dependencies",
        "name": "OCR_MODULES_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/engine.py",
      "line": 21,
      "column": 0,
      "pattern": "from image_loader import ImageLoader",
      "context": "Imports ImageLoader from image_loader",
      "category": "from_import",
      "code_snippet": "from .dependencies import OCR_MODULES_AVAILABLE\nfrom .image_loader import ImageLoader\nfrom .orchestrator import InferenceOrchestrator",
      "metadata": {
        "module": "image_loader",
        "name": "ImageLoader",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/engine.py",
      "line": 22,
      "column": 0,
      "pattern": "from orchestrator import InferenceOrchestrator",
      "context": "Imports InferenceOrchestrator from orchestrator",
      "category": "from_import",
      "code_snippet": "from .image_loader import ImageLoader\nfrom .orchestrator import InferenceOrchestrator\nfrom .utils import generate_mock_predictions",
      "metadata": {
        "module": "orchestrator",
        "name": "InferenceOrchestrator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/engine.py",
      "line": 23,
      "column": 0,
      "pattern": "from utils import generate_mock_predictions",
      "context": "Imports generate_mock_predictions from utils",
      "category": "from_import",
      "code_snippet": "from .orchestrator import InferenceOrchestrator\nfrom .utils import generate_mock_predictions\nfrom .utils import get_available_checkpoints as scan_checkpoints",
      "metadata": {
        "module": "utils",
        "name": "generate_mock_predictions",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/engine.py",
      "line": 24,
      "column": 0,
      "pattern": "from utils import get_available_checkpoints",
      "context": "Imports get_available_checkpoints from utils",
      "category": "from_import",
      "code_snippet": "from .utils import generate_mock_predictions\nfrom .utils import get_available_checkpoints as scan_checkpoints\n",
      "metadata": {
        "module": "utils",
        "name": "get_available_checkpoints",
        "alias": "scan_checkpoints"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/engine.py",
      "line": 310,
      "column": 8,
      "pattern": "from ocr.core.utils.orientation import remap_polygons",
      "context": "Imports remap_polygons from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "        # Import here to avoid circular dependencies\n        from ocr.core.utils.orientation import remap_polygons\n",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "remap_polygons",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/image_loader.py",
      "line": 9,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/image_loader.py",
      "line": 15,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/image_loader.py",
      "line": 16,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\nfrom PIL import Image",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/image_loader.py",
      "line": 17,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/image_loader.py",
      "line": 19,
      "column": 0,
      "pattern": "from ocr.core.utils.image_loading import load_image_optimized",
      "context": "Imports load_image_optimized from ocr.core.utils.image_loading",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.image_loading import load_image_optimized\nfrom ocr.core.utils.orientation import get_exif_orientation, normalize_pil_image",
      "metadata": {
        "module": "ocr.core.utils.image_loading",
        "name": "load_image_optimized",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/image_loader.py",
      "line": 20,
      "column": 0,
      "pattern": "from ocr.core.utils.orientation import get_exif_orientation",
      "context": "Imports get_exif_orientation from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.image_loading import load_image_optimized\nfrom ocr.core.utils.orientation import get_exif_orientation, normalize_pil_image\n",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "get_exif_orientation",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/image_loader.py",
      "line": 20,
      "column": 0,
      "pattern": "from ocr.core.utils.orientation import normalize_pil_image",
      "context": "Imports normalize_pil_image from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.image_loading import load_image_optimized\nfrom ocr.core.utils.orientation import get_exif_orientation, normalize_pil_image\n",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "normalize_pil_image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_loader.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_loader.py",
      "line": 9,
      "column": 0,
      "pattern": "from dependencies import OCR_MODULES_AVAILABLE",
      "context": "Imports OCR_MODULES_AVAILABLE from dependencies",
      "category": "from_import",
      "code_snippet": "\nfrom .dependencies import OCR_MODULES_AVAILABLE\n",
      "metadata": {
        "module": "dependencies",
        "name": "OCR_MODULES_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_loader.py",
      "line": 21,
      "column": 4,
      "pattern": "from ocr.core.models import get_model_by_cfg",
      "context": "Imports get_model_by_cfg from ocr.core.models",
      "category": "from_import",
      "code_snippet": "\n    from ocr.core.models import get_model_by_cfg\n",
      "metadata": {
        "module": "ocr.core.models",
        "name": "get_model_by_cfg",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_loader.py",
      "line": 38,
      "column": 4,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "    \"\"\"Load a checkpoint file into memory.\"\"\"\n    import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_loader.py",
      "line": 61,
      "column": 4,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "    \"\"\"Load a model state dict from checkpoint data.\"\"\"\n    import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_loader.py",
      "line": 103,
      "column": 4,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "def register_safe_globals() -> None:\n    import torch\n    from omegaconf import ListConfig",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_loader.py",
      "line": 104,
      "column": 4,
      "pattern": "from omegaconf import ListConfig",
      "context": "Imports ListConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "    import torch\n    from omegaconf import ListConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "ListConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_loader.py",
      "line": 110,
      "column": 8,
      "pattern": "from torch.serialization import add_safe_globals",
      "context": "Imports add_safe_globals from torch.serialization",
      "category": "from_import",
      "code_snippet": "    try:\n        from torch.serialization import add_safe_globals\n    except (ImportError, AttributeError):  # pragma: no cover - torch internals",
      "metadata": {
        "module": "torch.serialization",
        "name": "add_safe_globals",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 15,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 22,
      "column": 0,
      "pattern": "from ocr.core.utils.path_utils import get_path_resolver",
      "context": "Imports get_path_resolver from ocr.core.utils.path_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.path_utils import get_path_resolver\n",
      "metadata": {
        "module": "ocr.core.utils.path_utils",
        "name": "get_path_resolver",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 24,
      "column": 0,
      "pattern": "from config_loader import ModelConfigBundle",
      "context": "Imports ModelConfigBundle from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import ModelConfigBundle, load_model_config, resolve_config_path\nfrom .dependencies import OCR_MODULES_AVAILABLE",
      "metadata": {
        "module": "config_loader",
        "name": "ModelConfigBundle",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 24,
      "column": 0,
      "pattern": "from config_loader import load_model_config",
      "context": "Imports load_model_config from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import ModelConfigBundle, load_model_config, resolve_config_path\nfrom .dependencies import OCR_MODULES_AVAILABLE",
      "metadata": {
        "module": "config_loader",
        "name": "load_model_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 24,
      "column": 0,
      "pattern": "from config_loader import resolve_config_path",
      "context": "Imports resolve_config_path from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import ModelConfigBundle, load_model_config, resolve_config_path\nfrom .dependencies import OCR_MODULES_AVAILABLE",
      "metadata": {
        "module": "config_loader",
        "name": "resolve_config_path",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 25,
      "column": 0,
      "pattern": "from dependencies import OCR_MODULES_AVAILABLE",
      "context": "Imports OCR_MODULES_AVAILABLE from dependencies",
      "category": "from_import",
      "code_snippet": "from .config_loader import ModelConfigBundle, load_model_config, resolve_config_path\nfrom .dependencies import OCR_MODULES_AVAILABLE\nfrom .model_loader import instantiate_model, load_checkpoint, load_state_dict",
      "metadata": {
        "module": "dependencies",
        "name": "OCR_MODULES_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 26,
      "column": 0,
      "pattern": "from model_loader import instantiate_model",
      "context": "Imports instantiate_model from model_loader",
      "category": "from_import",
      "code_snippet": "from .dependencies import OCR_MODULES_AVAILABLE\nfrom .model_loader import instantiate_model, load_checkpoint, load_state_dict\n",
      "metadata": {
        "module": "model_loader",
        "name": "instantiate_model",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 26,
      "column": 0,
      "pattern": "from model_loader import load_checkpoint",
      "context": "Imports load_checkpoint from model_loader",
      "category": "from_import",
      "code_snippet": "from .dependencies import OCR_MODULES_AVAILABLE\nfrom .model_loader import instantiate_model, load_checkpoint, load_state_dict\n",
      "metadata": {
        "module": "model_loader",
        "name": "load_checkpoint",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 26,
      "column": 0,
      "pattern": "from model_loader import load_state_dict",
      "context": "Imports load_state_dict from model_loader",
      "category": "from_import",
      "code_snippet": "from .dependencies import OCR_MODULES_AVAILABLE\nfrom .model_loader import instantiate_model, load_checkpoint, load_state_dict\n",
      "metadata": {
        "module": "model_loader",
        "name": "load_state_dict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 53,
      "column": 12,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "        if device is None:\n            import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 185,
      "column": 8,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\n        import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/model_manager.py",
      "line": 194,
      "column": 12,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "            # Clear CUDA cache if available\n            import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 16,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 21,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 23,
      "column": 0,
      "pattern": "from config_loader import PostprocessSettings",
      "context": "Imports PostprocessSettings from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import PostprocessSettings\nfrom .dependencies import OCR_MODULES_AVAILABLE",
      "metadata": {
        "module": "config_loader",
        "name": "PostprocessSettings",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 24,
      "column": 0,
      "pattern": "from dependencies import OCR_MODULES_AVAILABLE",
      "context": "Imports OCR_MODULES_AVAILABLE from dependencies",
      "category": "from_import",
      "code_snippet": "from .config_loader import PostprocessSettings\nfrom .dependencies import OCR_MODULES_AVAILABLE\nfrom .model_manager import ModelManager",
      "metadata": {
        "module": "dependencies",
        "name": "OCR_MODULES_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 25,
      "column": 0,
      "pattern": "from model_manager import ModelManager",
      "context": "Imports ModelManager from model_manager",
      "category": "from_import",
      "code_snippet": "from .dependencies import OCR_MODULES_AVAILABLE\nfrom .model_manager import ModelManager\nfrom .postprocessing_pipeline import PostprocessingPipeline",
      "metadata": {
        "module": "model_manager",
        "name": "ModelManager",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 26,
      "column": 0,
      "pattern": "from postprocessing_pipeline import PostprocessingPipeline",
      "context": "Imports PostprocessingPipeline from postprocessing_pipeline",
      "category": "from_import",
      "code_snippet": "from .model_manager import ModelManager\nfrom .postprocessing_pipeline import PostprocessingPipeline\nfrom .preprocessing_pipeline import PreprocessingPipeline",
      "metadata": {
        "module": "postprocessing_pipeline",
        "name": "PostprocessingPipeline",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 27,
      "column": 0,
      "pattern": "from preprocessing_pipeline import PreprocessingPipeline",
      "context": "Imports PreprocessingPipeline from preprocessing_pipeline",
      "category": "from_import",
      "code_snippet": "from .postprocessing_pipeline import PostprocessingPipeline\nfrom .preprocessing_pipeline import PreprocessingPipeline\nfrom .preview_generator import PreviewGenerator",
      "metadata": {
        "module": "preprocessing_pipeline",
        "name": "PreprocessingPipeline",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 28,
      "column": 0,
      "pattern": "from preview_generator import PreviewGenerator",
      "context": "Imports PreviewGenerator from preview_generator",
      "category": "from_import",
      "code_snippet": "from .preprocessing_pipeline import PreprocessingPipeline\nfrom .preview_generator import PreviewGenerator\n",
      "metadata": {
        "module": "preview_generator",
        "name": "PreviewGenerator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 31,
      "column": 4,
      "pattern": "from ocr.features.kie.inference.extraction.field_extractor import ReceiptFieldExtractor",
      "context": "Imports ReceiptFieldExtractor from ocr.features.kie.inference.extraction.field_extractor",
      "category": "from_import",
      "code_snippet": "if TYPE_CHECKING:\n    from ocr.features.kie.inference.extraction.field_extractor import ReceiptFieldExtractor\n    from ocr.features.layout.inference.grouper import LineGrouper",
      "metadata": {
        "module": "ocr.features.kie.inference.extraction.field_extractor",
        "name": "ReceiptFieldExtractor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 32,
      "column": 4,
      "pattern": "from ocr.features.layout.inference.grouper import LineGrouper",
      "context": "Imports LineGrouper from ocr.features.layout.inference.grouper",
      "category": "from_import",
      "code_snippet": "    from ocr.features.kie.inference.extraction.field_extractor import ReceiptFieldExtractor\n    from ocr.features.layout.inference.grouper import LineGrouper\n",
      "metadata": {
        "module": "ocr.features.layout.inference.grouper",
        "name": "LineGrouper",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 34,
      "column": 4,
      "pattern": "from crop_extractor import CropExtractor",
      "context": "Imports CropExtractor from crop_extractor",
      "category": "from_import",
      "code_snippet": "\n    from .crop_extractor import CropExtractor\n    from .recognizer import TextRecognizer",
      "metadata": {
        "module": "crop_extractor",
        "name": "CropExtractor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 35,
      "column": 4,
      "pattern": "from recognizer import TextRecognizer",
      "context": "Imports TextRecognizer from recognizer",
      "category": "from_import",
      "code_snippet": "    from .crop_extractor import CropExtractor\n    from .recognizer import TextRecognizer\n",
      "metadata": {
        "module": "recognizer",
        "name": "TextRecognizer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 114,
      "column": 12,
      "pattern": "from crop_extractor import CropConfig",
      "context": "Imports CropConfig from crop_extractor",
      "category": "from_import",
      "code_snippet": "        try:\n            from .crop_extractor import CropConfig, CropExtractor\n            from .recognizer import RecognizerConfig, TextRecognizer",
      "metadata": {
        "module": "crop_extractor",
        "name": "CropConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 114,
      "column": 12,
      "pattern": "from crop_extractor import CropExtractor",
      "context": "Imports CropExtractor from crop_extractor",
      "category": "from_import",
      "code_snippet": "        try:\n            from .crop_extractor import CropConfig, CropExtractor\n            from .recognizer import RecognizerConfig, TextRecognizer",
      "metadata": {
        "module": "crop_extractor",
        "name": "CropExtractor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 115,
      "column": 12,
      "pattern": "from recognizer import RecognizerConfig",
      "context": "Imports RecognizerConfig from recognizer",
      "category": "from_import",
      "code_snippet": "            from .crop_extractor import CropConfig, CropExtractor\n            from .recognizer import RecognizerConfig, TextRecognizer\n",
      "metadata": {
        "module": "recognizer",
        "name": "RecognizerConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 115,
      "column": 12,
      "pattern": "from recognizer import TextRecognizer",
      "context": "Imports TextRecognizer from recognizer",
      "category": "from_import",
      "code_snippet": "            from .crop_extractor import CropConfig, CropExtractor\n            from .recognizer import RecognizerConfig, TextRecognizer\n",
      "metadata": {
        "module": "recognizer",
        "name": "TextRecognizer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 133,
      "column": 12,
      "pattern": "from ocr.features.kie.inference.extraction.field_extractor import ExtractorConfig",
      "context": "Imports ExtractorConfig from ocr.features.kie.inference.extraction.field_extractor",
      "category": "from_import",
      "code_snippet": "        try:\n            from ocr.features.kie.inference.extraction.field_extractor import ExtractorConfig, ReceiptFieldExtractor\n            from ocr.features.layout.inference.grouper import LineGrouper, LineGrouperConfig",
      "metadata": {
        "module": "ocr.features.kie.inference.extraction.field_extractor",
        "name": "ExtractorConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 133,
      "column": 12,
      "pattern": "from ocr.features.kie.inference.extraction.field_extractor import ReceiptFieldExtractor",
      "context": "Imports ReceiptFieldExtractor from ocr.features.kie.inference.extraction.field_extractor",
      "category": "from_import",
      "code_snippet": "        try:\n            from ocr.features.kie.inference.extraction.field_extractor import ExtractorConfig, ReceiptFieldExtractor\n            from ocr.features.layout.inference.grouper import LineGrouper, LineGrouperConfig",
      "metadata": {
        "module": "ocr.features.kie.inference.extraction.field_extractor",
        "name": "ReceiptFieldExtractor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 134,
      "column": 12,
      "pattern": "from ocr.features.layout.inference.grouper import LineGrouper",
      "context": "Imports LineGrouper from ocr.features.layout.inference.grouper",
      "category": "from_import",
      "code_snippet": "            from ocr.features.kie.inference.extraction.field_extractor import ExtractorConfig, ReceiptFieldExtractor\n            from ocr.features.layout.inference.grouper import LineGrouper, LineGrouperConfig\n",
      "metadata": {
        "module": "ocr.features.layout.inference.grouper",
        "name": "LineGrouper",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 134,
      "column": 12,
      "pattern": "from ocr.features.layout.inference.grouper import LineGrouperConfig",
      "context": "Imports LineGrouperConfig from ocr.features.layout.inference.grouper",
      "category": "from_import",
      "code_snippet": "            from ocr.features.kie.inference.extraction.field_extractor import ExtractorConfig, ReceiptFieldExtractor\n            from ocr.features.layout.inference.grouper import LineGrouper, LineGrouperConfig\n",
      "metadata": {
        "module": "ocr.features.layout.inference.grouper",
        "name": "LineGrouperConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 228,
      "column": 12,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "        try:\n            import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 280,
      "column": 12,
      "pattern": "from ocr.core.utils.perspective_correction import transform_polygons_inverse",
      "context": "Imports transform_polygons_inverse from ocr.core.utils.perspective_correction",
      "category": "from_import",
      "code_snippet": "            # Transform polygons back to original space\n            from ocr.core.utils.perspective_correction import transform_polygons_inverse\n",
      "metadata": {
        "module": "ocr.core.utils.perspective_correction",
        "name": "transform_polygons_inverse",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 393,
      "column": 12,
      "pattern": "from recognizer import RecognitionInput",
      "context": "Imports RecognitionInput from recognizer",
      "category": "from_import",
      "code_snippet": "        try:\n            from .recognizer import RecognitionInput\n",
      "metadata": {
        "module": "recognizer",
        "name": "RecognitionInput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 466,
      "column": 8,
      "pattern": "from ocr.features.layout.inference.contracts import BoundingBox",
      "context": "Imports BoundingBox from ocr.features.layout.inference.contracts",
      "category": "from_import",
      "code_snippet": "\n        from ocr.features.layout.inference.contracts import BoundingBox, TextElement\n",
      "metadata": {
        "module": "ocr.features.layout.inference.contracts",
        "name": "BoundingBox",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 466,
      "column": 8,
      "pattern": "from ocr.features.layout.inference.contracts import TextElement",
      "context": "Imports TextElement from ocr.features.layout.inference.contracts",
      "category": "from_import",
      "code_snippet": "\n        from ocr.features.layout.inference.contracts import BoundingBox, TextElement\n",
      "metadata": {
        "module": "ocr.features.layout.inference.contracts",
        "name": "TextElement",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 589,
      "column": 12,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "        try:\n            import cv2\n            from PIL import Image",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 590,
      "column": 12,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "            import cv2\n            from PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/orchestrator.py",
      "line": 592,
      "column": 12,
      "pattern": "from ocr.features.kie.inference.extraction.vlm_extractor import VLMExtractor",
      "context": "Imports VLMExtractor from ocr.features.kie.inference.extraction.vlm_extractor",
      "category": "from_import",
      "code_snippet": "\n            from ocr.features.kie.inference.extraction.vlm_extractor import VLMExtractor\n",
      "metadata": {
        "module": "ocr.features.kie.inference.extraction.vlm_extractor",
        "name": "VLMExtractor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocess.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocess.py",
      "line": 9,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocess.py",
      "line": 10,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocess.py",
      "line": 12,
      "column": 0,
      "pattern": "from config_loader import PostprocessSettings",
      "context": "Imports PostprocessSettings from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import PostprocessSettings\n",
      "metadata": {
        "module": "config_loader",
        "name": "PostprocessSettings",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocess.py",
      "line": 42,
      "column": 4,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "    \"\"\"\n    import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocess.py",
      "line": 48,
      "column": 4,
      "pattern": "from coordinate_manager import compute_inverse_matrix",
      "context": "Imports compute_inverse_matrix from coordinate_manager",
      "category": "from_import",
      "code_snippet": "    # Use the new coordinate_manager for consistent transformation logic\n    from .coordinate_manager import compute_inverse_matrix as _compute_inverse_matrix\n",
      "metadata": {
        "module": "coordinate_manager",
        "name": "compute_inverse_matrix",
        "alias": "_compute_inverse_matrix"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocess.py",
      "line": 110,
      "column": 4,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\n    import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocess.py",
      "line": 119,
      "column": 4,
      "pattern": "from coordinate_manager import calculate_transform_metadata",
      "context": "Imports calculate_transform_metadata from coordinate_manager",
      "category": "from_import",
      "code_snippet": "    # Use coordinate_manager for consistent transformation logic\n    from .coordinate_manager import calculate_transform_metadata\n",
      "metadata": {
        "module": "coordinate_manager",
        "name": "calculate_transform_metadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocessing_pipeline.py",
      "line": 13,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocessing_pipeline.py",
      "line": 19,
      "column": 0,
      "pattern": "from config_loader import PostprocessSettings",
      "context": "Imports PostprocessSettings from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import PostprocessSettings\nfrom .postprocess import decode_polygons_with_head, fallback_postprocess",
      "metadata": {
        "module": "config_loader",
        "name": "PostprocessSettings",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocessing_pipeline.py",
      "line": 20,
      "column": 0,
      "pattern": "from postprocess import decode_polygons_with_head",
      "context": "Imports decode_polygons_with_head from postprocess",
      "category": "from_import",
      "code_snippet": "from .config_loader import PostprocessSettings\nfrom .postprocess import decode_polygons_with_head, fallback_postprocess\n",
      "metadata": {
        "module": "postprocess",
        "name": "decode_polygons_with_head",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/postprocessing_pipeline.py",
      "line": 20,
      "column": 0,
      "pattern": "from postprocess import fallback_postprocess",
      "context": "Imports fallback_postprocess from postprocess",
      "category": "from_import",
      "code_snippet": "from .config_loader import PostprocessSettings\nfrom .postprocess import decode_polygons_with_head, fallback_postprocess\n",
      "metadata": {
        "module": "postprocess",
        "name": "fallback_postprocess",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 9,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\n",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core.utils.background_normalization import normalize_gray_world",
      "context": "Imports normalize_gray_world from ocr.core.utils.background_normalization",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.background_normalization import normalize_gray_world\nfrom ocr.core.utils.perspective_correction import (",
      "metadata": {
        "module": "ocr.core.utils.background_normalization",
        "name": "normalize_gray_world",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.utils.perspective_correction import correct_perspective_from_mask",
      "context": "Imports correct_perspective_from_mask from ocr.core.utils.perspective_correction",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.background_normalization import normalize_gray_world\nfrom ocr.core.utils.perspective_correction import (\n    correct_perspective_from_mask,",
      "metadata": {
        "module": "ocr.core.utils.perspective_correction",
        "name": "correct_perspective_from_mask",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.utils.perspective_correction import remove_background_and_mask",
      "context": "Imports remove_background_and_mask from ocr.core.utils.perspective_correction",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.background_normalization import normalize_gray_world\nfrom ocr.core.utils.perspective_correction import (\n    correct_perspective_from_mask,",
      "metadata": {
        "module": "ocr.core.utils.perspective_correction",
        "name": "remove_background_and_mask",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core.utils.sepia_enhancement import enhance_clahe",
      "context": "Imports enhance_clahe from ocr.core.utils.sepia_enhancement",
      "category": "from_import",
      "code_snippet": ")\nfrom ocr.core.utils.sepia_enhancement import enhance_clahe, enhance_sepia\n",
      "metadata": {
        "module": "ocr.core.utils.sepia_enhancement",
        "name": "enhance_clahe",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core.utils.sepia_enhancement import enhance_sepia",
      "context": "Imports enhance_sepia from ocr.core.utils.sepia_enhancement",
      "category": "from_import",
      "code_snippet": ")\nfrom ocr.core.utils.sepia_enhancement import enhance_clahe, enhance_sepia\n",
      "metadata": {
        "module": "ocr.core.utils.sepia_enhancement",
        "name": "enhance_sepia",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 18,
      "column": 0,
      "pattern": "from config_loader import PreprocessSettings",
      "context": "Imports PreprocessSettings from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import PreprocessSettings\n",
      "metadata": {
        "module": "config_loader",
        "name": "PreprocessSettings",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 32,
      "column": 4,
      "pattern": "import torchvision.transforms",
      "context": "Imports module: torchvision.transforms",
      "category": "import",
      "code_snippet": "    \"\"\"\n    import torchvision.transforms as transforms\n",
      "metadata": {
        "module": "torchvision.transforms",
        "alias": "transforms"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 140,
      "column": 4,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "    tensor = transform(image_rgb)\n    import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 171,
      "column": 12,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "        if return_matrix:\n            import numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocess.py",
      "line": 193,
      "column": 12,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "        if return_matrix:\n            import numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_metadata.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_metadata.py",
      "line": 14,
      "column": 0,
      "pattern": "from coordinate_manager import calculate_transform_metadata",
      "context": "Imports calculate_transform_metadata from coordinate_manager",
      "category": "from_import",
      "code_snippet": "\nfrom .coordinate_manager import calculate_transform_metadata\n",
      "metadata": {
        "module": "coordinate_manager",
        "name": "calculate_transform_metadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 13,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 20,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 22,
      "column": 0,
      "pattern": "from config_loader import PreprocessSettings",
      "context": "Imports PreprocessSettings from config_loader",
      "category": "from_import",
      "code_snippet": "\nfrom .config_loader import PreprocessSettings\nfrom .preprocess import apply_optional_perspective_correction, build_transform, preprocess_image",
      "metadata": {
        "module": "config_loader",
        "name": "PreprocessSettings",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 23,
      "column": 0,
      "pattern": "from preprocess import apply_optional_perspective_correction",
      "context": "Imports apply_optional_perspective_correction from preprocess",
      "category": "from_import",
      "code_snippet": "from .config_loader import PreprocessSettings\nfrom .preprocess import apply_optional_perspective_correction, build_transform, preprocess_image\nfrom .preprocessing_metadata import create_preprocessing_metadata",
      "metadata": {
        "module": "preprocess",
        "name": "apply_optional_perspective_correction",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 23,
      "column": 0,
      "pattern": "from preprocess import build_transform",
      "context": "Imports build_transform from preprocess",
      "category": "from_import",
      "code_snippet": "from .config_loader import PreprocessSettings\nfrom .preprocess import apply_optional_perspective_correction, build_transform, preprocess_image\nfrom .preprocessing_metadata import create_preprocessing_metadata",
      "metadata": {
        "module": "preprocess",
        "name": "build_transform",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 23,
      "column": 0,
      "pattern": "from preprocess import preprocess_image",
      "context": "Imports preprocess_image from preprocess",
      "category": "from_import",
      "code_snippet": "from .config_loader import PreprocessSettings\nfrom .preprocess import apply_optional_perspective_correction, build_transform, preprocess_image\nfrom .preprocessing_metadata import create_preprocessing_metadata",
      "metadata": {
        "module": "preprocess",
        "name": "preprocess_image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 24,
      "column": 0,
      "pattern": "from preprocessing_metadata import create_preprocessing_metadata",
      "context": "Imports create_preprocessing_metadata from preprocessing_metadata",
      "category": "from_import",
      "code_snippet": "from .preprocess import apply_optional_perspective_correction, build_transform, preprocess_image\nfrom .preprocessing_metadata import create_preprocessing_metadata\n",
      "metadata": {
        "module": "preprocessing_metadata",
        "name": "create_preprocessing_metadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 158,
      "column": 16,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "            try:\n                import cv2\n                from rembg import remove",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 159,
      "column": 16,
      "pattern": "from rembg import remove",
      "context": "Imports remove from rembg",
      "category": "from_import",
      "code_snippet": "                import cv2\n                from rembg import remove\n",
      "metadata": {
        "module": "rembg",
        "name": "remove",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preprocessing_pipeline.py",
      "line": 210,
      "column": 12,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "        if enable_grayscale:\n            import cv2\n",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preview_generator.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preview_generator.py",
      "line": 13,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preview_generator.py",
      "line": 14,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preview_generator.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/preview_generator.py",
      "line": 185,
      "column": 12,
      "pattern": "from coordinate_manager import transform_polygons_string_to_processed_space",
      "context": "Imports transform_polygons_string_to_processed_space from coordinate_manager",
      "category": "from_import",
      "code_snippet": "        try:\n            from .coordinate_manager import transform_polygons_string_to_processed_space\n",
      "metadata": {
        "module": "coordinate_manager",
        "name": "transform_polygons_string_to_processed_space",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/utils.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/utils.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/inference/utils.py",
      "line": 10,
      "column": 0,
      "pattern": "from dependencies import PROJECT_ROOT",
      "context": "Imports PROJECT_ROOT from dependencies",
      "category": "from_import",
      "code_snippet": "\nfrom .dependencies import PROJECT_ROOT\n",
      "metadata": {
        "module": "dependencies",
        "name": "PROJECT_ROOT",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/interfaces/losses.py",
      "line": 5,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "from typing import Any\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/interfaces/losses.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\n",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/interfaces/metrics.py",
      "line": 5,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/interfaces/metrics.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\n",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/interfaces/models.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/interfaces/models.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\n",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from ocr.data.datasets import get_datasets_by_cfg",
      "context": "Imports get_datasets_by_cfg from ocr.data.datasets",
      "category": "from_import",
      "code_snippet": "from ocr.data.datasets import get_datasets_by_cfg\nfrom ocr.core.models import get_model_by_cfg",
      "metadata": {
        "module": "ocr.data.datasets",
        "name": "get_datasets_by_cfg",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/__init__.py",
      "line": 2,
      "column": 0,
      "pattern": "from ocr.core.models import get_model_by_cfg",
      "context": "Imports get_model_by_cfg from ocr.core.models",
      "category": "from_import",
      "code_snippet": "from ocr.data.datasets import get_datasets_by_cfg\nfrom ocr.core.models import get_model_by_cfg\n",
      "metadata": {
        "module": "ocr.core.models",
        "name": "get_model_by_cfg",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from ocr_pl import OCRDataPLModule",
      "context": "Imports OCRDataPLModule from ocr_pl",
      "category": "from_import",
      "code_snippet": "\nfrom .ocr_pl import OCRDataPLModule, OCRPLModule\n",
      "metadata": {
        "module": "ocr_pl",
        "name": "OCRDataPLModule",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from ocr_pl import OCRPLModule",
      "context": "Imports OCRPLModule from ocr_pl",
      "category": "from_import",
      "code_snippet": "\nfrom .ocr_pl import OCRDataPLModule, OCRPLModule\n",
      "metadata": {
        "module": "ocr_pl",
        "name": "OCRPLModule",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/__init__.py",
      "line": 8,
      "column": 4,
      "pattern": "import hydra",
      "context": "Imports module: hydra",
      "category": "import",
      "code_snippet": "def get_pl_modules_by_cfg(config):\n    import hydra\n    # Inject vocab size into model config if needed",
      "metadata": {
        "module": "hydra",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from ocr.core.lightning.callbacks.metadata_callback import MetadataCallback",
      "context": "Imports MetadataCallback from ocr.core.lightning.callbacks.metadata_callback",
      "category": "from_import",
      "code_snippet": "from ocr.core.lightning.callbacks.metadata_callback import MetadataCallback\nfrom ocr.core.lightning.callbacks.performance_profiler import PerformanceProfilerCallback",
      "metadata": {
        "module": "ocr.core.lightning.callbacks.metadata_callback",
        "name": "MetadataCallback",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/__init__.py",
      "line": 2,
      "column": 0,
      "pattern": "from ocr.core.lightning.callbacks.performance_profiler import PerformanceProfilerCallback",
      "context": "Imports PerformanceProfilerCallback from ocr.core.lightning.callbacks.performance_profiler",
      "category": "from_import",
      "code_snippet": "from ocr.core.lightning.callbacks.metadata_callback import MetadataCallback\nfrom ocr.core.lightning.callbacks.performance_profiler import PerformanceProfilerCallback\n",
      "metadata": {
        "module": "ocr.core.lightning.callbacks.performance_profiler",
        "name": "PerformanceProfilerCallback",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 13,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 20,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nfrom lightning.pytorch.callbacks import Callback",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 21,
      "column": 0,
      "pattern": "from lightning.pytorch.callbacks import Callback",
      "context": "Imports Callback from lightning.pytorch.callbacks",
      "category": "from_import",
      "code_snippet": "import torch\nfrom lightning.pytorch.callbacks import Callback\n",
      "metadata": {
        "module": "lightning.pytorch.callbacks",
        "name": "Callback",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 24,
      "column": 4,
      "pattern": "from ocr.core.utils.checkpoints.types import CheckpointingConfig",
      "context": "Imports CheckpointingConfig from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "if TYPE_CHECKING:\n    from ocr.core.utils.checkpoints.types import (\n        CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "CheckpointingConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 24,
      "column": 4,
      "pattern": "from ocr.core.utils.checkpoints.types import MetricsInfo",
      "context": "Imports MetricsInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "if TYPE_CHECKING:\n    from ocr.core.utils.checkpoints.types import (\n        CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "MetricsInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 24,
      "column": 4,
      "pattern": "from ocr.core.utils.checkpoints.types import ModelInfo",
      "context": "Imports ModelInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "if TYPE_CHECKING:\n    from ocr.core.utils.checkpoints.types import (\n        CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "ModelInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 193,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.metadata_loader import save_metadata",
      "context": "Imports save_metadata from ocr.core.utils.checkpoints.metadata_loader",
      "category": "from_import",
      "code_snippet": "            # Import metadata types and save function\n            from ocr.core.utils.checkpoints.metadata_loader import save_metadata\n            from ocr.core.utils.checkpoints.types import (",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.metadata_loader",
        "name": "save_metadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 194,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import CheckpointMetadataV1",
      "context": "Imports CheckpointMetadataV1 from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            from ocr.core.utils.checkpoints.metadata_loader import save_metadata\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointMetadataV1,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "CheckpointMetadataV1",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 194,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import TrainingInfo",
      "context": "Imports TrainingInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            from ocr.core.utils.checkpoints.metadata_loader import save_metadata\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointMetadataV1,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "TrainingInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 256,
      "column": 16,
      "pattern": "from ocr.core.utils.checkpoints.index import CheckpointIndex",
      "context": "Imports CheckpointIndex from ocr.core.utils.checkpoints.index",
      "category": "from_import",
      "code_snippet": "            try:\n                from ocr.core.utils.checkpoints.index import CheckpointIndex\n",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.index",
        "name": "CheckpointIndex",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 283,
      "column": 8,
      "pattern": "from ocr.core.utils.checkpoints.types import DecoderInfo",
      "context": "Imports DecoderInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from ocr.core.utils.checkpoints.types import (\n            DecoderInfo,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "DecoderInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 283,
      "column": 8,
      "pattern": "from ocr.core.utils.checkpoints.types import EncoderInfo",
      "context": "Imports EncoderInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from ocr.core.utils.checkpoints.types import (\n            DecoderInfo,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "EncoderInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 283,
      "column": 8,
      "pattern": "from ocr.core.utils.checkpoints.types import HeadInfo",
      "context": "Imports HeadInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from ocr.core.utils.checkpoints.types import (\n            DecoderInfo,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "HeadInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 283,
      "column": 8,
      "pattern": "from ocr.core.utils.checkpoints.types import LossInfo",
      "context": "Imports LossInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from ocr.core.utils.checkpoints.types import (\n            DecoderInfo,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "LossInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 283,
      "column": 8,
      "pattern": "from ocr.core.utils.checkpoints.types import ModelInfo",
      "context": "Imports ModelInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from ocr.core.utils.checkpoints.types import (\n            DecoderInfo,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "ModelInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 391,
      "column": 8,
      "pattern": "from ocr.core.utils.checkpoints.types import MetricsInfo",
      "context": "Imports MetricsInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from ocr.core.utils.checkpoints.types import MetricsInfo\n",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "MetricsInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 464,
      "column": 8,
      "pattern": "from ocr.core.utils.checkpoints.types import CheckpointingConfig",
      "context": "Imports CheckpointingConfig from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from ocr.core.utils.checkpoints.types import CheckpointingConfig\n",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "CheckpointingConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/metadata_callback.py",
      "line": 514,
      "column": 12,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "        try:\n            import wandb  # type: ignore[import-untyped]\n",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/multi_line_progress_bar.py",
      "line": 2,
      "column": 0,
      "pattern": "from lightning.pytorch.callbacks import RichProgressBar",
      "context": "Imports RichProgressBar from lightning.pytorch.callbacks",
      "category": "from_import",
      "code_snippet": "# Custom progress bar with line breaks for better readability\nfrom lightning.pytorch.callbacks import RichProgressBar\n",
      "metadata": {
        "module": "lightning.pytorch.callbacks",
        "name": "RichProgressBar",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/performance_profiler.py",
      "line": 6,
      "column": 0,
      "pattern": "import psutil",
      "context": "Imports module: psutil",
      "category": "import",
      "code_snippet": "\nimport psutil\nimport torch",
      "metadata": {
        "module": "psutil",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/performance_profiler.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import psutil\nimport torch\nfrom lightning.pytorch import LightningModule, Trainer",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/performance_profiler.py",
      "line": 8,
      "column": 0,
      "pattern": "from lightning.pytorch import LightningModule",
      "context": "Imports LightningModule from lightning.pytorch",
      "category": "from_import",
      "code_snippet": "import torch\nfrom lightning.pytorch import LightningModule, Trainer\nfrom lightning.pytorch.callbacks import Callback",
      "metadata": {
        "module": "lightning.pytorch",
        "name": "LightningModule",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/performance_profiler.py",
      "line": 8,
      "column": 0,
      "pattern": "from lightning.pytorch import Trainer",
      "context": "Imports Trainer from lightning.pytorch",
      "category": "from_import",
      "code_snippet": "import torch\nfrom lightning.pytorch import LightningModule, Trainer\nfrom lightning.pytorch.callbacks import Callback",
      "metadata": {
        "module": "lightning.pytorch",
        "name": "Trainer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/performance_profiler.py",
      "line": 9,
      "column": 0,
      "pattern": "from lightning.pytorch.callbacks import Callback",
      "context": "Imports Callback from lightning.pytorch.callbacks",
      "category": "from_import",
      "code_snippet": "from lightning.pytorch import LightningModule, Trainer\nfrom lightning.pytorch.callbacks import Callback\n",
      "metadata": {
        "module": "lightning.pytorch.callbacks",
        "name": "Callback",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/performance_profiler.py",
      "line": 12,
      "column": 4,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "try:\n    import wandb\n",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/performance_profiler.py",
      "line": 167,
      "column": 8,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\n        import numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/performance_profiler.py",
      "line": 222,
      "column": 8,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\n        import numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nfrom lightning.pytorch.callbacks import ModelCheckpoint",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 8,
      "column": 0,
      "pattern": "from lightning.pytorch.callbacks import ModelCheckpoint",
      "context": "Imports ModelCheckpoint from lightning.pytorch.callbacks",
      "category": "from_import",
      "code_snippet": "import torch\nfrom lightning.pytorch.callbacks import ModelCheckpoint\n",
      "metadata": {
        "module": "lightning.pytorch.callbacks",
        "name": "ModelCheckpoint",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 10,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import ensure_dict",
      "context": "Imports ensure_dict from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import ensure_dict, is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "ensure_dict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 10,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import ensure_dict, is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 196,
      "column": 20,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "                try:\n                    import numpy as np  # type: ignore\n                except Exception:  # pragma: no cover - optional dependency",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 201,
      "column": 20,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "                try:\n                    from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 201,
      "column": 20,
      "pattern": "from omegaconf import ListConfig",
      "context": "Imports ListConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "                try:\n                    from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore\n",
      "metadata": {
        "module": "omegaconf",
        "name": "ListConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 201,
      "column": 20,
      "pattern": "from omegaconf import OmegaConf",
      "context": "Imports OmegaConf from omegaconf",
      "category": "from_import",
      "code_snippet": "                try:\n                    from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore\n",
      "metadata": {
        "module": "omegaconf",
        "name": "OmegaConf",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 246,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import CheckpointingConfig",
      "context": "Imports CheckpointingConfig from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            # Build metadata using schema\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "CheckpointingConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 246,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import CheckpointMetadataV1",
      "context": "Imports CheckpointMetadataV1 from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            # Build metadata using schema\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "CheckpointMetadataV1",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 246,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import DecoderInfo",
      "context": "Imports DecoderInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            # Build metadata using schema\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "DecoderInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 246,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import EncoderInfo",
      "context": "Imports EncoderInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            # Build metadata using schema\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "EncoderInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 246,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import HeadInfo",
      "context": "Imports HeadInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            # Build metadata using schema\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "HeadInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 246,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import LossInfo",
      "context": "Imports LossInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            # Build metadata using schema\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "LossInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 246,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import MetricsInfo",
      "context": "Imports MetricsInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            # Build metadata using schema\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "MetricsInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 246,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import ModelInfo",
      "context": "Imports ModelInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            # Build metadata using schema\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "ModelInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 246,
      "column": 12,
      "pattern": "from ocr.core.utils.checkpoints.types import TrainingInfo",
      "context": "Imports TrainingInfo from ocr.core.utils.checkpoints.types",
      "category": "from_import",
      "code_snippet": "            # Build metadata using schema\n            from ocr.core.utils.checkpoints.types import (\n                CheckpointingConfig,",
      "metadata": {
        "module": "ocr.core.utils.checkpoints.types",
        "name": "TrainingInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/unique_checkpoint.py",
      "line": 422,
      "column": 8,
      "pattern": "from ocr.core.utils.wandb_utils import _get_wandb",
      "context": "Imports _get_wandb from ocr.core.utils.wandb_utils",
      "category": "from_import",
      "code_snippet": "        # Log checkpoint directory to wandb\n        from ocr.core.utils.wandb_utils import _get_wandb\n",
      "metadata": {
        "module": "ocr.core.utils.wandb_utils",
        "name": "_get_wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_completion.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_completion.py",
      "line": 5,
      "column": 0,
      "pattern": "import lightning.pytorch",
      "context": "Imports module: lightning.pytorch",
      "category": "import",
      "code_snippet": "\nimport lightning.pytorch as pl\nfrom lightning.pytorch.callbacks import Callback",
      "metadata": {
        "module": "lightning.pytorch",
        "alias": "pl"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_completion.py",
      "line": 6,
      "column": 0,
      "pattern": "from lightning.pytorch.callbacks import Callback",
      "context": "Imports Callback from lightning.pytorch.callbacks",
      "category": "from_import",
      "code_snippet": "import lightning.pytorch as pl\nfrom lightning.pytorch.callbacks import Callback\n",
      "metadata": {
        "module": "lightning.pytorch.callbacks",
        "name": "Callback",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_completion.py",
      "line": 15,
      "column": 8,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:\n        import wandb\n",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_completion.py",
      "line": 48,
      "column": 8,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:\n        import wandb\n        current_run = getattr(wandb, \"run\", None)",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 7,
      "column": 0,
      "pattern": "import lightning.pytorch",
      "context": "Imports module: lightning.pytorch",
      "category": "import",
      "code_snippet": "\nimport lightning.pytorch as pl\nimport numpy as np",
      "metadata": {
        "module": "lightning.pytorch",
        "alias": "pl"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import lightning.pytorch as pl\nimport numpy as np\nfrom PIL import Image",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 9,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core.lightning.processors.image_processor import ImageProcessor",
      "context": "Imports ImageProcessor from ocr.core.lightning.processors.image_processor",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.lightning.processors.image_processor import ImageProcessor\nfrom ocr.core.utils.config_utils import is_config",
      "metadata": {
        "module": "ocr.core.lightning.processors.image_processor",
        "name": "ImageProcessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "from ocr.core.lightning.processors.image_processor import ImageProcessor\nfrom ocr.core.utils.config_utils import is_config\nfrom ocr.core.utils.geometry_utils import apply_padding_offset_to_polygons, compute_padding_offsets",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 13,
      "column": 0,
      "pattern": "from ocr.core.utils.geometry_utils import apply_padding_offset_to_polygons",
      "context": "Imports apply_padding_offset_to_polygons from ocr.core.utils.geometry_utils",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.config_utils import is_config\nfrom ocr.core.utils.geometry_utils import apply_padding_offset_to_polygons, compute_padding_offsets\nfrom ocr.core.utils.orientation import normalize_pil_image, remap_polygons",
      "metadata": {
        "module": "ocr.core.utils.geometry_utils",
        "name": "apply_padding_offset_to_polygons",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 13,
      "column": 0,
      "pattern": "from ocr.core.utils.geometry_utils import compute_padding_offsets",
      "context": "Imports compute_padding_offsets from ocr.core.utils.geometry_utils",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.config_utils import is_config\nfrom ocr.core.utils.geometry_utils import apply_padding_offset_to_polygons, compute_padding_offsets\nfrom ocr.core.utils.orientation import normalize_pil_image, remap_polygons",
      "metadata": {
        "module": "ocr.core.utils.geometry_utils",
        "name": "compute_padding_offsets",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 14,
      "column": 0,
      "pattern": "from ocr.core.utils.orientation import normalize_pil_image",
      "context": "Imports normalize_pil_image from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.geometry_utils import apply_padding_offset_to_polygons, compute_padding_offsets\nfrom ocr.core.utils.orientation import normalize_pil_image, remap_polygons\nfrom ocr.core.utils.polygon_utils import ensure_polygon_array",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "normalize_pil_image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 14,
      "column": 0,
      "pattern": "from ocr.core.utils.orientation import remap_polygons",
      "context": "Imports remap_polygons from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.geometry_utils import apply_padding_offset_to_polygons, compute_padding_offsets\nfrom ocr.core.utils.orientation import normalize_pil_image, remap_polygons\nfrom ocr.core.utils.polygon_utils import ensure_polygon_array",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "remap_polygons",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 15,
      "column": 0,
      "pattern": "from ocr.core.utils.polygon_utils import ensure_polygon_array",
      "context": "Imports ensure_polygon_array from ocr.core.utils.polygon_utils",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.orientation import normalize_pil_image, remap_polygons\nfrom ocr.core.utils.polygon_utils import ensure_polygon_array\nfrom ocr.core.utils.wandb_utils import log_validation_images",
      "metadata": {
        "module": "ocr.core.utils.polygon_utils",
        "name": "ensure_polygon_array",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core.utils.wandb_utils import log_validation_images",
      "context": "Imports log_validation_images from ocr.core.utils.wandb_utils",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.polygon_utils import ensure_polygon_array\nfrom ocr.core.utils.wandb_utils import log_validation_images\n",
      "metadata": {
        "module": "ocr.core.utils.wandb_utils",
        "name": "log_validation_images",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/callbacks/wandb_image_logging.py",
      "line": 235,
      "column": 8,
      "pattern": "from ocr.core.utils.polygon_utils import has_duplicate_consecutive_points",
      "context": "Imports has_duplicate_consecutive_points from ocr.core.utils.polygon_utils",
      "category": "from_import",
      "code_snippet": "        \"\"\"Filter degenerate polygons using shared validators from polygon_utils.\"\"\"\n        from ocr.core.utils.polygon_utils import has_duplicate_consecutive_points\n",
      "metadata": {
        "module": "ocr.core.utils.polygon_utils",
        "name": "has_duplicate_consecutive_points",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from ocr.core.utils.logging import get_rich_console",
      "context": "Imports get_rich_console from ocr.core.utils.logging",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.logging import get_rich_console\n",
      "metadata": {
        "module": "ocr.core.utils.logging",
        "name": "get_rich_console",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from wandb_loggers import WandbProblemLogger",
      "context": "Imports WandbProblemLogger from wandb_loggers",
      "category": "from_import",
      "code_snippet": "\nfrom .wandb_loggers import WandbProblemLogger\n",
      "metadata": {
        "module": "wandb_loggers",
        "name": "WandbProblemLogger",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/progress_logger.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/wandb_loggers.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/wandb_loggers.py",
      "line": 9,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom PIL import Image",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/wandb_loggers.py",
      "line": 10,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/wandb_loggers.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.lightning.processors import ImageProcessor",
      "context": "Imports ImageProcessor from ocr.core.lightning.processors",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.lightning.processors import ImageProcessor\nfrom ocr.core.metrics import CLEvalMetric",
      "metadata": {
        "module": "ocr.core.lightning.processors",
        "name": "ImageProcessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/wandb_loggers.py",
      "line": 13,
      "column": 0,
      "pattern": "from ocr.core.metrics import CLEvalMetric",
      "context": "Imports CLEvalMetric from ocr.core.metrics",
      "category": "from_import",
      "code_snippet": "from ocr.core.lightning.processors import ImageProcessor\nfrom ocr.core.metrics import CLEvalMetric\nfrom ocr.core.utils.orientation import remap_polygons",
      "metadata": {
        "module": "ocr.core.metrics",
        "name": "CLEvalMetric",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/wandb_loggers.py",
      "line": 14,
      "column": 0,
      "pattern": "from ocr.core.utils.orientation import remap_polygons",
      "context": "Imports remap_polygons from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "from ocr.core.metrics import CLEvalMetric\nfrom ocr.core.utils.orientation import remap_polygons\n",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "remap_polygons",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/wandb_loggers.py",
      "line": 17,
      "column": 4,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "if TYPE_CHECKING:\n    import wandb\nelse:",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/loggers/wandb_loggers.py",
      "line": 20,
      "column": 8,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "    try:\n        import wandb\n    except ImportError:",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 4,
      "column": 0,
      "pattern": "import lightning.pytorch",
      "context": "Imports module: lightning.pytorch",
      "category": "import",
      "code_snippet": "\nimport lightning.pytorch as pl\nimport numpy as np",
      "metadata": {
        "module": "lightning.pytorch",
        "alias": "pl"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 5,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import lightning.pytorch as pl\nimport numpy as np\nimport torch",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import numpy as np\nimport torch\nfrom hydra.utils import instantiate",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 7,
      "column": 0,
      "pattern": "from hydra.utils import instantiate",
      "context": "Imports instantiate from hydra.utils",
      "category": "from_import",
      "code_snippet": "import torch\nfrom hydra.utils import instantiate\nfrom omegaconf import DictConfig",
      "metadata": {
        "module": "hydra.utils",
        "name": "instantiate",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 8,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "from hydra.utils import instantiate\nfrom omegaconf import DictConfig\nfrom pydantic import ValidationError",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 9,
      "column": 0,
      "pattern": "from pydantic import ValidationError",
      "context": "Imports ValidationError from pydantic",
      "category": "from_import",
      "code_snippet": "from omegaconf import DictConfig\nfrom pydantic import ValidationError\nfrom torch.utils.data import DataLoader",
      "metadata": {
        "module": "pydantic",
        "name": "ValidationError",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 10,
      "column": 0,
      "pattern": "from torch.utils.data import DataLoader",
      "context": "Imports DataLoader from torch.utils.data",
      "category": "from_import",
      "code_snippet": "from pydantic import ValidationError\nfrom torch.utils.data import DataLoader\n",
      "metadata": {
        "module": "torch.utils.data",
        "name": "DataLoader",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import CollateOutput",
      "context": "Imports CollateOutput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import CollateOutput, ValidatedTensorData\nfrom ocr.core.evaluation import CLEvalEvaluator",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "CollateOutput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import ValidatedTensorData",
      "context": "Imports ValidatedTensorData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import CollateOutput, ValidatedTensorData\nfrom ocr.core.evaluation import CLEvalEvaluator",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ValidatedTensorData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 13,
      "column": 0,
      "pattern": "from ocr.core.evaluation import CLEvalEvaluator",
      "context": "Imports CLEvalEvaluator from ocr.core.evaluation",
      "category": "from_import",
      "code_snippet": "from ocr.core.validation import CollateOutput, ValidatedTensorData\nfrom ocr.core.evaluation import CLEvalEvaluator\nfrom ocr.core.lightning.loggers import WandbProblemLogger",
      "metadata": {
        "module": "ocr.core.evaluation",
        "name": "CLEvalEvaluator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 14,
      "column": 0,
      "pattern": "from ocr.core.lightning.loggers import WandbProblemLogger",
      "context": "Imports WandbProblemLogger from ocr.core.lightning.loggers",
      "category": "from_import",
      "code_snippet": "from ocr.core.evaluation import CLEvalEvaluator\nfrom ocr.core.lightning.loggers import WandbProblemLogger\nfrom torchmetrics.text import CharErrorRate",
      "metadata": {
        "module": "ocr.core.lightning.loggers",
        "name": "WandbProblemLogger",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 15,
      "column": 0,
      "pattern": "from torchmetrics.text import CharErrorRate",
      "context": "Imports CharErrorRate from torchmetrics.text",
      "category": "from_import",
      "code_snippet": "from ocr.core.lightning.loggers import WandbProblemLogger\nfrom torchmetrics.text import CharErrorRate\nfrom ocr.core.lightning.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats, format_predictions",
      "metadata": {
        "module": "torchmetrics.text",
        "name": "CharErrorRate",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core.lightning.utils import CheckpointHandler",
      "context": "Imports CheckpointHandler from ocr.core.lightning.utils",
      "category": "from_import",
      "code_snippet": "from torchmetrics.text import CharErrorRate\nfrom ocr.core.lightning.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats, format_predictions\nfrom ocr.core.metrics import CLEvalMetric",
      "metadata": {
        "module": "ocr.core.lightning.utils",
        "name": "CheckpointHandler",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core.lightning.utils import extract_metric_kwargs",
      "context": "Imports extract_metric_kwargs from ocr.core.lightning.utils",
      "category": "from_import",
      "code_snippet": "from torchmetrics.text import CharErrorRate\nfrom ocr.core.lightning.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats, format_predictions\nfrom ocr.core.metrics import CLEvalMetric",
      "metadata": {
        "module": "ocr.core.lightning.utils",
        "name": "extract_metric_kwargs",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core.lightning.utils import extract_normalize_stats",
      "context": "Imports extract_normalize_stats from ocr.core.lightning.utils",
      "category": "from_import",
      "code_snippet": "from torchmetrics.text import CharErrorRate\nfrom ocr.core.lightning.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats, format_predictions\nfrom ocr.core.metrics import CLEvalMetric",
      "metadata": {
        "module": "ocr.core.lightning.utils",
        "name": "extract_normalize_stats",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core.lightning.utils import format_predictions",
      "context": "Imports format_predictions from ocr.core.lightning.utils",
      "category": "from_import",
      "code_snippet": "from torchmetrics.text import CharErrorRate\nfrom ocr.core.lightning.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats, format_predictions\nfrom ocr.core.metrics import CLEvalMetric",
      "metadata": {
        "module": "ocr.core.lightning.utils",
        "name": "format_predictions",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 17,
      "column": 0,
      "pattern": "from ocr.core.metrics import CLEvalMetric",
      "context": "Imports CLEvalMetric from ocr.core.metrics",
      "category": "from_import",
      "code_snippet": "from ocr.core.lightning.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats, format_predictions\nfrom ocr.core.metrics import CLEvalMetric\nfrom ocr.core.utils.submission import SubmissionWriter",
      "metadata": {
        "module": "ocr.core.metrics",
        "name": "CLEvalMetric",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 18,
      "column": 0,
      "pattern": "from ocr.core.utils.submission import SubmissionWriter",
      "context": "Imports SubmissionWriter from ocr.core.utils.submission",
      "category": "from_import",
      "code_snippet": "from ocr.core.metrics import CLEvalMetric\nfrom ocr.core.utils.submission import SubmissionWriter\n",
      "metadata": {
        "module": "ocr.core.utils.submission",
        "name": "SubmissionWriter",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 46,
      "column": 12,
      "pattern": "import torch._dynamo",
      "context": "Imports module: torch._dynamo",
      "category": "import",
      "code_snippet": "            # Configure torch.compile to handle scalar outputs better\n            import torch._dynamo\n",
      "metadata": {
        "module": "torch._dynamo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 278,
      "column": 24,
      "pattern": "from ocr.core.utils.wandb_utils import log_recognition_images",
      "context": "Imports log_recognition_images from ocr.core.utils.wandb_utils",
      "category": "from_import",
      "code_snippet": "\n                        from ocr.core.utils.wandb_utils import log_recognition_images\n",
      "metadata": {
        "module": "ocr.core.utils.wandb_utils",
        "name": "log_recognition_images",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 295,
      "column": 8,
      "pattern": "import ocr.data.datasets.db_collate_fn",
      "context": "Imports module: ocr.data.datasets.db_collate_fn",
      "category": "import",
      "code_snippet": "        # Reset collate function logging flag at the start of each training epoch\n        import ocr.data.datasets.db_collate_fn\n",
      "metadata": {
        "module": "ocr.data.datasets.db_collate_fn",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py",
      "line": 305,
      "column": 8,
      "pattern": "import ocr.data.datasets.db_collate_fn",
      "context": "Imports module: ocr.data.datasets.db_collate_fn",
      "category": "import",
      "code_snippet": "        # Reset collate function logging flag at the start of each validation epoch\n        import ocr.data.datasets.db_collate_fn\n",
      "metadata": {
        "module": "ocr.data.datasets.db_collate_fn",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/processors/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from image_processor import ImageProcessor",
      "context": "Imports ImageProcessor from image_processor",
      "category": "from_import",
      "code_snippet": "from .image_processor import ImageProcessor\n",
      "metadata": {
        "module": "image_processor",
        "name": "ImageProcessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/processors/image_processor.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/processors/image_processor.py",
      "line": 7,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nimport torch",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/processors/image_processor.py",
      "line": 8,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import numpy as np\nimport torch\nfrom PIL import Image",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/processors/image_processor.py",
      "line": 9,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import torch\nfrom PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from checkpoint_utils import CheckpointHandler",
      "context": "Imports CheckpointHandler from checkpoint_utils",
      "category": "from_import",
      "code_snippet": "from .checkpoint_utils import CheckpointHandler\nfrom .config_utils import extract_metric_kwargs, extract_normalize_stats",
      "metadata": {
        "module": "checkpoint_utils",
        "name": "CheckpointHandler",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/__init__.py",
      "line": 2,
      "column": 0,
      "pattern": "from config_utils import extract_metric_kwargs",
      "context": "Imports extract_metric_kwargs from config_utils",
      "category": "from_import",
      "code_snippet": "from .checkpoint_utils import CheckpointHandler\nfrom .config_utils import extract_metric_kwargs, extract_normalize_stats\nfrom .prediction_utils import format_predictions",
      "metadata": {
        "module": "config_utils",
        "name": "extract_metric_kwargs",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/__init__.py",
      "line": 2,
      "column": 0,
      "pattern": "from config_utils import extract_normalize_stats",
      "context": "Imports extract_normalize_stats from config_utils",
      "category": "from_import",
      "code_snippet": "from .checkpoint_utils import CheckpointHandler\nfrom .config_utils import extract_metric_kwargs, extract_normalize_stats\nfrom .prediction_utils import format_predictions",
      "metadata": {
        "module": "config_utils",
        "name": "extract_normalize_stats",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from prediction_utils import format_predictions",
      "context": "Imports format_predictions from prediction_utils",
      "category": "from_import",
      "code_snippet": "from .config_utils import extract_metric_kwargs, extract_normalize_stats\nfrom .prediction_utils import format_predictions\n",
      "metadata": {
        "module": "prediction_utils",
        "name": "format_predictions",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/checkpoint_utils.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/config_utils.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/config_utils.py",
      "line": 5,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom omegaconf import DictConfig, ListConfig",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/config_utils.py",
      "line": 6,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom omegaconf import DictConfig, ListConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/config_utils.py",
      "line": 6,
      "column": 0,
      "pattern": "from omegaconf import ListConfig",
      "context": "Imports ListConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom omegaconf import DictConfig, ListConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "ListConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/config_utils.py",
      "line": 8,
      "column": 0,
      "pattern": "from ocr.core.validation import MetricConfig",
      "context": "Imports MetricConfig from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import MetricConfig\nfrom ocr.core.utils.config_utils import ensure_dict",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "MetricConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/config_utils.py",
      "line": 9,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import ensure_dict",
      "context": "Imports ensure_dict from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "from ocr.core.validation import MetricConfig\nfrom ocr.core.utils.config_utils import ensure_dict\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "ensure_dict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/model_utils.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/model_utils.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/prediction_utils.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/prediction_utils.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/utils/prediction_utils.py",
      "line": 10,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from cleval_metric import CLEvalMetric",
      "context": "Imports CLEvalMetric from cleval_metric",
      "category": "from_import",
      "code_snippet": "from .cleval_metric import CLEvalMetric  # noqa: F401",
      "metadata": {
        "module": "cleval_metric",
        "name": "CLEvalMetric",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/box_types.py",
      "line": 20,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nimport Polygon as polygon3",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/box_types.py",
      "line": 21,
      "column": 0,
      "pattern": "import Polygon",
      "context": "Imports module: Polygon",
      "category": "import",
      "code_snippet": "import numpy as np\nimport Polygon as polygon3\nfrom scipy.spatial import ConvexHull",
      "metadata": {
        "module": "Polygon",
        "alias": "polygon3"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/box_types.py",
      "line": 22,
      "column": 0,
      "pattern": "from scipy.spatial import ConvexHull",
      "context": "Imports ConvexHull from scipy.spatial",
      "category": "from_import",
      "code_snippet": "import Polygon as polygon3\nfrom scipy.spatial import ConvexHull\n",
      "metadata": {
        "module": "scipy.spatial",
        "name": "ConvexHull",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/cleval_metric.py",
      "line": 12,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nfrom torchmetrics import Metric",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/cleval_metric.py",
      "line": 13,
      "column": 0,
      "pattern": "from torchmetrics import Metric",
      "context": "Imports Metric from torchmetrics",
      "category": "from_import",
      "code_snippet": "import torch\nfrom torchmetrics import Metric\n",
      "metadata": {
        "module": "torchmetrics",
        "name": "Metric",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/cleval_metric.py",
      "line": 15,
      "column": 0,
      "pattern": "from utils.logging import logger",
      "context": "Imports logger from utils.logging",
      "category": "from_import",
      "code_snippet": "\nfrom ..utils.logging import logger\nfrom .box_types import POLY",
      "metadata": {
        "module": "utils.logging",
        "name": "logger",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/cleval_metric.py",
      "line": 16,
      "column": 0,
      "pattern": "from box_types import POLY",
      "context": "Imports POLY from box_types",
      "category": "from_import",
      "code_snippet": "from ..utils.logging import logger\nfrom .box_types import POLY\nfrom .data import SampleResult",
      "metadata": {
        "module": "box_types",
        "name": "POLY",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/cleval_metric.py",
      "line": 17,
      "column": 0,
      "pattern": "from data import SampleResult",
      "context": "Imports SampleResult from data",
      "category": "from_import",
      "code_snippet": "from .box_types import POLY\nfrom .data import SampleResult\nfrom .eval_functions import evaluation",
      "metadata": {
        "module": "data",
        "name": "SampleResult",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/cleval_metric.py",
      "line": 18,
      "column": 0,
      "pattern": "from eval_functions import evaluation",
      "context": "Imports evaluation from eval_functions",
      "category": "from_import",
      "code_snippet": "from .data import SampleResult\nfrom .eval_functions import evaluation\n",
      "metadata": {
        "module": "eval_functions",
        "name": "evaluation",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/data.py",
      "line": 14,
      "column": 0,
      "pattern": "from utils import harmonic_mean",
      "context": "Imports harmonic_mean from utils",
      "category": "from_import",
      "code_snippet": "\nfrom .utils import harmonic_mean\n",
      "metadata": {
        "module": "utils",
        "name": "harmonic_mean",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 14,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom numba import njit",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 15,
      "column": 0,
      "pattern": "from numba import njit",
      "context": "Imports njit from numba",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom numba import njit\nfrom numpy.typing import NDArray",
      "metadata": {
        "module": "numba",
        "name": "njit",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 16,
      "column": 0,
      "pattern": "from numpy.typing import NDArray",
      "context": "Imports NDArray from numpy.typing",
      "category": "from_import",
      "code_snippet": "from numba import njit\nfrom numpy.typing import NDArray\n",
      "metadata": {
        "module": "numpy.typing",
        "name": "NDArray",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 18,
      "column": 0,
      "pattern": "from data import DetBoxResult",
      "context": "Imports DetBoxResult from data",
      "category": "from_import",
      "code_snippet": "\nfrom .data import DetBoxResult, GTBoxResult, MatchReleation, MatchResult, Point, SampleResult\nfrom .utils import harmonic_mean, lcs",
      "metadata": {
        "module": "data",
        "name": "DetBoxResult",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 18,
      "column": 0,
      "pattern": "from data import GTBoxResult",
      "context": "Imports GTBoxResult from data",
      "category": "from_import",
      "code_snippet": "\nfrom .data import DetBoxResult, GTBoxResult, MatchReleation, MatchResult, Point, SampleResult\nfrom .utils import harmonic_mean, lcs",
      "metadata": {
        "module": "data",
        "name": "GTBoxResult",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 18,
      "column": 0,
      "pattern": "from data import MatchReleation",
      "context": "Imports MatchReleation from data",
      "category": "from_import",
      "code_snippet": "\nfrom .data import DetBoxResult, GTBoxResult, MatchReleation, MatchResult, Point, SampleResult\nfrom .utils import harmonic_mean, lcs",
      "metadata": {
        "module": "data",
        "name": "MatchReleation",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 18,
      "column": 0,
      "pattern": "from data import MatchResult",
      "context": "Imports MatchResult from data",
      "category": "from_import",
      "code_snippet": "\nfrom .data import DetBoxResult, GTBoxResult, MatchReleation, MatchResult, Point, SampleResult\nfrom .utils import harmonic_mean, lcs",
      "metadata": {
        "module": "data",
        "name": "MatchResult",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 18,
      "column": 0,
      "pattern": "from data import Point",
      "context": "Imports Point from data",
      "category": "from_import",
      "code_snippet": "\nfrom .data import DetBoxResult, GTBoxResult, MatchReleation, MatchResult, Point, SampleResult\nfrom .utils import harmonic_mean, lcs",
      "metadata": {
        "module": "data",
        "name": "Point",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 18,
      "column": 0,
      "pattern": "from data import SampleResult",
      "context": "Imports SampleResult from data",
      "category": "from_import",
      "code_snippet": "\nfrom .data import DetBoxResult, GTBoxResult, MatchReleation, MatchResult, Point, SampleResult\nfrom .utils import harmonic_mean, lcs",
      "metadata": {
        "module": "data",
        "name": "SampleResult",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 19,
      "column": 0,
      "pattern": "from utils import harmonic_mean",
      "context": "Imports harmonic_mean from utils",
      "category": "from_import",
      "code_snippet": "from .data import DetBoxResult, GTBoxResult, MatchReleation, MatchResult, Point, SampleResult\nfrom .utils import harmonic_mean, lcs\n",
      "metadata": {
        "module": "utils",
        "name": "harmonic_mean",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/eval_functions.py",
      "line": 19,
      "column": 0,
      "pattern": "from utils import lcs",
      "context": "Imports lcs from utils",
      "category": "from_import",
      "code_snippet": "from .data import DetBoxResult, GTBoxResult, MatchReleation, MatchResult, Point, SampleResult\nfrom .utils import harmonic_mean, lcs\n",
      "metadata": {
        "module": "utils",
        "name": "lcs",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/utils.py",
      "line": 12,
      "column": 0,
      "pattern": "import codecs",
      "context": "Imports module: codecs",
      "category": "import",
      "code_snippet": "\nimport codecs\nimport json",
      "metadata": {
        "module": "codecs",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/metrics/utils.py",
      "line": 18,
      "column": 0,
      "pattern": "from numba import njit",
      "context": "Imports njit from numba",
      "category": "from_import",
      "code_snippet": "\nfrom numba import njit\n",
      "metadata": {
        "module": "numba",
        "name": "njit",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from  import architectures",
      "context": "Imports architectures from ",
      "category": "from_import",
      "code_snippet": "from . import architectures as _architectures  # noqa: F401\n",
      "metadata": {
        "module": "",
        "name": "architectures",
        "alias": "_architectures"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py",
      "line": 5,
      "column": 4,
      "pattern": "from ocr.core.models.architecture import OCRModel",
      "context": "Imports OCRModel from ocr.core.models.architecture",
      "category": "from_import",
      "code_snippet": "def get_model_by_cfg(config):\n    from ocr.core.models.architecture import OCRModel\n",
      "metadata": {
        "module": "ocr.core.models.architecture",
        "name": "OCRModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py",
      "line": 9,
      "column": 8,
      "pattern": "from ocr.features.recognition.models import PARSeq",
      "context": "Imports PARSeq from ocr.features.recognition.models",
      "category": "from_import",
      "code_snippet": "    if arch_name == \"parseq\":\n        from ocr.features.recognition.models import PARSeq\n        return PARSeq(config)",
      "metadata": {
        "module": "ocr.features.recognition.models",
        "name": "PARSeq",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 4,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 5,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\nfrom omegaconf import OmegaConf, DictConfig",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 6,
      "column": 0,
      "pattern": "from omegaconf import OmegaConf",
      "context": "Imports OmegaConf from omegaconf",
      "category": "from_import",
      "code_snippet": "import torch.nn as nn\nfrom omegaconf import OmegaConf, DictConfig\nfrom hydra.utils import instantiate",
      "metadata": {
        "module": "omegaconf",
        "name": "OmegaConf",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 6,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "import torch.nn as nn\nfrom omegaconf import OmegaConf, DictConfig\nfrom hydra.utils import instantiate",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 7,
      "column": 0,
      "pattern": "from hydra.utils import instantiate",
      "context": "Imports instantiate from hydra.utils",
      "category": "from_import",
      "code_snippet": "from omegaconf import OmegaConf, DictConfig\nfrom hydra.utils import instantiate\n",
      "metadata": {
        "module": "hydra.utils",
        "name": "instantiate",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 9,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import ensure_dict",
      "context": "Imports ensure_dict from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import ensure_dict, is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "ensure_dict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 9,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import ensure_dict, is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core import get_registry",
      "context": "Imports get_registry from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import get_registry\n",
      "metadata": {
        "module": "ocr.core",
        "name": "get_registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 187,
      "column": 8,
      "pattern": "from ocr.core.models.decoder import get_decoder_by_cfg",
      "context": "Imports get_decoder_by_cfg from ocr.core.models.decoder",
      "category": "from_import",
      "code_snippet": "    def _init_from_components(self, cfg):\n        from ocr.core.models.decoder import get_decoder_by_cfg\n        from ocr.core.models.encoder import get_encoder_by_cfg",
      "metadata": {
        "module": "ocr.core.models.decoder",
        "name": "get_decoder_by_cfg",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 188,
      "column": 8,
      "pattern": "from ocr.core.models.encoder import get_encoder_by_cfg",
      "context": "Imports get_encoder_by_cfg from ocr.core.models.encoder",
      "category": "from_import",
      "code_snippet": "        from ocr.core.models.decoder import get_decoder_by_cfg\n        from ocr.core.models.encoder import get_encoder_by_cfg\n        from ocr.core.models.head import get_head_by_cfg",
      "metadata": {
        "module": "ocr.core.models.encoder",
        "name": "get_encoder_by_cfg",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 189,
      "column": 8,
      "pattern": "from ocr.core.models.head import get_head_by_cfg",
      "context": "Imports get_head_by_cfg from ocr.core.models.head",
      "category": "from_import",
      "code_snippet": "        from ocr.core.models.encoder import get_encoder_by_cfg\n        from ocr.core.models.head import get_head_by_cfg\n        from ocr.core.models.loss import get_loss_by_cfg",
      "metadata": {
        "module": "ocr.core.models.head",
        "name": "get_head_by_cfg",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py",
      "line": 190,
      "column": 8,
      "pattern": "from ocr.core.models.loss import get_loss_by_cfg",
      "context": "Imports get_loss_by_cfg from ocr.core.models.loss",
      "category": "from_import",
      "code_snippet": "        from ocr.core.models.head import get_head_by_cfg\n        from ocr.core.models.loss import get_loss_by_cfg\n",
      "metadata": {
        "module": "ocr.core.models.loss",
        "name": "get_loss_by_cfg",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architectures/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from ocr.features.detection.models.architectures import craft",
      "context": "Imports craft from ocr.features.detection.models.architectures",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.detection.models.architectures import craft, dbnet, dbnetpp  # noqa: F401\nfrom ocr.features.recognition.models import architecture as recognition_arch  # noqa: F401",
      "metadata": {
        "module": "ocr.features.detection.models.architectures",
        "name": "craft",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architectures/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from ocr.features.detection.models.architectures import dbnet",
      "context": "Imports dbnet from ocr.features.detection.models.architectures",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.detection.models.architectures import craft, dbnet, dbnetpp  # noqa: F401\nfrom ocr.features.recognition.models import architecture as recognition_arch  # noqa: F401",
      "metadata": {
        "module": "ocr.features.detection.models.architectures",
        "name": "dbnet",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architectures/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from ocr.features.detection.models.architectures import dbnetpp",
      "context": "Imports dbnetpp from ocr.features.detection.models.architectures",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.detection.models.architectures import craft, dbnet, dbnetpp  # noqa: F401\nfrom ocr.features.recognition.models import architecture as recognition_arch  # noqa: F401",
      "metadata": {
        "module": "ocr.features.detection.models.architectures",
        "name": "dbnetpp",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architectures/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from ocr.features.recognition.models import architecture",
      "context": "Imports architecture from ocr.features.recognition.models",
      "category": "from_import",
      "code_snippet": "from ocr.features.detection.models.architectures import craft, dbnet, dbnetpp  # noqa: F401\nfrom ocr.features.recognition.models import architecture as recognition_arch  # noqa: F401\nfrom . import shared_decoders  # noqa: F401",
      "metadata": {
        "module": "ocr.features.recognition.models",
        "name": "architecture",
        "alias": "recognition_arch"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architectures/__init__.py",
      "line": 5,
      "column": 0,
      "pattern": "from  import shared_decoders",
      "context": "Imports shared_decoders from ",
      "category": "from_import",
      "code_snippet": "from ocr.features.recognition.models import architecture as recognition_arch  # noqa: F401\nfrom . import shared_decoders  # noqa: F401\n",
      "metadata": {
        "module": "",
        "name": "shared_decoders",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architectures/shared_decoders.py",
      "line": 3,
      "column": 0,
      "pattern": "from ocr.core import registry",
      "context": "Imports registry from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import registry\nfrom ocr.features.detection.models.decoders.fpn_decoder import FPNDecoder",
      "metadata": {
        "module": "ocr.core",
        "name": "registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architectures/shared_decoders.py",
      "line": 4,
      "column": 0,
      "pattern": "from ocr.features.detection.models.decoders.fpn_decoder import FPNDecoder",
      "context": "Imports FPNDecoder from ocr.features.detection.models.decoders.fpn_decoder",
      "category": "from_import",
      "code_snippet": "from ocr.core import registry\nfrom ocr.features.detection.models.decoders.fpn_decoder import FPNDecoder\nfrom ocr.core.models.decoder.pan_decoder import PANDecoder",
      "metadata": {
        "module": "ocr.features.detection.models.decoders.fpn_decoder",
        "name": "FPNDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architectures/shared_decoders.py",
      "line": 5,
      "column": 0,
      "pattern": "from ocr.core.models.decoder.pan_decoder import PANDecoder",
      "context": "Imports PANDecoder from ocr.core.models.decoder.pan_decoder",
      "category": "from_import",
      "code_snippet": "from ocr.features.detection.models.decoders.fpn_decoder import FPNDecoder\nfrom ocr.core.models.decoder.pan_decoder import PANDecoder\n",
      "metadata": {
        "module": "ocr.core.models.decoder.pan_decoder",
        "name": "PANDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from hydra.utils import instantiate",
      "context": "Imports instantiate from hydra.utils",
      "category": "from_import",
      "code_snippet": "from hydra.utils import instantiate\n",
      "metadata": {
        "module": "hydra.utils",
        "name": "instantiate",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from pan_decoder import PANDecoder",
      "context": "Imports PANDecoder from pan_decoder",
      "category": "from_import",
      "code_snippet": "\nfrom .pan_decoder import PANDecoder  # noqa: F401\nfrom .unet import UNetDecoder  # noqa: F401",
      "metadata": {
        "module": "pan_decoder",
        "name": "PANDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from unet import UNetDecoder",
      "context": "Imports UNetDecoder from unet",
      "category": "from_import",
      "code_snippet": "from .pan_decoder import PANDecoder  # noqa: F401\nfrom .unet import UNetDecoder  # noqa: F401\n",
      "metadata": {
        "module": "unet",
        "name": "UNetDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/pan_decoder.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/pan_decoder.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/pan_decoder.py",
      "line": 8,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/pan_decoder.py",
      "line": 9,
      "column": 0,
      "pattern": "import torch.nn.functional",
      "context": "Imports module: torch.nn.functional",
      "category": "import",
      "code_snippet": "import torch.nn as nn\nimport torch.nn.functional as F\n",
      "metadata": {
        "module": "torch.nn.functional",
        "alias": "F"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/pan_decoder.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core import BaseDecoder",
      "context": "Imports BaseDecoder from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseDecoder\n",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/unet.py",
      "line": 5,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/unet.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\n",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/unet.py",
      "line": 8,
      "column": 0,
      "pattern": "from ocr.core import BaseDecoder",
      "context": "Imports BaseDecoder from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseDecoder\n",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/encoder/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from hydra.utils import instantiate",
      "context": "Imports instantiate from hydra.utils",
      "category": "from_import",
      "code_snippet": "from hydra.utils import instantiate\n",
      "metadata": {
        "module": "hydra.utils",
        "name": "instantiate",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/encoder/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from timm_backbone import TimmBackbone",
      "context": "Imports TimmBackbone from timm_backbone",
      "category": "from_import",
      "code_snippet": "\nfrom .timm_backbone import TimmBackbone  # noqa: F401\n",
      "metadata": {
        "module": "timm_backbone",
        "name": "TimmBackbone",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/encoder/timm_backbone.py",
      "line": 5,
      "column": 0,
      "pattern": "import timm",
      "context": "Imports module: timm",
      "category": "import",
      "code_snippet": "\nimport timm\nimport torch",
      "metadata": {
        "module": "timm",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/encoder/timm_backbone.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import timm\nimport torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/encoder/timm_backbone.py",
      "line": 8,
      "column": 0,
      "pattern": "from ocr.core import BaseEncoder",
      "context": "Imports BaseEncoder from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseEncoder\n",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseEncoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/head/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from hydra.utils import instantiate",
      "context": "Imports instantiate from hydra.utils",
      "category": "from_import",
      "code_snippet": "from hydra.utils import instantiate\n",
      "metadata": {
        "module": "hydra.utils",
        "name": "instantiate",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/layers/common.py",
      "line": 3,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "\nimport torch.nn as nn\n",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/__init__.py",
      "line": 1,
      "column": 0,
      "pattern": "from hydra.utils import instantiate",
      "context": "Imports instantiate from hydra.utils",
      "category": "from_import",
      "code_snippet": "from hydra.utils import instantiate\n",
      "metadata": {
        "module": "hydra.utils",
        "name": "instantiate",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from bce_loss import BCELoss",
      "context": "Imports BCELoss from bce_loss",
      "category": "from_import",
      "code_snippet": "\nfrom .bce_loss import BCELoss  # noqa: F401\nfrom .craft_loss import CraftLoss  # noqa: F401",
      "metadata": {
        "module": "bce_loss",
        "name": "BCELoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from craft_loss import CraftLoss",
      "context": "Imports CraftLoss from craft_loss",
      "category": "from_import",
      "code_snippet": "from .bce_loss import BCELoss  # noqa: F401\nfrom .craft_loss import CraftLoss  # noqa: F401\nfrom .cross_entropy_loss import CrossEntropyLoss  # noqa: F401",
      "metadata": {
        "module": "craft_loss",
        "name": "CraftLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/__init__.py",
      "line": 5,
      "column": 0,
      "pattern": "from cross_entropy_loss import CrossEntropyLoss",
      "context": "Imports CrossEntropyLoss from cross_entropy_loss",
      "category": "from_import",
      "code_snippet": "from .craft_loss import CraftLoss  # noqa: F401\nfrom .cross_entropy_loss import CrossEntropyLoss  # noqa: F401\nfrom .db_loss import DBLoss  # noqa: F401",
      "metadata": {
        "module": "cross_entropy_loss",
        "name": "CrossEntropyLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from db_loss import DBLoss",
      "context": "Imports DBLoss from db_loss",
      "category": "from_import",
      "code_snippet": "from .cross_entropy_loss import CrossEntropyLoss  # noqa: F401\nfrom .db_loss import DBLoss  # noqa: F401\nfrom .dice_loss import DiceLoss  # noqa: F401",
      "metadata": {
        "module": "db_loss",
        "name": "DBLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from dice_loss import DiceLoss",
      "context": "Imports DiceLoss from dice_loss",
      "category": "from_import",
      "code_snippet": "from .db_loss import DBLoss  # noqa: F401\nfrom .dice_loss import DiceLoss  # noqa: F401\nfrom .l1_loss import MaskL1Loss  # noqa: F401",
      "metadata": {
        "module": "dice_loss",
        "name": "DiceLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/__init__.py",
      "line": 8,
      "column": 0,
      "pattern": "from l1_loss import MaskL1Loss",
      "context": "Imports MaskL1Loss from l1_loss",
      "category": "from_import",
      "code_snippet": "from .dice_loss import DiceLoss  # noqa: F401\nfrom .l1_loss import MaskL1Loss  # noqa: F401\n",
      "metadata": {
        "module": "l1_loss",
        "name": "MaskL1Loss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/bce_loss.py",
      "line": 14,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/bce_loss.py",
      "line": 15,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\nfrom pydantic import ValidationError",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/bce_loss.py",
      "line": 16,
      "column": 0,
      "pattern": "from pydantic import ValidationError",
      "context": "Imports ValidationError from pydantic",
      "category": "from_import",
      "code_snippet": "import torch.nn as nn\nfrom pydantic import ValidationError\n",
      "metadata": {
        "module": "pydantic",
        "name": "ValidationError",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/bce_loss.py",
      "line": 18,
      "column": 0,
      "pattern": "from ocr.core.validation import ValidatedTensorData",
      "context": "Imports ValidatedTensorData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import ValidatedTensorData\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ValidatedTensorData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/craft_loss.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/craft_loss.py",
      "line": 5,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn.functional as F",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/craft_loss.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch.nn.functional",
      "context": "Imports module: torch.nn.functional",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn.functional as F\n",
      "metadata": {
        "module": "torch.nn.functional",
        "alias": "F"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/craft_loss.py",
      "line": 8,
      "column": 0,
      "pattern": "from ocr.core import BaseLoss",
      "context": "Imports BaseLoss from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseLoss\n",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/cross_entropy_loss.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn.functional as F",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/cross_entropy_loss.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch.nn.functional",
      "context": "Imports module: torch.nn.functional",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn.functional as F\n",
      "metadata": {
        "module": "torch.nn.functional",
        "alias": "F"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/cross_entropy_loss.py",
      "line": 9,
      "column": 0,
      "pattern": "from ocr.core import BaseLoss",
      "context": "Imports BaseLoss from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseLoss\n",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/db_loss.py",
      "line": 14,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/db_loss.py",
      "line": 16,
      "column": 0,
      "pattern": "from ocr.core import BaseLoss",
      "context": "Imports BaseLoss from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseLoss\nfrom .bce_loss import BCELoss",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/db_loss.py",
      "line": 17,
      "column": 0,
      "pattern": "from bce_loss import BCELoss",
      "context": "Imports BCELoss from bce_loss",
      "category": "from_import",
      "code_snippet": "from ocr.core import BaseLoss\nfrom .bce_loss import BCELoss\nfrom .dice_loss import DiceLoss",
      "metadata": {
        "module": "bce_loss",
        "name": "BCELoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/db_loss.py",
      "line": 18,
      "column": 0,
      "pattern": "from dice_loss import DiceLoss",
      "context": "Imports DiceLoss from dice_loss",
      "category": "from_import",
      "code_snippet": "from .bce_loss import BCELoss\nfrom .dice_loss import DiceLoss\nfrom .l1_loss import MaskL1Loss",
      "metadata": {
        "module": "dice_loss",
        "name": "DiceLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/db_loss.py",
      "line": 19,
      "column": 0,
      "pattern": "from l1_loss import MaskL1Loss",
      "context": "Imports MaskL1Loss from l1_loss",
      "category": "from_import",
      "code_snippet": "from .dice_loss import DiceLoss\nfrom .l1_loss import MaskL1Loss\n",
      "metadata": {
        "module": "l1_loss",
        "name": "MaskL1Loss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/dice_loss.py",
      "line": 14,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/dice_loss.py",
      "line": 15,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\nfrom pydantic import ValidationError",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/dice_loss.py",
      "line": 16,
      "column": 0,
      "pattern": "from pydantic import ValidationError",
      "context": "Imports ValidationError from pydantic",
      "category": "from_import",
      "code_snippet": "import torch.nn as nn\nfrom pydantic import ValidationError\n",
      "metadata": {
        "module": "pydantic",
        "name": "ValidationError",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/dice_loss.py",
      "line": 18,
      "column": 0,
      "pattern": "from ocr.core.validation import ValidatedTensorData",
      "context": "Imports ValidatedTensorData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import ValidatedTensorData\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ValidatedTensorData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/l1_loss.py",
      "line": 14,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/loss/l1_loss.py",
      "line": 15,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\n",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/api_usage_tracker.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/background_normalization.py",
      "line": 8,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/background_normalization.py",
      "line": 10,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/cache_manager.py",
      "line": 53,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/cache_manager.py",
      "line": 59,
      "column": 0,
      "pattern": "from ocr.core.validation import CacheConfig",
      "context": "Imports CacheConfig from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import CacheConfig, DataItem, ImageData, MapData\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "CacheConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/cache_manager.py",
      "line": 59,
      "column": 0,
      "pattern": "from ocr.core.validation import DataItem",
      "context": "Imports DataItem from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import CacheConfig, DataItem, ImageData, MapData\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "DataItem",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/cache_manager.py",
      "line": 59,
      "column": 0,
      "pattern": "from ocr.core.validation import ImageData",
      "context": "Imports ImageData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import CacheConfig, DataItem, ImageData, MapData\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ImageData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/cache_manager.py",
      "line": 59,
      "column": 0,
      "pattern": "from ocr.core.validation import MapData",
      "context": "Imports MapData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import CacheConfig, DataItem, ImageData, MapData\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "MapData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/callbacks.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/callbacks.py",
      "line": 11,
      "column": 0,
      "pattern": "import hydra",
      "context": "Imports module: hydra",
      "category": "import",
      "code_snippet": "\nimport hydra\nfrom omegaconf import DictConfig, OmegaConf",
      "metadata": {
        "module": "hydra",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/callbacks.py",
      "line": 12,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "import hydra\nfrom omegaconf import DictConfig, OmegaConf\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/callbacks.py",
      "line": 12,
      "column": 0,
      "pattern": "from omegaconf import OmegaConf",
      "context": "Imports OmegaConf from omegaconf",
      "category": "from_import",
      "code_snippet": "import hydra\nfrom omegaconf import DictConfig, OmegaConf\n",
      "metadata": {
        "module": "omegaconf",
        "name": "OmegaConf",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/checkpoints/metadata_loader.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\nimport yaml",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/checkpoints/metadata_loader.py",
      "line": 2,
      "column": 0,
      "pattern": "import yaml",
      "context": "Imports module: yaml",
      "category": "import",
      "code_snippet": "from __future__ import annotations\nimport yaml\nfrom pathlib import Path",
      "metadata": {
        "module": "yaml",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/checkpoints/types.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/checkpoints/types.py",
      "line": 4,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "from typing import Any\nfrom pydantic import BaseModel\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/__init__.py",
      "line": 8,
      "column": 0,
      "pattern": "from builder import CommandBuilder",
      "context": "Imports CommandBuilder from builder",
      "category": "from_import",
      "code_snippet": "\nfrom .builder import CommandBuilder\nfrom .executor import CommandExecutor",
      "metadata": {
        "module": "builder",
        "name": "CommandBuilder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/__init__.py",
      "line": 9,
      "column": 0,
      "pattern": "from executor import CommandExecutor",
      "context": "Imports CommandExecutor from executor",
      "category": "from_import",
      "code_snippet": "from .builder import CommandBuilder\nfrom .executor import CommandExecutor\nfrom .validator import CommandValidator",
      "metadata": {
        "module": "executor",
        "name": "CommandExecutor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/__init__.py",
      "line": 10,
      "column": 0,
      "pattern": "from validator import CommandValidator",
      "context": "Imports CommandValidator from validator",
      "category": "from_import",
      "code_snippet": "from .executor import CommandExecutor\nfrom .validator import CommandValidator\n",
      "metadata": {
        "module": "validator",
        "name": "CommandValidator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/builder.py",
      "line": 9,
      "column": 0,
      "pattern": "from models import CommandParams",
      "context": "Imports CommandParams from models",
      "category": "from_import",
      "code_snippet": "\nfrom .models import CommandParams, PredictCommandParams, TestCommandParams, TrainCommandParams\nfrom .quoting import quote_override",
      "metadata": {
        "module": "models",
        "name": "CommandParams",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/builder.py",
      "line": 9,
      "column": 0,
      "pattern": "from models import PredictCommandParams",
      "context": "Imports PredictCommandParams from models",
      "category": "from_import",
      "code_snippet": "\nfrom .models import CommandParams, PredictCommandParams, TestCommandParams, TrainCommandParams\nfrom .quoting import quote_override",
      "metadata": {
        "module": "models",
        "name": "PredictCommandParams",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/builder.py",
      "line": 9,
      "column": 0,
      "pattern": "from models import TestCommandParams",
      "context": "Imports TestCommandParams from models",
      "category": "from_import",
      "code_snippet": "\nfrom .models import CommandParams, PredictCommandParams, TestCommandParams, TrainCommandParams\nfrom .quoting import quote_override",
      "metadata": {
        "module": "models",
        "name": "TestCommandParams",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/builder.py",
      "line": 9,
      "column": 0,
      "pattern": "from models import TrainCommandParams",
      "context": "Imports TrainCommandParams from models",
      "category": "from_import",
      "code_snippet": "\nfrom .models import CommandParams, PredictCommandParams, TestCommandParams, TrainCommandParams\nfrom .quoting import quote_override",
      "metadata": {
        "module": "models",
        "name": "TrainCommandParams",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/builder.py",
      "line": 10,
      "column": 0,
      "pattern": "from quoting import quote_override",
      "context": "Imports quote_override from quoting",
      "category": "from_import",
      "code_snippet": "from .models import CommandParams, PredictCommandParams, TestCommandParams, TrainCommandParams\nfrom .quoting import quote_override\nfrom ocr.core.utils.path_utils import PROJECT_ROOT",
      "metadata": {
        "module": "quoting",
        "name": "quote_override",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/builder.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core.utils.path_utils import PROJECT_ROOT",
      "context": "Imports PROJECT_ROOT from ocr.core.utils.path_utils",
      "category": "from_import",
      "code_snippet": "from .quoting import quote_override\nfrom ocr.core.utils.path_utils import PROJECT_ROOT\n",
      "metadata": {
        "module": "ocr.core.utils.path_utils",
        "name": "PROJECT_ROOT",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/executor.py",
      "line": 8,
      "column": 0,
      "pattern": "import shlex",
      "context": "Imports module: shlex",
      "category": "import",
      "code_snippet": "import os\nimport shlex\nimport signal",
      "metadata": {
        "module": "shlex",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/command/executor.py",
      "line": 9,
      "column": 0,
      "pattern": "import signal",
      "context": "Imports module: signal",
      "category": "import",
      "code_snippet": "import shlex\nimport signal\nimport subprocess",
      "metadata": {
        "module": "signal",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config.py",
      "line": 21,
      "column": 0,
      "pattern": "import yaml",
      "context": "Imports module: yaml",
      "category": "import",
      "code_snippet": "\nimport yaml\n",
      "metadata": {
        "module": "yaml",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config.py",
      "line": 23,
      "column": 0,
      "pattern": "from ocr.core.utils.path_utils import get_path_resolver",
      "context": "Imports get_path_resolver from ocr.core.utils.path_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.path_utils import get_path_resolver\n",
      "metadata": {
        "module": "ocr.core.utils.path_utils",
        "name": "get_path_resolver",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config.py",
      "line": 81,
      "column": 12,
      "pattern": "from ocr.core import registry",
      "context": "Imports registry from ocr.core",
      "category": "from_import",
      "code_snippet": "        try:\n            from ocr.core import registry\n",
      "metadata": {
        "module": "ocr.core",
        "name": "registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config.py",
      "line": 124,
      "column": 8,
      "pattern": "from ocr.core.models import architectures",
      "context": "Imports architectures from ocr.core.models",
      "category": "from_import",
      "code_snippet": "        # Ensure architectures are registered by importing the module\n        from ocr.core.models import architectures  # noqa: F401\n        from ocr.core import registry",
      "metadata": {
        "module": "ocr.core.models",
        "name": "architectures",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config.py",
      "line": 125,
      "column": 8,
      "pattern": "from ocr.core import registry",
      "context": "Imports registry from ocr.core",
      "category": "from_import",
      "code_snippet": "        from ocr.core.models import architectures  # noqa: F401\n        from ocr.core import registry\n",
      "metadata": {
        "module": "ocr.core",
        "name": "registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config_utils.py",
      "line": 6,
      "column": 0,
      "pattern": "import yaml",
      "context": "Imports module: yaml",
      "category": "import",
      "code_snippet": "\nimport yaml\nfrom omegaconf import DictConfig",
      "metadata": {
        "module": "yaml",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config_utils.py",
      "line": 7,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "import yaml\nfrom omegaconf import DictConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config_utils.py",
      "line": 9,
      "column": 0,
      "pattern": "from ocr.core.utils.path_utils import get_path_resolver",
      "context": "Imports get_path_resolver from ocr.core.utils.path_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.path_utils import get_path_resolver\n",
      "metadata": {
        "module": "ocr.core.utils.path_utils",
        "name": "get_path_resolver",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config_utils.py",
      "line": 16,
      "column": 0,
      "pattern": "from omegaconf import ListConfig",
      "context": "Imports ListConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "\nfrom omegaconf import ListConfig, OmegaConf\n",
      "metadata": {
        "module": "omegaconf",
        "name": "ListConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config_utils.py",
      "line": 16,
      "column": 0,
      "pattern": "from omegaconf import OmegaConf",
      "context": "Imports OmegaConf from omegaconf",
      "category": "from_import",
      "code_snippet": "\nfrom omegaconf import ListConfig, OmegaConf\n",
      "metadata": {
        "module": "omegaconf",
        "name": "OmegaConf",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config_utils.py",
      "line": 61,
      "column": 4,
      "pattern": "from omegaconf import OmegaConf",
      "context": "Imports OmegaConf from omegaconf",
      "category": "from_import",
      "code_snippet": "\n    from omegaconf import OmegaConf\n",
      "metadata": {
        "module": "omegaconf",
        "name": "OmegaConf",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config_utils.py",
      "line": 252,
      "column": 4,
      "pattern": "from ocr.core import registry",
      "context": "Imports registry from ocr.core",
      "category": "from_import",
      "code_snippet": "    # Local import to avoid circular dependency\n    from ocr.core import registry\n",
      "metadata": {
        "module": "ocr.core",
        "name": "registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/config_validation.py",
      "line": 10,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/convert_submission.py",
      "line": 6,
      "column": 0,
      "pattern": "import pandas",
      "context": "Imports module: pandas",
      "category": "import",
      "code_snippet": "\nimport pandas as pd\n",
      "metadata": {
        "module": "pandas",
        "alias": "pd"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/data_utils.py",
      "line": 5,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/experiment_index.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/experiment_name.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/experiment_name.py",
      "line": 10,
      "column": 0,
      "pattern": "import yaml",
      "context": "Imports module: yaml",
      "category": "import",
      "code_snippet": "\nimport yaml\n",
      "metadata": {
        "module": "yaml",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/experiment_name.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/geometry_utils.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/geometry_utils.py",
      "line": 7,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/image_loading.py",
      "line": 25,
      "column": 4,
      "pattern": "from turbojpeg import TurboJPEG",
      "context": "Imports TurboJPEG from turbojpeg",
      "category": "from_import",
      "code_snippet": "try:\n    from turbojpeg import TurboJPEG\n",
      "metadata": {
        "module": "turbojpeg",
        "name": "TurboJPEG",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/image_loading.py",
      "line": 31,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "\nfrom PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/image_utils.py",
      "line": 53,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/image_utils.py",
      "line": 57,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom PIL import Image",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/image_utils.py",
      "line": 58,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/image_utils.py",
      "line": 60,
      "column": 0,
      "pattern": "from ocr.core.validation import ImageLoadingConfig",
      "context": "Imports ImageLoadingConfig from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import ImageLoadingConfig\nfrom ocr.core.utils.image_loading import load_image_optimized",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ImageLoadingConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/image_utils.py",
      "line": 61,
      "column": 0,
      "pattern": "from ocr.core.utils.image_loading import load_image_optimized",
      "context": "Imports load_image_optimized from ocr.core.utils.image_loading",
      "category": "from_import",
      "code_snippet": "from ocr.core.validation import ImageLoadingConfig\nfrom ocr.core.utils.image_loading import load_image_optimized\n",
      "metadata": {
        "module": "ocr.core.utils.image_loading",
        "name": "load_image_optimized",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logger_factory.py",
      "line": 7,
      "column": 0,
      "pattern": "from lightning.pytorch.loggers import Logger",
      "context": "Imports Logger from lightning.pytorch.loggers",
      "category": "from_import",
      "code_snippet": "\nfrom lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger\nfrom omegaconf import DictConfig, OmegaConf",
      "metadata": {
        "module": "lightning.pytorch.loggers",
        "name": "Logger",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logger_factory.py",
      "line": 7,
      "column": 0,
      "pattern": "from lightning.pytorch.loggers import TensorBoardLogger",
      "context": "Imports TensorBoardLogger from lightning.pytorch.loggers",
      "category": "from_import",
      "code_snippet": "\nfrom lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger\nfrom omegaconf import DictConfig, OmegaConf",
      "metadata": {
        "module": "lightning.pytorch.loggers",
        "name": "TensorBoardLogger",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logger_factory.py",
      "line": 7,
      "column": 0,
      "pattern": "from lightning.pytorch.loggers import WandbLogger",
      "context": "Imports WandbLogger from lightning.pytorch.loggers",
      "category": "from_import",
      "code_snippet": "\nfrom lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger\nfrom omegaconf import DictConfig, OmegaConf",
      "metadata": {
        "module": "lightning.pytorch.loggers",
        "name": "WandbLogger",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logger_factory.py",
      "line": 8,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger\nfrom omegaconf import DictConfig, OmegaConf\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logger_factory.py",
      "line": 8,
      "column": 0,
      "pattern": "from omegaconf import OmegaConf",
      "context": "Imports OmegaConf from omegaconf",
      "category": "from_import",
      "code_snippet": "from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger\nfrom omegaconf import DictConfig, OmegaConf\n",
      "metadata": {
        "module": "omegaconf",
        "name": "OmegaConf",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logger_factory.py",
      "line": 10,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logger_factory.py",
      "line": 61,
      "column": 4,
      "pattern": "from ocr.core.utils.wandb_utils import generate_run_name",
      "context": "Imports generate_run_name from ocr.core.utils.wandb_utils",
      "category": "from_import",
      "code_snippet": "    \"\"\"Create Weights & Biases logger (default).\"\"\"\n    from ocr.core.utils.wandb_utils import generate_run_name, load_env_variables\n",
      "metadata": {
        "module": "ocr.core.utils.wandb_utils",
        "name": "generate_run_name",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logger_factory.py",
      "line": 61,
      "column": 4,
      "pattern": "from ocr.core.utils.wandb_utils import load_env_variables",
      "context": "Imports load_env_variables from ocr.core.utils.wandb_utils",
      "category": "from_import",
      "code_snippet": "    \"\"\"Create Weights & Biases logger (default).\"\"\"\n    from ocr.core.utils.wandb_utils import generate_run_name, load_env_variables\n",
      "metadata": {
        "module": "ocr.core.utils.wandb_utils",
        "name": "load_env_variables",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 15,
      "column": 4,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "try:\n    import torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 22,
      "column": 4,
      "pattern": "from rich.console import Console",
      "context": "Imports Console from rich.console",
      "category": "from_import",
      "code_snippet": "try:\n    from rich.console import Console\n    from rich.logging import RichHandler",
      "metadata": {
        "module": "rich.console",
        "name": "Console",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 23,
      "column": 4,
      "pattern": "from rich.logging import RichHandler",
      "context": "Imports RichHandler from rich.logging",
      "category": "from_import",
      "code_snippet": "    from rich.console import Console\n    from rich.logging import RichHandler\n    from rich.panel import Panel",
      "metadata": {
        "module": "rich.logging",
        "name": "RichHandler",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 24,
      "column": 4,
      "pattern": "from rich.panel import Panel",
      "context": "Imports Panel from rich.panel",
      "category": "from_import",
      "code_snippet": "    from rich.logging import RichHandler\n    from rich.panel import Panel\n    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn",
      "metadata": {
        "module": "rich.panel",
        "name": "Panel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 25,
      "column": 4,
      "pattern": "from rich.progress import BarColumn",
      "context": "Imports BarColumn from rich.progress",
      "category": "from_import",
      "code_snippet": "    from rich.panel import Panel\n    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn\n    from rich.table import Table",
      "metadata": {
        "module": "rich.progress",
        "name": "BarColumn",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 25,
      "column": 4,
      "pattern": "from rich.progress import Progress",
      "context": "Imports Progress from rich.progress",
      "category": "from_import",
      "code_snippet": "    from rich.panel import Panel\n    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn\n    from rich.table import Table",
      "metadata": {
        "module": "rich.progress",
        "name": "Progress",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 25,
      "column": 4,
      "pattern": "from rich.progress import SpinnerColumn",
      "context": "Imports SpinnerColumn from rich.progress",
      "category": "from_import",
      "code_snippet": "    from rich.panel import Panel\n    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn\n    from rich.table import Table",
      "metadata": {
        "module": "rich.progress",
        "name": "SpinnerColumn",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 25,
      "column": 4,
      "pattern": "from rich.progress import TextColumn",
      "context": "Imports TextColumn from rich.progress",
      "category": "from_import",
      "code_snippet": "    from rich.panel import Panel\n    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn\n    from rich.table import Table",
      "metadata": {
        "module": "rich.progress",
        "name": "TextColumn",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 25,
      "column": 4,
      "pattern": "from rich.progress import TimeElapsedColumn",
      "context": "Imports TimeElapsedColumn from rich.progress",
      "category": "from_import",
      "code_snippet": "    from rich.panel import Panel\n    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn\n    from rich.table import Table",
      "metadata": {
        "module": "rich.progress",
        "name": "TimeElapsedColumn",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 26,
      "column": 4,
      "pattern": "from rich.table import Table",
      "context": "Imports Table from rich.table",
      "category": "from_import",
      "code_snippet": "    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn\n    from rich.table import Table\n",
      "metadata": {
        "module": "rich.table",
        "name": "Table",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 35,
      "column": 4,
      "pattern": "from icecream import ic",
      "context": "Imports ic from icecream",
      "category": "from_import",
      "code_snippet": "try:\n    from icecream import ic, install  # type: ignore\n",
      "metadata": {
        "module": "icecream",
        "name": "ic",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 35,
      "column": 4,
      "pattern": "from icecream import install",
      "context": "Imports install from icecream",
      "category": "from_import",
      "code_snippet": "try:\n    from icecream import ic, install  # type: ignore\n",
      "metadata": {
        "module": "icecream",
        "name": "install",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/logging.py",
      "line": 192,
      "column": 16,
      "pattern": "from tqdm import tqdm",
      "context": "Imports tqdm from tqdm",
      "category": "from_import",
      "code_snippet": "            try:\n                from tqdm import tqdm\n",
      "metadata": {
        "module": "tqdm",
        "name": "tqdm",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/ocr_utils.py",
      "line": 1,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/ocr_utils.py",
      "line": 2,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/ocr_utils.py",
      "line": 4,
      "column": 0,
      "pattern": "from text_rendering import put_text_with_outline",
      "context": "Imports put_text_with_outline from text_rendering",
      "category": "from_import",
      "code_snippet": "\nfrom .text_rendering import put_text_with_outline\n",
      "metadata": {
        "module": "text_rendering",
        "name": "put_text_with_outline",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/orientation.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/orientation.py",
      "line": 11,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom PIL import Image",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/orientation.py",
      "line": 12,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/orientation.py",
      "line": 14,
      "column": 0,
      "pattern": "from orientation_constants import EXIF_ORIENTATION_TAG",
      "context": "Imports EXIF_ORIENTATION_TAG from orientation_constants",
      "category": "from_import",
      "code_snippet": "\nfrom .orientation_constants import (\n    EXIF_ORIENTATION_TAG,",
      "metadata": {
        "module": "orientation_constants",
        "name": "EXIF_ORIENTATION_TAG",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/orientation.py",
      "line": 14,
      "column": 0,
      "pattern": "from orientation_constants import VALID_ORIENTATIONS",
      "context": "Imports VALID_ORIENTATIONS from orientation_constants",
      "category": "from_import",
      "code_snippet": "\nfrom .orientation_constants import (\n    EXIF_ORIENTATION_TAG,",
      "metadata": {
        "module": "orientation_constants",
        "name": "VALID_ORIENTATIONS",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/orientation.py",
      "line": 14,
      "column": 0,
      "pattern": "from orientation_constants import get_orientation_transform",
      "context": "Imports get_orientation_transform from orientation_constants",
      "category": "from_import",
      "code_snippet": "\nfrom .orientation_constants import (\n    EXIF_ORIENTATION_TAG,",
      "metadata": {
        "module": "orientation_constants",
        "name": "get_orientation_transform",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/orientation_constants.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/orientation_constants.py",
      "line": 12,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom PIL import Image",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/orientation_constants.py",
      "line": 13,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/path_utils.py",
      "line": 27,
      "column": 0,
      "pattern": "from omegaconf import OmegaConf",
      "context": "Imports OmegaConf from omegaconf",
      "category": "from_import",
      "code_snippet": "\nfrom omegaconf import OmegaConf\n",
      "metadata": {
        "module": "omegaconf",
        "name": "OmegaConf",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/path_utils.py",
      "line": 29,
      "column": 0,
      "pattern": "from ocr.core.utils.experiment_index import get_next_experiment_index",
      "context": "Imports get_next_experiment_index from ocr.core.utils.experiment_index",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.experiment_index import get_next_experiment_index\n",
      "metadata": {
        "module": "ocr.core.utils.experiment_index",
        "name": "get_next_experiment_index",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from core import calculate_target_dimensions",
      "context": "Imports calculate_target_dimensions from core",
      "category": "from_import",
      "code_snippet": "\nfrom .core import (\n    calculate_target_dimensions,",
      "metadata": {
        "module": "core",
        "name": "calculate_target_dimensions",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from core import correct_perspective_from_mask",
      "context": "Imports correct_perspective_from_mask from core",
      "category": "from_import",
      "code_snippet": "\nfrom .core import (\n    calculate_target_dimensions,",
      "metadata": {
        "module": "core",
        "name": "correct_perspective_from_mask",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from core import four_point_transform",
      "context": "Imports four_point_transform from core",
      "category": "from_import",
      "code_snippet": "\nfrom .core import (\n    calculate_target_dimensions,",
      "metadata": {
        "module": "core",
        "name": "four_point_transform",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from core import remove_background_and_mask",
      "context": "Imports remove_background_and_mask from core",
      "category": "from_import",
      "code_snippet": "\nfrom .core import (\n    calculate_target_dimensions,",
      "metadata": {
        "module": "core",
        "name": "remove_background_and_mask",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from core import transform_polygons_inverse",
      "context": "Imports transform_polygons_inverse from core",
      "category": "from_import",
      "code_snippet": "\nfrom .core import (\n    calculate_target_dimensions,",
      "metadata": {
        "module": "core",
        "name": "transform_polygons_inverse",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/__init__.py",
      "line": 13,
      "column": 0,
      "pattern": "from fitting import fit_mask_rectangle",
      "context": "Imports fit_mask_rectangle from fitting",
      "category": "from_import",
      "code_snippet": ")\nfrom .fitting import fit_mask_rectangle\nfrom .types import LineQualityReport, MaskRectangleResult",
      "metadata": {
        "module": "fitting",
        "name": "fit_mask_rectangle",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/core.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/core.py",
      "line": 7,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/core.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/core.py",
      "line": 10,
      "column": 0,
      "pattern": "from fitting import fit_mask_rectangle",
      "context": "Imports fit_mask_rectangle from fitting",
      "category": "from_import",
      "code_snippet": "\nfrom .fitting import fit_mask_rectangle\nfrom .types import MaskRectangleResult",
      "metadata": {
        "module": "fitting",
        "name": "fit_mask_rectangle",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/core.py",
      "line": 15,
      "column": 4,
      "pattern": "from rembg import remove",
      "context": "Imports remove from rembg",
      "category": "from_import",
      "code_snippet": "try:\n    from rembg import remove as _rembg_remove\n",
      "metadata": {
        "module": "rembg",
        "name": "remove",
        "alias": "_rembg_remove"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 8,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 9,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 11,
      "column": 0,
      "pattern": "from geometry import _blend_corners",
      "context": "Imports _blend_corners from geometry",
      "category": "from_import",
      "code_snippet": "\nfrom .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points\nfrom .quality_metrics import (",
      "metadata": {
        "module": "geometry",
        "name": "_blend_corners",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 11,
      "column": 0,
      "pattern": "from geometry import _geometric_synthesis",
      "context": "Imports _geometric_synthesis from geometry",
      "category": "from_import",
      "code_snippet": "\nfrom .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points\nfrom .quality_metrics import (",
      "metadata": {
        "module": "geometry",
        "name": "_geometric_synthesis",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 11,
      "column": 0,
      "pattern": "from geometry import _intersect_lines",
      "context": "Imports _intersect_lines from geometry",
      "category": "from_import",
      "code_snippet": "\nfrom .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points\nfrom .quality_metrics import (",
      "metadata": {
        "module": "geometry",
        "name": "_intersect_lines",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 11,
      "column": 0,
      "pattern": "from geometry import _order_points",
      "context": "Imports _order_points from geometry",
      "category": "from_import",
      "code_snippet": "\nfrom .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points\nfrom .quality_metrics import (",
      "metadata": {
        "module": "geometry",
        "name": "_order_points",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 12,
      "column": 0,
      "pattern": "from quality_metrics import _compute_corner_sharpness_deviation",
      "context": "Imports _compute_corner_sharpness_deviation from quality_metrics",
      "category": "from_import",
      "code_snippet": "from .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points\nfrom .quality_metrics import (\n    _compute_corner_sharpness_deviation,",
      "metadata": {
        "module": "quality_metrics",
        "name": "_compute_corner_sharpness_deviation",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 12,
      "column": 0,
      "pattern": "from quality_metrics import _compute_edge_support_metrics",
      "context": "Imports _compute_edge_support_metrics from quality_metrics",
      "category": "from_import",
      "code_snippet": "from .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points\nfrom .quality_metrics import (\n    _compute_corner_sharpness_deviation,",
      "metadata": {
        "module": "quality_metrics",
        "name": "_compute_edge_support_metrics",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 12,
      "column": 0,
      "pattern": "from quality_metrics import _compute_linearity_rmse",
      "context": "Imports _compute_linearity_rmse from quality_metrics",
      "category": "from_import",
      "code_snippet": "from .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points\nfrom .quality_metrics import (\n    _compute_corner_sharpness_deviation,",
      "metadata": {
        "module": "quality_metrics",
        "name": "_compute_linearity_rmse",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 12,
      "column": 0,
      "pattern": "from quality_metrics import _compute_parallelism_misalignment",
      "context": "Imports _compute_parallelism_misalignment from quality_metrics",
      "category": "from_import",
      "code_snippet": "from .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points\nfrom .quality_metrics import (\n    _compute_corner_sharpness_deviation,",
      "metadata": {
        "module": "quality_metrics",
        "name": "_compute_parallelism_misalignment",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 12,
      "column": 0,
      "pattern": "from quality_metrics import _compute_solidity_metrics",
      "context": "Imports _compute_solidity_metrics from quality_metrics",
      "category": "from_import",
      "code_snippet": "from .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points\nfrom .quality_metrics import (\n    _compute_corner_sharpness_deviation,",
      "metadata": {
        "module": "quality_metrics",
        "name": "_compute_solidity_metrics",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 20,
      "column": 0,
      "pattern": "from validation import _validate_contour_alignment",
      "context": "Imports _validate_contour_alignment from validation",
      "category": "from_import",
      "code_snippet": "from .types import LineQualityReport, MaskRectangleResult\nfrom .validation import (\n    _validate_contour_alignment,",
      "metadata": {
        "module": "validation",
        "name": "_validate_contour_alignment",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 20,
      "column": 0,
      "pattern": "from validation import _validate_contour_segments",
      "context": "Imports _validate_contour_segments from validation",
      "category": "from_import",
      "code_snippet": "from .types import LineQualityReport, MaskRectangleResult\nfrom .validation import (\n    _validate_contour_alignment,",
      "metadata": {
        "module": "validation",
        "name": "_validate_contour_segments",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 20,
      "column": 0,
      "pattern": "from validation import _validate_edge_angles",
      "context": "Imports _validate_edge_angles from validation",
      "category": "from_import",
      "code_snippet": "from .types import LineQualityReport, MaskRectangleResult\nfrom .validation import (\n    _validate_contour_alignment,",
      "metadata": {
        "module": "validation",
        "name": "_validate_edge_angles",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/fitting.py",
      "line": 20,
      "column": 0,
      "pattern": "from validation import _validate_edge_lengths",
      "context": "Imports _validate_edge_lengths from validation",
      "category": "from_import",
      "code_snippet": "from .types import LineQualityReport, MaskRectangleResult\nfrom .validation import (\n    _validate_contour_alignment,",
      "metadata": {
        "module": "validation",
        "name": "_validate_edge_lengths",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/geometry.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/geometry.py",
      "line": 7,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/geometry.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/quality_metrics.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/quality_metrics.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/quality_metrics.py",
      "line": 10,
      "column": 0,
      "pattern": "from geometry import _compute_edge_vectors",
      "context": "Imports _compute_edge_vectors from geometry",
      "category": "from_import",
      "code_snippet": "\nfrom .geometry import _compute_edge_vectors\n",
      "metadata": {
        "module": "geometry",
        "name": "_compute_edge_vectors",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/types.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/types.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/validation.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/validation.py",
      "line": 7,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/validation.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/perspective_correction/validation.py",
      "line": 10,
      "column": 0,
      "pattern": "from geometry import _compute_edge_vectors",
      "context": "Imports _compute_edge_vectors from geometry",
      "category": "from_import",
      "code_snippet": "\nfrom .geometry import _compute_edge_vectors\n",
      "metadata": {
        "module": "geometry",
        "name": "_compute_edge_vectors",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/polygon_utils.py",
      "line": 53,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/polygon_utils.py",
      "line": 210,
      "column": 4,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "    \"\"\"\n    import cv2\n",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/registry.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/registry.py",
      "line": 8,
      "column": 0,
      "pattern": "from interfaces.losses import BaseLoss",
      "context": "Imports BaseLoss from interfaces.losses",
      "category": "from_import",
      "code_snippet": "\nfrom ..interfaces.losses import BaseLoss\nfrom ..interfaces.metrics import BaseMetric",
      "metadata": {
        "module": "interfaces.losses",
        "name": "BaseLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/registry.py",
      "line": 9,
      "column": 0,
      "pattern": "from interfaces.metrics import BaseMetric",
      "context": "Imports BaseMetric from interfaces.metrics",
      "category": "from_import",
      "code_snippet": "from ..interfaces.losses import BaseLoss\nfrom ..interfaces.metrics import BaseMetric\nfrom ..interfaces.models import BaseDecoder, BaseEncoder, BaseHead",
      "metadata": {
        "module": "interfaces.metrics",
        "name": "BaseMetric",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/registry.py",
      "line": 10,
      "column": 0,
      "pattern": "from interfaces.models import BaseDecoder",
      "context": "Imports BaseDecoder from interfaces.models",
      "category": "from_import",
      "code_snippet": "from ..interfaces.metrics import BaseMetric\nfrom ..interfaces.models import BaseDecoder, BaseEncoder, BaseHead\n",
      "metadata": {
        "module": "interfaces.models",
        "name": "BaseDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/registry.py",
      "line": 10,
      "column": 0,
      "pattern": "from interfaces.models import BaseEncoder",
      "context": "Imports BaseEncoder from interfaces.models",
      "category": "from_import",
      "code_snippet": "from ..interfaces.metrics import BaseMetric\nfrom ..interfaces.models import BaseDecoder, BaseEncoder, BaseHead\n",
      "metadata": {
        "module": "interfaces.models",
        "name": "BaseEncoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/registry.py",
      "line": 10,
      "column": 0,
      "pattern": "from interfaces.models import BaseHead",
      "context": "Imports BaseHead from interfaces.models",
      "category": "from_import",
      "code_snippet": "from ..interfaces.metrics import BaseMetric\nfrom ..interfaces.models import BaseDecoder, BaseEncoder, BaseHead\n",
      "metadata": {
        "module": "interfaces.models",
        "name": "BaseHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/sepia_enhancement.py",
      "line": 12,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/sepia_enhancement.py",
      "line": 14,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/sepia_enhancement.py",
      "line": 15,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/submission.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/submission.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/text_rendering.py",
      "line": 8,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/text_rendering.py",
      "line": 10,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/text_rendering.py",
      "line": 11,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\nfrom PIL import Image, ImageDraw, ImageFont",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/text_rendering.py",
      "line": 12,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/text_rendering.py",
      "line": 12,
      "column": 0,
      "pattern": "from PIL import ImageDraw",
      "context": "Imports ImageDraw from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "ImageDraw",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/text_rendering.py",
      "line": 12,
      "column": 0,
      "pattern": "from PIL import ImageFont",
      "context": "Imports ImageFont from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "ImageFont",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 5,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 16,
      "column": 4,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "if TYPE_CHECKING:\n    import numpy as np\n    from omegaconf import DictConfig",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 17,
      "column": 4,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "    import numpy as np\n    from omegaconf import DictConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 27,
      "column": 4,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "    \"\"\"\n    import wandb\n",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 49,
      "column": 32,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "                            try:\n                                import wandb\n",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 68,
      "column": 4,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "    \"\"\"Safely retrieve a nested value from a DictConfig or mapping.\"\"\"\n    from omegaconf import DictConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 162,
      "column": 4,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "    \"\"\"\n    import cv2\n    import numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 163,
      "column": 4,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "    import cv2\n    import numpy as np\n    import torch",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 164,
      "column": 4,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "    import numpy as np\n    import torch\n    from PIL import Image as PILImage",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 165,
      "column": 4,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "    import torch\n    from PIL import Image as PILImage\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": "PILImage"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 217,
      "column": 4,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "    \"\"\"\n    import numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 241,
      "column": 8,
      "pattern": "from ocr.core import registry",
      "context": "Imports registry from ocr.core",
      "category": "from_import",
      "code_snippet": "    try:\n        from ocr.core import registry as _registry\n",
      "metadata": {
        "module": "ocr.core",
        "name": "registry",
        "alias": "_registry"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 429,
      "column": 4,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "    \"\"\"Finalize the active W&B run with the most relevant metric.\"\"\"\n    import wandb\n",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 512,
      "column": 4,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "    # Lazy imports to reduce module coupling\n    import cv2\n    import numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 513,
      "column": 4,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "    import cv2\n    import numpy as np\n    import wandb",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 514,
      "column": 4,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "    import numpy as np\n    import wandb\n",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 516,
      "column": 4,
      "pattern": "from text_rendering import put_text_utf8",
      "context": "Imports put_text_utf8 from text_rendering",
      "category": "from_import",
      "code_snippet": "\n    from .text_rendering import put_text_utf8\n",
      "metadata": {
        "module": "text_rendering",
        "name": "put_text_utf8",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 729,
      "column": 4,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "    \"\"\"\n    import cv2\n    import numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 730,
      "column": 4,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "    import cv2\n    import numpy as np\n    import wandb",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 731,
      "column": 4,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "    import numpy as np\n    import wandb\n    from PIL import Image, ImageDraw, ImageFont",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 732,
      "column": 4,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "    import wandb\n    from PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 732,
      "column": 4,
      "pattern": "from PIL import ImageDraw",
      "context": "Imports ImageDraw from PIL",
      "category": "from_import",
      "code_snippet": "    import wandb\n    from PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "ImageDraw",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py",
      "line": 732,
      "column": 4,
      "pattern": "from PIL import ImageFont",
      "context": "Imports ImageFont from PIL",
      "category": "from_import",
      "code_snippet": "    import wandb\n    from PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "ImageFont",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 13,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 20,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nimport torch",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 21,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import numpy as np\nimport torch\nfrom pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 22,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator\nfrom pydantic_core import InitErrorDetails, PydanticCustomError",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 22,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator\nfrom pydantic_core import InitErrorDetails, PydanticCustomError",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 22,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator\nfrom pydantic_core import InitErrorDetails, PydanticCustomError",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 22,
      "column": 0,
      "pattern": "from pydantic import ValidationError",
      "context": "Imports ValidationError from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator\nfrom pydantic_core import InitErrorDetails, PydanticCustomError",
      "metadata": {
        "module": "pydantic",
        "name": "ValidationError",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 22,
      "column": 0,
      "pattern": "from pydantic import ValidationInfo",
      "context": "Imports ValidationInfo from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator\nfrom pydantic_core import InitErrorDetails, PydanticCustomError",
      "metadata": {
        "module": "pydantic",
        "name": "ValidationInfo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 22,
      "column": 0,
      "pattern": "from pydantic import field_validator",
      "context": "Imports field_validator from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator\nfrom pydantic_core import InitErrorDetails, PydanticCustomError",
      "metadata": {
        "module": "pydantic",
        "name": "field_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 22,
      "column": 0,
      "pattern": "from pydantic import model_validator",
      "context": "Imports model_validator from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator\nfrom pydantic_core import InitErrorDetails, PydanticCustomError",
      "metadata": {
        "module": "pydantic",
        "name": "model_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 23,
      "column": 0,
      "pattern": "from pydantic_core import InitErrorDetails",
      "context": "Imports InitErrorDetails from pydantic_core",
      "category": "from_import",
      "code_snippet": "from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator\nfrom pydantic_core import InitErrorDetails, PydanticCustomError\n",
      "metadata": {
        "module": "pydantic_core",
        "name": "InitErrorDetails",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/validation.py",
      "line": 23,
      "column": 0,
      "pattern": "from pydantic_core import PydanticCustomError",
      "context": "Imports PydanticCustomError from pydantic_core",
      "category": "from_import",
      "code_snippet": "from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator\nfrom pydantic_core import InitErrorDetails, PydanticCustomError\n",
      "metadata": {
        "module": "pydantic_core",
        "name": "PydanticCustomError",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/__init__.py",
      "line": 45,
      "column": 4,
      "pattern": "from hydra.utils import instantiate",
      "context": "Imports instantiate from hydra.utils",
      "category": "from_import",
      "code_snippet": "def get_datasets_by_cfg(datasets_config, data_config=None, full_config=None):\n    from hydra.utils import instantiate\n    from omegaconf import OmegaConf",
      "metadata": {
        "module": "hydra.utils",
        "name": "instantiate",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/__init__.py",
      "line": 46,
      "column": 4,
      "pattern": "from omegaconf import OmegaConf",
      "context": "Imports OmegaConf from omegaconf",
      "category": "from_import",
      "code_snippet": "    from hydra.utils import instantiate\n    from omegaconf import OmegaConf\n    from torch.utils.data import Subset",
      "metadata": {
        "module": "omegaconf",
        "name": "OmegaConf",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/__init__.py",
      "line": 47,
      "column": 4,
      "pattern": "from torch.utils.data import Subset",
      "context": "Imports Subset from torch.utils.data",
      "category": "from_import",
      "code_snippet": "    from omegaconf import OmegaConf\n    from torch.utils.data import Subset\n",
      "metadata": {
        "module": "torch.utils.data",
        "name": "Subset",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 57,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom PIL import Image",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 58,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom PIL import Image\nfrom pydantic import ValidationError",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 59,
      "column": 0,
      "pattern": "from pydantic import ValidationError",
      "context": "Imports ValidationError from pydantic",
      "category": "from_import",
      "code_snippet": "from PIL import Image\nfrom pydantic import ValidationError\nfrom torch.utils.data import Dataset",
      "metadata": {
        "module": "pydantic",
        "name": "ValidationError",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 60,
      "column": 0,
      "pattern": "from torch.utils.data import Dataset",
      "context": "Imports Dataset from torch.utils.data",
      "category": "from_import",
      "code_snippet": "from pydantic import ValidationError\nfrom torch.utils.data import Dataset\n",
      "metadata": {
        "module": "torch.utils.data",
        "name": "Dataset",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 66,
      "column": 0,
      "pattern": "from ocr.core.validation import DatasetConfig",
      "context": "Imports DatasetConfig from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import DatasetConfig, ImageData, ImageMetadata, PolygonData, TransformInput, ValidatedPolygonData\nfrom ocr.core.utils.background_normalization import normalize_gray_world",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "DatasetConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 66,
      "column": 0,
      "pattern": "from ocr.core.validation import ImageData",
      "context": "Imports ImageData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import DatasetConfig, ImageData, ImageMetadata, PolygonData, TransformInput, ValidatedPolygonData\nfrom ocr.core.utils.background_normalization import normalize_gray_world",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ImageData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 66,
      "column": 0,
      "pattern": "from ocr.core.validation import ImageMetadata",
      "context": "Imports ImageMetadata from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import DatasetConfig, ImageData, ImageMetadata, PolygonData, TransformInput, ValidatedPolygonData\nfrom ocr.core.utils.background_normalization import normalize_gray_world",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ImageMetadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 66,
      "column": 0,
      "pattern": "from ocr.core.validation import PolygonData",
      "context": "Imports PolygonData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import DatasetConfig, ImageData, ImageMetadata, PolygonData, TransformInput, ValidatedPolygonData\nfrom ocr.core.utils.background_normalization import normalize_gray_world",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "PolygonData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 66,
      "column": 0,
      "pattern": "from ocr.core.validation import TransformInput",
      "context": "Imports TransformInput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import DatasetConfig, ImageData, ImageMetadata, PolygonData, TransformInput, ValidatedPolygonData\nfrom ocr.core.utils.background_normalization import normalize_gray_world",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "TransformInput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 66,
      "column": 0,
      "pattern": "from ocr.core.validation import ValidatedPolygonData",
      "context": "Imports ValidatedPolygonData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import DatasetConfig, ImageData, ImageMetadata, PolygonData, TransformInput, ValidatedPolygonData\nfrom ocr.core.utils.background_normalization import normalize_gray_world",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ValidatedPolygonData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 67,
      "column": 0,
      "pattern": "from ocr.core.utils.background_normalization import normalize_gray_world",
      "context": "Imports normalize_gray_world from ocr.core.utils.background_normalization",
      "category": "from_import",
      "code_snippet": "from ocr.core.validation import DatasetConfig, ImageData, ImageMetadata, PolygonData, TransformInput, ValidatedPolygonData\nfrom ocr.core.utils.background_normalization import normalize_gray_world\nfrom ocr.core.utils.orientation import (",
      "metadata": {
        "module": "ocr.core.utils.background_normalization",
        "name": "normalize_gray_world",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 68,
      "column": 0,
      "pattern": "from ocr.core.utils.orientation import EXIF_ORIENTATION_TAG",
      "context": "Imports EXIF_ORIENTATION_TAG from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.background_normalization import normalize_gray_world\nfrom ocr.core.utils.orientation import (\n    EXIF_ORIENTATION_TAG,",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "EXIF_ORIENTATION_TAG",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 68,
      "column": 0,
      "pattern": "from ocr.core.utils.orientation import normalize_pil_image",
      "context": "Imports normalize_pil_image from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.background_normalization import normalize_gray_world\nfrom ocr.core.utils.orientation import (\n    EXIF_ORIENTATION_TAG,",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "normalize_pil_image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 137,
      "column": 8,
      "pattern": "from ocr.core.utils.cache_manager import CacheManager",
      "context": "Imports CacheManager from ocr.core.utils.cache_manager",
      "category": "from_import",
      "code_snippet": "        # Instantiate CacheManager using configuration from config with versioning\n        from ocr.core.utils.cache_manager import CacheManager\n",
      "metadata": {
        "module": "ocr.core.utils.cache_manager",
        "name": "CacheManager",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 445,
      "column": 8,
      "pattern": "from ocr.core.validation import DataItem",
      "context": "Imports DataItem from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "        # Cache contains fully processed DataItem objects\n        from ocr.core.validation import DataItem\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "DataItem",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 487,
      "column": 12,
      "pattern": "from ocr.core.utils.orientation import orientation_requires_rotation",
      "context": "Imports orientation_requires_rotation from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "            # Use polygon_utils to handle orientation remapping\n            from ocr.core.utils.orientation import orientation_requires_rotation, polygons_in_canonical_frame, remap_polygons\n",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "orientation_requires_rotation",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 487,
      "column": 12,
      "pattern": "from ocr.core.utils.orientation import polygons_in_canonical_frame",
      "context": "Imports polygons_in_canonical_frame from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "            # Use polygon_utils to handle orientation remapping\n            from ocr.core.utils.orientation import orientation_requires_rotation, polygons_in_canonical_frame, remap_polygons\n",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "polygons_in_canonical_frame",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 487,
      "column": 12,
      "pattern": "from ocr.core.utils.orientation import remap_polygons",
      "context": "Imports remap_polygons from ocr.core.utils.orientation",
      "category": "from_import",
      "code_snippet": "            # Use polygon_utils to handle orientation remapping\n            from ocr.core.utils.orientation import orientation_requires_rotation, polygons_in_canonical_frame, remap_polygons\n",
      "metadata": {
        "module": "ocr.core.utils.orientation",
        "name": "remap_polygons",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 510,
      "column": 8,
      "pattern": "from ocr.core.validation import TransformInput",
      "context": "Imports TransformInput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\n        from ocr.core.validation import TransformInput\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "TransformInput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 568,
      "column": 8,
      "pattern": "from ocr.core.utils.polygon_utils import ensure_polygon_array",
      "context": "Imports ensure_polygon_array from ocr.core.utils.polygon_utils",
      "category": "from_import",
      "code_snippet": "        # Filter degenerate polygons using polygon_utils\n        from ocr.core.utils.polygon_utils import ensure_polygon_array, filter_degenerate_polygons\n",
      "metadata": {
        "module": "ocr.core.utils.polygon_utils",
        "name": "ensure_polygon_array",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 568,
      "column": 8,
      "pattern": "from ocr.core.utils.polygon_utils import filter_degenerate_polygons",
      "context": "Imports filter_degenerate_polygons from ocr.core.utils.polygon_utils",
      "category": "from_import",
      "code_snippet": "        # Filter degenerate polygons using polygon_utils\n        from ocr.core.utils.polygon_utils import ensure_polygon_array, filter_degenerate_polygons\n",
      "metadata": {
        "module": "ocr.core.utils.polygon_utils",
        "name": "filter_degenerate_polygons",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 600,
      "column": 24,
      "pattern": "from ocr.core.utils.polygon_utils import validate_map_shapes",
      "context": "Imports validate_map_shapes from ocr.core.utils.polygon_utils",
      "category": "from_import",
      "code_snippet": "                        # Validate map shapes\n                        from ocr.core.utils.polygon_utils import validate_map_shapes\n",
      "metadata": {
        "module": "ocr.core.utils.polygon_utils",
        "name": "validate_map_shapes",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 612,
      "column": 28,
      "pattern": "from ocr.core.validation import MapData",
      "context": "Imports MapData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "                            # Cache the maps\n                            from ocr.core.validation import MapData\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "MapData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 700,
      "column": 8,
      "pattern": "from ocr.core.validation import ImageData",
      "context": "Imports ImageData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "        image_path = self.config.image_path / filename\n        from ocr.core.validation import ImageData\n        from ocr.core.utils.image_utils import ensure_rgb, load_pil_image, pil_to_numpy, safe_get_image_size",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ImageData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 701,
      "column": 8,
      "pattern": "from ocr.core.utils.image_utils import ensure_rgb",
      "context": "Imports ensure_rgb from ocr.core.utils.image_utils",
      "category": "from_import",
      "code_snippet": "        from ocr.core.validation import ImageData\n        from ocr.core.utils.image_utils import ensure_rgb, load_pil_image, pil_to_numpy, safe_get_image_size\n",
      "metadata": {
        "module": "ocr.core.utils.image_utils",
        "name": "ensure_rgb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 701,
      "column": 8,
      "pattern": "from ocr.core.utils.image_utils import load_pil_image",
      "context": "Imports load_pil_image from ocr.core.utils.image_utils",
      "category": "from_import",
      "code_snippet": "        from ocr.core.validation import ImageData\n        from ocr.core.utils.image_utils import ensure_rgb, load_pil_image, pil_to_numpy, safe_get_image_size\n",
      "metadata": {
        "module": "ocr.core.utils.image_utils",
        "name": "load_pil_image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 701,
      "column": 8,
      "pattern": "from ocr.core.utils.image_utils import pil_to_numpy",
      "context": "Imports pil_to_numpy from ocr.core.utils.image_utils",
      "category": "from_import",
      "code_snippet": "        from ocr.core.validation import ImageData\n        from ocr.core.utils.image_utils import ensure_rgb, load_pil_image, pil_to_numpy, safe_get_image_size\n",
      "metadata": {
        "module": "ocr.core.utils.image_utils",
        "name": "pil_to_numpy",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 701,
      "column": 8,
      "pattern": "from ocr.core.utils.image_utils import safe_get_image_size",
      "context": "Imports safe_get_image_size from ocr.core.utils.image_utils",
      "category": "from_import",
      "code_snippet": "        from ocr.core.validation import ImageData\n        from ocr.core.utils.image_utils import ensure_rgb, load_pil_image, pil_to_numpy, safe_get_image_size\n",
      "metadata": {
        "module": "ocr.core.utils.image_utils",
        "name": "safe_get_image_size",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 719,
      "column": 12,
      "pattern": "from ocr.core.utils.image_utils import prenormalize_imagenet",
      "context": "Imports prenormalize_imagenet from ocr.core.utils.image_utils",
      "category": "from_import",
      "code_snippet": "        if self.config.prenormalize_images:\n            from ocr.core.utils.image_utils import prenormalize_imagenet\n",
      "metadata": {
        "module": "ocr.core.utils.image_utils",
        "name": "prenormalize_imagenet",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 751,
      "column": 8,
      "pattern": "from tqdm import tqdm",
      "context": "Imports tqdm from tqdm",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from tqdm import tqdm\n",
      "metadata": {
        "module": "tqdm",
        "name": "tqdm",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 785,
      "column": 8,
      "pattern": "from tqdm import tqdm",
      "context": "Imports tqdm from tqdm",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from tqdm import tqdm\n",
      "metadata": {
        "module": "tqdm",
        "name": "tqdm",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/base.py",
      "line": 806,
      "column": 24,
      "pattern": "from ocr.core.validation import MapData",
      "context": "Imports MapData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "                    if prob_map.ndim == 3 and thresh_map.ndim == 3:\n                        from ocr.core.validation import MapData\n",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "MapData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/craft_collate_fn.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/craft_collate_fn.py",
      "line": 9,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/craft_collate_fn.py",
      "line": 10,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\nimport torch",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/craft_collate_fn.py",
      "line": 11,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import numpy as np\nimport torch\nfrom numpy.typing import NDArray",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/craft_collate_fn.py",
      "line": 12,
      "column": 0,
      "pattern": "from numpy.typing import NDArray",
      "context": "Imports NDArray from numpy.typing",
      "category": "from_import",
      "code_snippet": "import torch\nfrom numpy.typing import NDArray\n",
      "metadata": {
        "module": "numpy.typing",
        "name": "NDArray",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/craft_collate_fn.py",
      "line": 14,
      "column": 0,
      "pattern": "from ocr.core.utils.data_utils import extract_metadata",
      "context": "Imports extract_metadata from ocr.core.utils.data_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.data_utils import extract_metadata\n",
      "metadata": {
        "module": "ocr.core.utils.data_utils",
        "name": "extract_metadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/db_collate_fn.py",
      "line": 16,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/db_collate_fn.py",
      "line": 17,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\nimport pyclipper",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/db_collate_fn.py",
      "line": 18,
      "column": 0,
      "pattern": "import pyclipper",
      "context": "Imports module: pyclipper",
      "category": "import",
      "code_snippet": "import numpy as np\nimport pyclipper\nimport torch",
      "metadata": {
        "module": "pyclipper",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/db_collate_fn.py",
      "line": 19,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import pyclipper\nimport torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/db_collate_fn.py",
      "line": 21,
      "column": 0,
      "pattern": "from ocr.core.utils.data_utils import extract_metadata",
      "context": "Imports extract_metadata from ocr.core.utils.data_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.data_utils import extract_metadata\n",
      "metadata": {
        "module": "ocr.core.utils.data_utils",
        "name": "extract_metadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/db_collate_fn.py",
      "line": 336,
      "column": 8,
      "pattern": "from ocr.core.utils.polygon_utils import is_valid_polygon",
      "context": "Imports is_valid_polygon from ocr.core.utils.polygon_utils",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from ocr.core.utils.polygon_utils import is_valid_polygon\n",
      "metadata": {
        "module": "ocr.core.utils.polygon_utils",
        "name": "is_valid_polygon",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 5,
      "column": 0,
      "pattern": "from advanced_detector import AdvancedDetectionConfig",
      "context": "Imports AdvancedDetectionConfig from advanced_detector",
      "category": "from_import",
      "code_snippet": "\nfrom .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .advanced_preprocessor import (",
      "metadata": {
        "module": "advanced_detector",
        "name": "AdvancedDetectionConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 5,
      "column": 0,
      "pattern": "from advanced_detector import AdvancedDocumentDetector",
      "context": "Imports AdvancedDocumentDetector from advanced_detector",
      "category": "from_import",
      "code_snippet": "\nfrom .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .advanced_preprocessor import (",
      "metadata": {
        "module": "advanced_detector",
        "name": "AdvancedDocumentDetector",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from advanced_preprocessor import AdvancedDocumentPreprocessor",
      "context": "Imports AdvancedDocumentPreprocessor from advanced_preprocessor",
      "category": "from_import",
      "code_snippet": "from .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .advanced_preprocessor import (\n    AdvancedDocumentPreprocessor,",
      "metadata": {
        "module": "advanced_preprocessor",
        "name": "AdvancedDocumentPreprocessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from advanced_preprocessor import AdvancedPreprocessingConfig",
      "context": "Imports AdvancedPreprocessingConfig from advanced_preprocessor",
      "category": "from_import",
      "code_snippet": "from .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .advanced_preprocessor import (\n    AdvancedDocumentPreprocessor,",
      "metadata": {
        "module": "advanced_preprocessor",
        "name": "AdvancedPreprocessingConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from advanced_preprocessor import OfficeLensPreprocessorAlbumentations",
      "context": "Imports OfficeLensPreprocessorAlbumentations from advanced_preprocessor",
      "category": "from_import",
      "code_snippet": "from .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .advanced_preprocessor import (\n    AdvancedDocumentPreprocessor,",
      "metadata": {
        "module": "advanced_preprocessor",
        "name": "OfficeLensPreprocessorAlbumentations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from advanced_preprocessor import create_high_accuracy_preprocessor",
      "context": "Imports create_high_accuracy_preprocessor from advanced_preprocessor",
      "category": "from_import",
      "code_snippet": "from .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .advanced_preprocessor import (\n    AdvancedDocumentPreprocessor,",
      "metadata": {
        "module": "advanced_preprocessor",
        "name": "create_high_accuracy_preprocessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from advanced_preprocessor import create_legacy_office_lens_preprocessor",
      "context": "Imports create_legacy_office_lens_preprocessor from advanced_preprocessor",
      "category": "from_import",
      "code_snippet": "from .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .advanced_preprocessor import (\n    AdvancedDocumentPreprocessor,",
      "metadata": {
        "module": "advanced_preprocessor",
        "name": "create_legacy_office_lens_preprocessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 13,
      "column": 0,
      "pattern": "from config import DocumentPreprocessorConfig",
      "context": "Imports DocumentPreprocessorConfig from config",
      "category": "from_import",
      "code_snippet": ")\nfrom .config import DocumentPreprocessorConfig\nfrom .enhanced_pipeline import (",
      "metadata": {
        "module": "config",
        "name": "DocumentPreprocessorConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 14,
      "column": 0,
      "pattern": "from enhanced_pipeline import EnhancedDocumentPreprocessor",
      "context": "Imports EnhancedDocumentPreprocessor from enhanced_pipeline",
      "category": "from_import",
      "code_snippet": "from .config import DocumentPreprocessorConfig\nfrom .enhanced_pipeline import (\n    EnhancedDocumentPreprocessor,",
      "metadata": {
        "module": "enhanced_pipeline",
        "name": "EnhancedDocumentPreprocessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 14,
      "column": 0,
      "pattern": "from enhanced_pipeline import create_fast_preprocessor",
      "context": "Imports create_fast_preprocessor from enhanced_pipeline",
      "category": "from_import",
      "code_snippet": "from .config import DocumentPreprocessorConfig\nfrom .enhanced_pipeline import (\n    EnhancedDocumentPreprocessor,",
      "metadata": {
        "module": "enhanced_pipeline",
        "name": "create_fast_preprocessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 14,
      "column": 0,
      "pattern": "from enhanced_pipeline import create_office_lens_preprocessor",
      "context": "Imports create_office_lens_preprocessor from enhanced_pipeline",
      "category": "from_import",
      "code_snippet": "from .config import DocumentPreprocessorConfig\nfrom .enhanced_pipeline import (\n    EnhancedDocumentPreprocessor,",
      "metadata": {
        "module": "enhanced_pipeline",
        "name": "create_office_lens_preprocessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 19,
      "column": 0,
      "pattern": "from external import ALBUMENTATIONS_AVAILABLE",
      "context": "Imports ALBUMENTATIONS_AVAILABLE from external",
      "category": "from_import",
      "code_snippet": ")\nfrom .external import (\n    ALBUMENTATIONS_AVAILABLE,",
      "metadata": {
        "module": "external",
        "name": "ALBUMENTATIONS_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 19,
      "column": 0,
      "pattern": "from external import DOCTR_AVAILABLE",
      "context": "Imports DOCTR_AVAILABLE from external",
      "category": "from_import",
      "code_snippet": ")\nfrom .external import (\n    ALBUMENTATIONS_AVAILABLE,",
      "metadata": {
        "module": "external",
        "name": "DOCTR_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 19,
      "column": 0,
      "pattern": "from external import A",
      "context": "Imports A from external",
      "category": "from_import",
      "code_snippet": ")\nfrom .external import (\n    ALBUMENTATIONS_AVAILABLE,",
      "metadata": {
        "module": "external",
        "name": "A",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 19,
      "column": 0,
      "pattern": "from external import doctr_remove_image_padding",
      "context": "Imports doctr_remove_image_padding from external",
      "category": "from_import",
      "code_snippet": ")\nfrom .external import (\n    ALBUMENTATIONS_AVAILABLE,",
      "metadata": {
        "module": "external",
        "name": "doctr_remove_image_padding",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 19,
      "column": 0,
      "pattern": "from external import doctr_rotate_image",
      "context": "Imports doctr_rotate_image from external",
      "category": "from_import",
      "code_snippet": ")\nfrom .external import (\n    ALBUMENTATIONS_AVAILABLE,",
      "metadata": {
        "module": "external",
        "name": "doctr_rotate_image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 19,
      "column": 0,
      "pattern": "from external import estimate_page_angle",
      "context": "Imports estimate_page_angle from external",
      "category": "from_import",
      "code_snippet": ")\nfrom .external import (\n    ALBUMENTATIONS_AVAILABLE,",
      "metadata": {
        "module": "external",
        "name": "estimate_page_angle",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 19,
      "column": 0,
      "pattern": "from external import extract_rcrops",
      "context": "Imports extract_rcrops from external",
      "category": "from_import",
      "code_snippet": ")\nfrom .external import (\n    ALBUMENTATIONS_AVAILABLE,",
      "metadata": {
        "module": "external",
        "name": "extract_rcrops",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 28,
      "column": 0,
      "pattern": "from metadata import DocumentMetadata",
      "context": "Imports DocumentMetadata from metadata",
      "category": "from_import",
      "code_snippet": ")\nfrom .metadata import DocumentMetadata, PreprocessingState\nfrom .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations",
      "metadata": {
        "module": "metadata",
        "name": "DocumentMetadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 28,
      "column": 0,
      "pattern": "from metadata import PreprocessingState",
      "context": "Imports PreprocessingState from metadata",
      "category": "from_import",
      "code_snippet": ")\nfrom .metadata import DocumentMetadata, PreprocessingState\nfrom .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations",
      "metadata": {
        "module": "metadata",
        "name": "PreprocessingState",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 29,
      "column": 0,
      "pattern": "from pipeline import DocumentPreprocessor",
      "context": "Imports DocumentPreprocessor from pipeline",
      "category": "from_import",
      "code_snippet": "from .metadata import DocumentMetadata, PreprocessingState\nfrom .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations\nfrom .validators import (",
      "metadata": {
        "module": "pipeline",
        "name": "DocumentPreprocessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 29,
      "column": 0,
      "pattern": "from pipeline import LensStylePreprocessorAlbumentations",
      "context": "Imports LensStylePreprocessorAlbumentations from pipeline",
      "category": "from_import",
      "code_snippet": "from .metadata import DocumentMetadata, PreprocessingState\nfrom .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations\nfrom .validators import (",
      "metadata": {
        "module": "pipeline",
        "name": "LensStylePreprocessorAlbumentations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 30,
      "column": 0,
      "pattern": "from validators import ContractValidator",
      "context": "Imports ContractValidator from validators",
      "category": "from_import",
      "code_snippet": "from .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations\nfrom .validators import (\n    ContractValidator,",
      "metadata": {
        "module": "validators",
        "name": "ContractValidator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 30,
      "column": 0,
      "pattern": "from validators import CornerArray",
      "context": "Imports CornerArray from validators",
      "category": "from_import",
      "code_snippet": "from .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations\nfrom .validators import (\n    ContractValidator,",
      "metadata": {
        "module": "validators",
        "name": "CornerArray",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 30,
      "column": 0,
      "pattern": "from validators import ImageArray",
      "context": "Imports ImageArray from validators",
      "category": "from_import",
      "code_snippet": "from .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations\nfrom .validators import (\n    ContractValidator,",
      "metadata": {
        "module": "validators",
        "name": "ImageArray",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 30,
      "column": 0,
      "pattern": "from validators import ImageValidator",
      "context": "Imports ImageValidator from validators",
      "category": "from_import",
      "code_snippet": "from .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations\nfrom .validators import (\n    ContractValidator,",
      "metadata": {
        "module": "validators",
        "name": "ImageValidator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 30,
      "column": 0,
      "pattern": "from validators import NumpyArray",
      "context": "Imports NumpyArray from validators",
      "category": "from_import",
      "code_snippet": "from .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations\nfrom .validators import (\n    ContractValidator,",
      "metadata": {
        "module": "validators",
        "name": "NumpyArray",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/__init__.py",
      "line": 30,
      "column": 0,
      "pattern": "from validators import TransformationMatrix",
      "context": "Imports TransformationMatrix from validators",
      "category": "from_import",
      "code_snippet": "from .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations\nfrom .validators import (\n    ContractValidator,",
      "metadata": {
        "module": "validators",
        "name": "TransformationMatrix",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_detector.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_detector.py",
      "line": 9,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_detector.py",
      "line": 10,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\nfrom scipy.spatial import distance as dist",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_detector.py",
      "line": 11,
      "column": 0,
      "pattern": "from scipy.spatial import distance",
      "context": "Imports distance from scipy.spatial",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom scipy.spatial import distance as dist\n",
      "metadata": {
        "module": "scipy.spatial",
        "name": "distance",
        "alias": "dist"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_noise_elimination.py",
      "line": 12,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_noise_elimination.py",
      "line": 13,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_noise_elimination.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_noise_elimination.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_noise_elimination.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_noise_elimination.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import field_validator",
      "context": "Imports field_validator from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "field_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 10,
      "column": 0,
      "pattern": "from advanced_detector import AdvancedDetectionConfig",
      "context": "Imports AdvancedDetectionConfig from advanced_detector",
      "category": "from_import",
      "code_snippet": "\nfrom .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .config import DocumentPreprocessorConfig, EnhancementMethod",
      "metadata": {
        "module": "advanced_detector",
        "name": "AdvancedDetectionConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 10,
      "column": 0,
      "pattern": "from advanced_detector import AdvancedDocumentDetector",
      "context": "Imports AdvancedDocumentDetector from advanced_detector",
      "category": "from_import",
      "code_snippet": "\nfrom .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .config import DocumentPreprocessorConfig, EnhancementMethod",
      "metadata": {
        "module": "advanced_detector",
        "name": "AdvancedDocumentDetector",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 11,
      "column": 0,
      "pattern": "from config import DocumentPreprocessorConfig",
      "context": "Imports DocumentPreprocessorConfig from config",
      "category": "from_import",
      "code_snippet": "from .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .config import DocumentPreprocessorConfig, EnhancementMethod\nfrom .enhancement import ImageEnhancer",
      "metadata": {
        "module": "config",
        "name": "DocumentPreprocessorConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 11,
      "column": 0,
      "pattern": "from config import EnhancementMethod",
      "context": "Imports EnhancementMethod from config",
      "category": "from_import",
      "code_snippet": "from .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector\nfrom .config import DocumentPreprocessorConfig, EnhancementMethod\nfrom .enhancement import ImageEnhancer",
      "metadata": {
        "module": "config",
        "name": "EnhancementMethod",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 12,
      "column": 0,
      "pattern": "from enhancement import ImageEnhancer",
      "context": "Imports ImageEnhancer from enhancement",
      "category": "from_import",
      "code_snippet": "from .config import DocumentPreprocessorConfig, EnhancementMethod\nfrom .enhancement import ImageEnhancer\nfrom .metadata import DocumentMetadata, PreprocessingState",
      "metadata": {
        "module": "enhancement",
        "name": "ImageEnhancer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 13,
      "column": 0,
      "pattern": "from metadata import DocumentMetadata",
      "context": "Imports DocumentMetadata from metadata",
      "category": "from_import",
      "code_snippet": "from .enhancement import ImageEnhancer\nfrom .metadata import DocumentMetadata, PreprocessingState\nfrom .padding import PaddingCleanup",
      "metadata": {
        "module": "metadata",
        "name": "DocumentMetadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 13,
      "column": 0,
      "pattern": "from metadata import PreprocessingState",
      "context": "Imports PreprocessingState from metadata",
      "category": "from_import",
      "code_snippet": "from .enhancement import ImageEnhancer\nfrom .metadata import DocumentMetadata, PreprocessingState\nfrom .padding import PaddingCleanup",
      "metadata": {
        "module": "metadata",
        "name": "PreprocessingState",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 14,
      "column": 0,
      "pattern": "from padding import PaddingCleanup",
      "context": "Imports PaddingCleanup from padding",
      "category": "from_import",
      "code_snippet": "from .metadata import DocumentMetadata, PreprocessingState\nfrom .padding import PaddingCleanup\nfrom .perspective import PerspectiveCorrector",
      "metadata": {
        "module": "padding",
        "name": "PaddingCleanup",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 15,
      "column": 0,
      "pattern": "from perspective import PerspectiveCorrector",
      "context": "Imports PerspectiveCorrector from perspective",
      "category": "from_import",
      "code_snippet": "from .padding import PaddingCleanup\nfrom .perspective import PerspectiveCorrector\nfrom .resize import FinalResizer",
      "metadata": {
        "module": "perspective",
        "name": "PerspectiveCorrector",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/advanced_preprocessor.py",
      "line": 16,
      "column": 0,
      "pattern": "from resize import FinalResizer",
      "context": "Imports FinalResizer from resize",
      "category": "from_import",
      "code_snippet": "from .perspective import PerspectiveCorrector\nfrom .resize import FinalResizer\n",
      "metadata": {
        "module": "resize",
        "name": "FinalResizer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/background_removal.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/background_removal.py",
      "line": 9,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom albumentations import ImageOnlyTransform",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/background_removal.py",
      "line": 10,
      "column": 0,
      "pattern": "from albumentations import ImageOnlyTransform",
      "context": "Imports ImageOnlyTransform from albumentations",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom albumentations import ImageOnlyTransform\n",
      "metadata": {
        "module": "albumentations",
        "name": "ImageOnlyTransform",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/background_removal.py",
      "line": 13,
      "column": 4,
      "pattern": "from rembg import remove",
      "context": "Imports remove from rembg",
      "category": "from_import",
      "code_snippet": "try:\n    from rembg import remove\n",
      "metadata": {
        "module": "rembg",
        "name": "remove",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/config.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/config.py",
      "line": 7,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, Field, field_validator, model_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/config.py",
      "line": 7,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, Field, field_validator, model_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/config.py",
      "line": 7,
      "column": 0,
      "pattern": "from pydantic import field_validator",
      "context": "Imports field_validator from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, Field, field_validator, model_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "field_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/config.py",
      "line": 7,
      "column": 0,
      "pattern": "from pydantic import model_validator",
      "context": "Imports model_validator from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, Field, field_validator, model_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "model_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/config.py",
      "line": 12,
      "column": 4,
      "pattern": "from omegaconf import ListConfig",
      "context": "Imports ListConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "try:\n    from omegaconf import ListConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "ListConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/contracts.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/contracts.py",
      "line": 7,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom pydantic import BaseModel, Field, field_validator",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/contracts.py",
      "line": 8,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/contracts.py",
      "line": 8,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/contracts.py",
      "line": 8,
      "column": 0,
      "pattern": "from pydantic import field_validator",
      "context": "Imports field_validator from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "field_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/contracts.py",
      "line": 10,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/detector.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/detector.py",
      "line": 10,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/detector.py",
      "line": 11,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\ntry:",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/detector.py",
      "line": 13,
      "column": 4,
      "pattern": "from pylsd.lsd import lsd",
      "context": "Imports lsd from pylsd.lsd",
      "category": "from_import",
      "code_snippet": "try:\n    from pylsd.lsd import lsd\nexcept ImportError:",
      "metadata": {
        "module": "pylsd.lsd",
        "name": "lsd",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/detector.py",
      "line": 17,
      "column": 0,
      "pattern": "from contracts import ContractEnforcer",
      "context": "Imports ContractEnforcer from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import ContractEnforcer\nfrom .external import DOCTR_AVAILABLE",
      "metadata": {
        "module": "contracts",
        "name": "ContractEnforcer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/detector.py",
      "line": 18,
      "column": 0,
      "pattern": "from external import DOCTR_AVAILABLE",
      "context": "Imports DOCTR_AVAILABLE from external",
      "category": "from_import",
      "code_snippet": "from .contracts import ContractEnforcer\nfrom .external import DOCTR_AVAILABLE\n",
      "metadata": {
        "module": "external",
        "name": "DOCTR_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/detector.py",
      "line": 109,
      "column": 12,
      "pattern": "from doctr.models import zoo",
      "context": "Imports zoo from doctr.models",
      "category": "from_import",
      "code_snippet": "        try:\n            from doctr.models import zoo\n        except ImportError:",
      "metadata": {
        "module": "doctr.models",
        "name": "zoo",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/detector.py",
      "line": 437,
      "column": 8,
      "pattern": "from scipy.spatial import distance",
      "context": "Imports distance from scipy.spatial",
      "category": "from_import",
      "code_snippet": "        # our bottom-right point\n        from scipy.spatial import distance as dist\n",
      "metadata": {
        "module": "scipy.spatial",
        "name": "distance",
        "alias": "dist"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/document_flattening.py",
      "line": 12,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/document_flattening.py",
      "line": 13,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/document_flattening.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\nfrom scipy.interpolate import Rbf",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/document_flattening.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\nfrom scipy.interpolate import Rbf",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/document_flattening.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\nfrom scipy.interpolate import Rbf",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/document_flattening.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import field_validator",
      "context": "Imports field_validator from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\nfrom scipy.interpolate import Rbf",
      "metadata": {
        "module": "pydantic",
        "name": "field_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/document_flattening.py",
      "line": 15,
      "column": 0,
      "pattern": "from scipy.interpolate import Rbf",
      "context": "Imports Rbf from scipy.interpolate",
      "category": "from_import",
      "code_snippet": "from pydantic import BaseModel, ConfigDict, Field, field_validator\nfrom scipy.interpolate import Rbf\nfrom scipy.ndimage import gaussian_filter",
      "metadata": {
        "module": "scipy.interpolate",
        "name": "Rbf",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/document_flattening.py",
      "line": 16,
      "column": 0,
      "pattern": "from scipy.ndimage import gaussian_filter",
      "context": "Imports gaussian_filter from scipy.ndimage",
      "category": "from_import",
      "code_snippet": "from scipy.interpolate import Rbf\nfrom scipy.ndimage import gaussian_filter\n",
      "metadata": {
        "module": "scipy.ndimage",
        "name": "gaussian_filter",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 15,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 22,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 23,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 23,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 23,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 25,
      "column": 0,
      "pattern": "from advanced_noise_elimination import AdvancedNoiseEliminator",
      "context": "Imports AdvancedNoiseEliminator from advanced_noise_elimination",
      "category": "from_import",
      "code_snippet": "\nfrom .advanced_noise_elimination import AdvancedNoiseEliminator, NoiseEliminationConfig, NoiseReductionMethod\nfrom .config import DocumentPreprocessorConfig, EnhancementMethod",
      "metadata": {
        "module": "advanced_noise_elimination",
        "name": "AdvancedNoiseEliminator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 25,
      "column": 0,
      "pattern": "from advanced_noise_elimination import NoiseEliminationConfig",
      "context": "Imports NoiseEliminationConfig from advanced_noise_elimination",
      "category": "from_import",
      "code_snippet": "\nfrom .advanced_noise_elimination import AdvancedNoiseEliminator, NoiseEliminationConfig, NoiseReductionMethod\nfrom .config import DocumentPreprocessorConfig, EnhancementMethod",
      "metadata": {
        "module": "advanced_noise_elimination",
        "name": "NoiseEliminationConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 25,
      "column": 0,
      "pattern": "from advanced_noise_elimination import NoiseReductionMethod",
      "context": "Imports NoiseReductionMethod from advanced_noise_elimination",
      "category": "from_import",
      "code_snippet": "\nfrom .advanced_noise_elimination import AdvancedNoiseEliminator, NoiseEliminationConfig, NoiseReductionMethod\nfrom .config import DocumentPreprocessorConfig, EnhancementMethod",
      "metadata": {
        "module": "advanced_noise_elimination",
        "name": "NoiseReductionMethod",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 26,
      "column": 0,
      "pattern": "from config import DocumentPreprocessorConfig",
      "context": "Imports DocumentPreprocessorConfig from config",
      "category": "from_import",
      "code_snippet": "from .advanced_noise_elimination import AdvancedNoiseEliminator, NoiseEliminationConfig, NoiseReductionMethod\nfrom .config import DocumentPreprocessorConfig, EnhancementMethod\nfrom .contracts import validate_image_input_with_fallback",
      "metadata": {
        "module": "config",
        "name": "DocumentPreprocessorConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 26,
      "column": 0,
      "pattern": "from config import EnhancementMethod",
      "context": "Imports EnhancementMethod from config",
      "category": "from_import",
      "code_snippet": "from .advanced_noise_elimination import AdvancedNoiseEliminator, NoiseEliminationConfig, NoiseReductionMethod\nfrom .config import DocumentPreprocessorConfig, EnhancementMethod\nfrom .contracts import validate_image_input_with_fallback",
      "metadata": {
        "module": "config",
        "name": "EnhancementMethod",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 27,
      "column": 0,
      "pattern": "from contracts import validate_image_input_with_fallback",
      "context": "Imports validate_image_input_with_fallback from contracts",
      "category": "from_import",
      "code_snippet": "from .config import DocumentPreprocessorConfig, EnhancementMethod\nfrom .contracts import validate_image_input_with_fallback\nfrom .document_flattening import DocumentFlattener, FlatteningConfig, FlatteningMethod",
      "metadata": {
        "module": "contracts",
        "name": "validate_image_input_with_fallback",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 28,
      "column": 0,
      "pattern": "from document_flattening import DocumentFlattener",
      "context": "Imports DocumentFlattener from document_flattening",
      "category": "from_import",
      "code_snippet": "from .contracts import validate_image_input_with_fallback\nfrom .document_flattening import DocumentFlattener, FlatteningConfig, FlatteningMethod\nfrom .intelligent_brightness import BrightnessConfig, BrightnessMethod, IntelligentBrightnessAdjuster",
      "metadata": {
        "module": "document_flattening",
        "name": "DocumentFlattener",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 28,
      "column": 0,
      "pattern": "from document_flattening import FlatteningConfig",
      "context": "Imports FlatteningConfig from document_flattening",
      "category": "from_import",
      "code_snippet": "from .contracts import validate_image_input_with_fallback\nfrom .document_flattening import DocumentFlattener, FlatteningConfig, FlatteningMethod\nfrom .intelligent_brightness import BrightnessConfig, BrightnessMethod, IntelligentBrightnessAdjuster",
      "metadata": {
        "module": "document_flattening",
        "name": "FlatteningConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 28,
      "column": 0,
      "pattern": "from document_flattening import FlatteningMethod",
      "context": "Imports FlatteningMethod from document_flattening",
      "category": "from_import",
      "code_snippet": "from .contracts import validate_image_input_with_fallback\nfrom .document_flattening import DocumentFlattener, FlatteningConfig, FlatteningMethod\nfrom .intelligent_brightness import BrightnessConfig, BrightnessMethod, IntelligentBrightnessAdjuster",
      "metadata": {
        "module": "document_flattening",
        "name": "FlatteningMethod",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 29,
      "column": 0,
      "pattern": "from intelligent_brightness import BrightnessConfig",
      "context": "Imports BrightnessConfig from intelligent_brightness",
      "category": "from_import",
      "code_snippet": "from .document_flattening import DocumentFlattener, FlatteningConfig, FlatteningMethod\nfrom .intelligent_brightness import BrightnessConfig, BrightnessMethod, IntelligentBrightnessAdjuster\nfrom .pipeline import DocumentPreprocessor",
      "metadata": {
        "module": "intelligent_brightness",
        "name": "BrightnessConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 29,
      "column": 0,
      "pattern": "from intelligent_brightness import BrightnessMethod",
      "context": "Imports BrightnessMethod from intelligent_brightness",
      "category": "from_import",
      "code_snippet": "from .document_flattening import DocumentFlattener, FlatteningConfig, FlatteningMethod\nfrom .intelligent_brightness import BrightnessConfig, BrightnessMethod, IntelligentBrightnessAdjuster\nfrom .pipeline import DocumentPreprocessor",
      "metadata": {
        "module": "intelligent_brightness",
        "name": "BrightnessMethod",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 29,
      "column": 0,
      "pattern": "from intelligent_brightness import IntelligentBrightnessAdjuster",
      "context": "Imports IntelligentBrightnessAdjuster from intelligent_brightness",
      "category": "from_import",
      "code_snippet": "from .document_flattening import DocumentFlattener, FlatteningConfig, FlatteningMethod\nfrom .intelligent_brightness import BrightnessConfig, BrightnessMethod, IntelligentBrightnessAdjuster\nfrom .pipeline import DocumentPreprocessor",
      "metadata": {
        "module": "intelligent_brightness",
        "name": "IntelligentBrightnessAdjuster",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhanced_pipeline.py",
      "line": 30,
      "column": 0,
      "pattern": "from pipeline import DocumentPreprocessor",
      "context": "Imports DocumentPreprocessor from pipeline",
      "category": "from_import",
      "code_snippet": "from .intelligent_brightness import BrightnessConfig, BrightnessMethod, IntelligentBrightnessAdjuster\nfrom .pipeline import DocumentPreprocessor\n",
      "metadata": {
        "module": "pipeline",
        "name": "DocumentPreprocessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhancement.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhancement.py",
      "line": 5,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/enhancement.py",
      "line": 6,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 27,
      "column": 4,
      "pattern": "import albumentations",
      "context": "Imports module: albumentations",
      "category": "import",
      "code_snippet": "try:  # pragma: no cover - optional dependency guard\n    import albumentations as _albumentations\n    from albumentations.core.transforms_interface import ImageOnlyTransform as _ImageOnlyTransform",
      "metadata": {
        "module": "albumentations",
        "alias": "_albumentations"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 28,
      "column": 4,
      "pattern": "from albumentations.core.transforms_interface import ImageOnlyTransform",
      "context": "Imports ImageOnlyTransform from albumentations.core.transforms_interface",
      "category": "from_import",
      "code_snippet": "    import albumentations as _albumentations\n    from albumentations.core.transforms_interface import ImageOnlyTransform as _ImageOnlyTransform\n",
      "metadata": {
        "module": "albumentations.core.transforms_interface",
        "name": "ImageOnlyTransform",
        "alias": "_ImageOnlyTransform"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 39,
      "column": 4,
      "pattern": "from doctr.utils.geometry import estimate_page_angle",
      "context": "Imports estimate_page_angle from doctr.utils.geometry",
      "category": "from_import",
      "code_snippet": "try:  # pragma: no cover - optional dependency guard\n    from doctr.utils.geometry import estimate_page_angle as _estimate_page_angle\n    from doctr.utils.geometry import extract_rcrops as _extract_rcrops",
      "metadata": {
        "module": "doctr.utils.geometry",
        "name": "estimate_page_angle",
        "alias": "_estimate_page_angle"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 40,
      "column": 4,
      "pattern": "from doctr.utils.geometry import extract_rcrops",
      "context": "Imports extract_rcrops from doctr.utils.geometry",
      "category": "from_import",
      "code_snippet": "    from doctr.utils.geometry import estimate_page_angle as _estimate_page_angle\n    from doctr.utils.geometry import extract_rcrops as _extract_rcrops\n    from doctr.utils.geometry import remove_image_padding as _doctr_remove_image_padding",
      "metadata": {
        "module": "doctr.utils.geometry",
        "name": "extract_rcrops",
        "alias": "_extract_rcrops"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 41,
      "column": 4,
      "pattern": "from doctr.utils.geometry import remove_image_padding",
      "context": "Imports remove_image_padding from doctr.utils.geometry",
      "category": "from_import",
      "code_snippet": "    from doctr.utils.geometry import extract_rcrops as _extract_rcrops\n    from doctr.utils.geometry import remove_image_padding as _doctr_remove_image_padding\n    from doctr.utils.geometry import rotate_image as _doctr_rotate_image",
      "metadata": {
        "module": "doctr.utils.geometry",
        "name": "remove_image_padding",
        "alias": "_doctr_remove_image_padding"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 42,
      "column": 4,
      "pattern": "from doctr.utils.geometry import rotate_image",
      "context": "Imports rotate_image from doctr.utils.geometry",
      "category": "from_import",
      "code_snippet": "    from doctr.utils.geometry import remove_image_padding as _doctr_remove_image_padding\n    from doctr.utils.geometry import rotate_image as _doctr_rotate_image\n",
      "metadata": {
        "module": "doctr.utils.geometry",
        "name": "rotate_image",
        "alias": "_doctr_rotate_image"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 58,
      "column": 4,
      "pattern": "from background_removal import BackgroundRemoval",
      "context": "Imports BackgroundRemoval from background_removal",
      "category": "from_import",
      "code_snippet": "try:  # pragma: no cover - optional dependency guard\n    from .background_removal import BackgroundRemoval as _BackgroundRemoval\n    from .background_removal import create_background_removal_transform as _create_background_removal_transform",
      "metadata": {
        "module": "background_removal",
        "name": "BackgroundRemoval",
        "alias": "_BackgroundRemoval"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/external.py",
      "line": 59,
      "column": 4,
      "pattern": "from background_removal import create_background_removal_transform",
      "context": "Imports create_background_removal_transform from background_removal",
      "category": "from_import",
      "code_snippet": "    from .background_removal import BackgroundRemoval as _BackgroundRemoval\n    from .background_removal import create_background_removal_transform as _create_background_removal_transform\n",
      "metadata": {
        "module": "background_removal",
        "name": "create_background_removal_transform",
        "alias": "_create_background_removal_transform"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/intelligent_brightness.py",
      "line": 18,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/intelligent_brightness.py",
      "line": 19,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/intelligent_brightness.py",
      "line": 20,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/intelligent_brightness.py",
      "line": 20,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/intelligent_brightness.py",
      "line": 20,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/intelligent_brightness.py",
      "line": 20,
      "column": 0,
      "pattern": "from pydantic import field_validator",
      "context": "Imports field_validator from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "field_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/metadata.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/metadata.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/metadata.py",
      "line": 9,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/metadata.py",
      "line": 9,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/metadata.py",
      "line": 9,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/metadata.py",
      "line": 9,
      "column": 0,
      "pattern": "from pydantic import field_validator",
      "context": "Imports field_validator from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "field_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/orientation.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/orientation.py",
      "line": 9,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/orientation.py",
      "line": 11,
      "column": 0,
      "pattern": "from detector import DocumentDetector",
      "context": "Imports DocumentDetector from detector",
      "category": "from_import",
      "code_snippet": "\nfrom .detector import DocumentDetector\nfrom .external import doctr_rotate_image, estimate_page_angle",
      "metadata": {
        "module": "detector",
        "name": "DocumentDetector",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/orientation.py",
      "line": 12,
      "column": 0,
      "pattern": "from external import doctr_rotate_image",
      "context": "Imports doctr_rotate_image from external",
      "category": "from_import",
      "code_snippet": "from .detector import DocumentDetector\nfrom .external import doctr_rotate_image, estimate_page_angle\n",
      "metadata": {
        "module": "external",
        "name": "doctr_rotate_image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/orientation.py",
      "line": 12,
      "column": 0,
      "pattern": "from external import estimate_page_angle",
      "context": "Imports estimate_page_angle from external",
      "category": "from_import",
      "code_snippet": "from .detector import DocumentDetector\nfrom .external import doctr_rotate_image, estimate_page_angle\n",
      "metadata": {
        "module": "external",
        "name": "estimate_page_angle",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/padding.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/padding.py",
      "line": 7,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/padding.py",
      "line": 9,
      "column": 0,
      "pattern": "from external import doctr_remove_image_padding",
      "context": "Imports doctr_remove_image_padding from external",
      "category": "from_import",
      "code_snippet": "\nfrom .external import doctr_remove_image_padding\n",
      "metadata": {
        "module": "external",
        "name": "doctr_remove_image_padding",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/perspective.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/perspective.py",
      "line": 8,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/perspective.py",
      "line": 9,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/perspective.py",
      "line": 11,
      "column": 0,
      "pattern": "from external import extract_rcrops",
      "context": "Imports extract_rcrops from external",
      "category": "from_import",
      "code_snippet": "\nfrom .external import extract_rcrops\n",
      "metadata": {
        "module": "external",
        "name": "extract_rcrops",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 10,
      "column": 0,
      "pattern": "from config import DocumentPreprocessorConfig",
      "context": "Imports DocumentPreprocessorConfig from config",
      "category": "from_import",
      "code_snippet": "\nfrom .config import DocumentPreprocessorConfig\nfrom .contracts import validate_image_input_with_fallback, validate_preprocessing_result_with_fallback",
      "metadata": {
        "module": "config",
        "name": "DocumentPreprocessorConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 11,
      "column": 0,
      "pattern": "from contracts import validate_image_input_with_fallback",
      "context": "Imports validate_image_input_with_fallback from contracts",
      "category": "from_import",
      "code_snippet": "from .config import DocumentPreprocessorConfig\nfrom .contracts import validate_image_input_with_fallback, validate_preprocessing_result_with_fallback\nfrom .detector import DocumentDetector",
      "metadata": {
        "module": "contracts",
        "name": "validate_image_input_with_fallback",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 11,
      "column": 0,
      "pattern": "from contracts import validate_preprocessing_result_with_fallback",
      "context": "Imports validate_preprocessing_result_with_fallback from contracts",
      "category": "from_import",
      "code_snippet": "from .config import DocumentPreprocessorConfig\nfrom .contracts import validate_image_input_with_fallback, validate_preprocessing_result_with_fallback\nfrom .detector import DocumentDetector",
      "metadata": {
        "module": "contracts",
        "name": "validate_preprocessing_result_with_fallback",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 12,
      "column": 0,
      "pattern": "from detector import DocumentDetector",
      "context": "Imports DocumentDetector from detector",
      "category": "from_import",
      "code_snippet": "from .contracts import validate_image_input_with_fallback, validate_preprocessing_result_with_fallback\nfrom .detector import DocumentDetector\nfrom .enhancement import ImageEnhancer",
      "metadata": {
        "module": "detector",
        "name": "DocumentDetector",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 13,
      "column": 0,
      "pattern": "from enhancement import ImageEnhancer",
      "context": "Imports ImageEnhancer from enhancement",
      "category": "from_import",
      "code_snippet": "from .detector import DocumentDetector\nfrom .enhancement import ImageEnhancer\nfrom .external import ALBUMENTATIONS_AVAILABLE, DOCTR_AVAILABLE, A, ImageOnlyTransform",
      "metadata": {
        "module": "enhancement",
        "name": "ImageEnhancer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 14,
      "column": 0,
      "pattern": "from external import ALBUMENTATIONS_AVAILABLE",
      "context": "Imports ALBUMENTATIONS_AVAILABLE from external",
      "category": "from_import",
      "code_snippet": "from .enhancement import ImageEnhancer\nfrom .external import ALBUMENTATIONS_AVAILABLE, DOCTR_AVAILABLE, A, ImageOnlyTransform\nfrom .metadata import DocumentMetadata, PreprocessingState",
      "metadata": {
        "module": "external",
        "name": "ALBUMENTATIONS_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 14,
      "column": 0,
      "pattern": "from external import DOCTR_AVAILABLE",
      "context": "Imports DOCTR_AVAILABLE from external",
      "category": "from_import",
      "code_snippet": "from .enhancement import ImageEnhancer\nfrom .external import ALBUMENTATIONS_AVAILABLE, DOCTR_AVAILABLE, A, ImageOnlyTransform\nfrom .metadata import DocumentMetadata, PreprocessingState",
      "metadata": {
        "module": "external",
        "name": "DOCTR_AVAILABLE",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 14,
      "column": 0,
      "pattern": "from external import A",
      "context": "Imports A from external",
      "category": "from_import",
      "code_snippet": "from .enhancement import ImageEnhancer\nfrom .external import ALBUMENTATIONS_AVAILABLE, DOCTR_AVAILABLE, A, ImageOnlyTransform\nfrom .metadata import DocumentMetadata, PreprocessingState",
      "metadata": {
        "module": "external",
        "name": "A",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 14,
      "column": 0,
      "pattern": "from external import ImageOnlyTransform",
      "context": "Imports ImageOnlyTransform from external",
      "category": "from_import",
      "code_snippet": "from .enhancement import ImageEnhancer\nfrom .external import ALBUMENTATIONS_AVAILABLE, DOCTR_AVAILABLE, A, ImageOnlyTransform\nfrom .metadata import DocumentMetadata, PreprocessingState",
      "metadata": {
        "module": "external",
        "name": "ImageOnlyTransform",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 15,
      "column": 0,
      "pattern": "from metadata import DocumentMetadata",
      "context": "Imports DocumentMetadata from metadata",
      "category": "from_import",
      "code_snippet": "from .external import ALBUMENTATIONS_AVAILABLE, DOCTR_AVAILABLE, A, ImageOnlyTransform\nfrom .metadata import DocumentMetadata, PreprocessingState\nfrom .orientation import OrientationCorrector",
      "metadata": {
        "module": "metadata",
        "name": "DocumentMetadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 15,
      "column": 0,
      "pattern": "from metadata import PreprocessingState",
      "context": "Imports PreprocessingState from metadata",
      "category": "from_import",
      "code_snippet": "from .external import ALBUMENTATIONS_AVAILABLE, DOCTR_AVAILABLE, A, ImageOnlyTransform\nfrom .metadata import DocumentMetadata, PreprocessingState\nfrom .orientation import OrientationCorrector",
      "metadata": {
        "module": "metadata",
        "name": "PreprocessingState",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 16,
      "column": 0,
      "pattern": "from orientation import OrientationCorrector",
      "context": "Imports OrientationCorrector from orientation",
      "category": "from_import",
      "code_snippet": "from .metadata import DocumentMetadata, PreprocessingState\nfrom .orientation import OrientationCorrector\nfrom .padding import PaddingCleanup",
      "metadata": {
        "module": "orientation",
        "name": "OrientationCorrector",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 17,
      "column": 0,
      "pattern": "from padding import PaddingCleanup",
      "context": "Imports PaddingCleanup from padding",
      "category": "from_import",
      "code_snippet": "from .orientation import OrientationCorrector\nfrom .padding import PaddingCleanup\nfrom .perspective import PerspectiveCorrector",
      "metadata": {
        "module": "padding",
        "name": "PaddingCleanup",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 18,
      "column": 0,
      "pattern": "from perspective import PerspectiveCorrector",
      "context": "Imports PerspectiveCorrector from perspective",
      "category": "from_import",
      "code_snippet": "from .padding import PaddingCleanup\nfrom .perspective import PerspectiveCorrector\nfrom .resize import FinalResizer",
      "metadata": {
        "module": "perspective",
        "name": "PerspectiveCorrector",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/pipeline.py",
      "line": 19,
      "column": 0,
      "pattern": "from resize import FinalResizer",
      "context": "Imports FinalResizer from resize",
      "category": "from_import",
      "code_snippet": "from .perspective import PerspectiveCorrector\nfrom .resize import FinalResizer\n",
      "metadata": {
        "module": "resize",
        "name": "FinalResizer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/resize.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/resize.py",
      "line": 5,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/resize.py",
      "line": 6,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/telemetry.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/telemetry.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/telemetry.py",
      "line": 14,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/validators.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/validators.py",
      "line": 7,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom pydantic import GetCoreSchemaHandler",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/validators.py",
      "line": 8,
      "column": 0,
      "pattern": "from pydantic import GetCoreSchemaHandler",
      "context": "Imports GetCoreSchemaHandler from pydantic",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom pydantic import GetCoreSchemaHandler\nfrom pydantic_core import core_schema",
      "metadata": {
        "module": "pydantic",
        "name": "GetCoreSchemaHandler",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/validators.py",
      "line": 9,
      "column": 0,
      "pattern": "from pydantic_core import core_schema",
      "context": "Imports core_schema from pydantic_core",
      "category": "from_import",
      "code_snippet": "from pydantic import GetCoreSchemaHandler\nfrom pydantic_core import core_schema\n",
      "metadata": {
        "module": "pydantic_core",
        "name": "core_schema",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/preprocessing/validators.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/recognition_collate_fn.py",
      "line": 2,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\"\"\"Collate function for text recognition datasets.\"\"\"\nimport torch\nfrom typing import Any",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import CacheConfig",
      "context": "Imports CacheConfig from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "CacheConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import DataItem",
      "context": "Imports DataItem from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "DataItem",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import DatasetConfig",
      "context": "Imports DatasetConfig from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "DatasetConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import ImageData",
      "context": "Imports ImageData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ImageData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import ImageLoadingConfig",
      "context": "Imports ImageLoadingConfig from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ImageLoadingConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import ImageMetadata",
      "context": "Imports ImageMetadata from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ImageMetadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import MapData",
      "context": "Imports MapData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "MapData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import PolygonData",
      "context": "Imports PolygonData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "PolygonData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import TransformConfig",
      "context": "Imports TransformConfig from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "TransformConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import TransformInput",
      "context": "Imports TransformInput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "TransformInput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import TransformOutput",
      "context": "Imports TransformOutput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "TransformOutput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/schemas.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import ValidatedPolygonData",
      "context": "Imports ValidatedPolygonData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    CacheConfig,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ValidatedPolygonData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 62,
      "column": 0,
      "pattern": "import albumentations",
      "context": "Imports module: albumentations",
      "category": "import",
      "code_snippet": "\nimport albumentations as A\nimport numpy as np",
      "metadata": {
        "module": "albumentations",
        "alias": "A"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 63,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import albumentations as A\nimport numpy as np\nfrom albumentations.pytorch import ToTensorV2",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 64,
      "column": 0,
      "pattern": "from albumentations.pytorch import ToTensorV2",
      "context": "Imports ToTensorV2 from albumentations.pytorch",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom albumentations.pytorch import ToTensorV2\nfrom pydantic import ValidationError",
      "metadata": {
        "module": "albumentations.pytorch",
        "name": "ToTensorV2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 65,
      "column": 0,
      "pattern": "from pydantic import ValidationError",
      "context": "Imports ValidationError from pydantic",
      "category": "from_import",
      "code_snippet": "from albumentations.pytorch import ToTensorV2\nfrom pydantic import ValidationError\n",
      "metadata": {
        "module": "pydantic",
        "name": "ValidationError",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 67,
      "column": 0,
      "pattern": "from ocr.core.validation import ImageMetadata",
      "context": "Imports ImageMetadata from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import ImageMetadata, PolygonData, TransformInput, TransformOutput\nfrom ocr.core.utils.config_utils import is_config",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ImageMetadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 67,
      "column": 0,
      "pattern": "from ocr.core.validation import PolygonData",
      "context": "Imports PolygonData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import ImageMetadata, PolygonData, TransformInput, TransformOutput\nfrom ocr.core.utils.config_utils import is_config",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "PolygonData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 67,
      "column": 0,
      "pattern": "from ocr.core.validation import TransformInput",
      "context": "Imports TransformInput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import ImageMetadata, PolygonData, TransformInput, TransformOutput\nfrom ocr.core.utils.config_utils import is_config",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "TransformInput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 67,
      "column": 0,
      "pattern": "from ocr.core.validation import TransformOutput",
      "context": "Imports TransformOutput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.validation import ImageMetadata, PolygonData, TransformInput, TransformOutput\nfrom ocr.core.utils.config_utils import is_config",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "TransformOutput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 68,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "from ocr.core.validation import ImageMetadata, PolygonData, TransformInput, TransformOutput\nfrom ocr.core.utils.config_utils import is_config\nfrom ocr.core.utils.geometry_utils import calculate_cropbox, calculate_inverse_transform",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 69,
      "column": 0,
      "pattern": "from ocr.core.utils.geometry_utils import calculate_cropbox",
      "context": "Imports calculate_cropbox from ocr.core.utils.geometry_utils",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.config_utils import is_config\nfrom ocr.core.utils.geometry_utils import calculate_cropbox, calculate_inverse_transform\n",
      "metadata": {
        "module": "ocr.core.utils.geometry_utils",
        "name": "calculate_cropbox",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 69,
      "column": 0,
      "pattern": "from ocr.core.utils.geometry_utils import calculate_inverse_transform",
      "context": "Imports calculate_inverse_transform from ocr.core.utils.geometry_utils",
      "category": "from_import",
      "code_snippet": "from ocr.core.utils.config_utils import is_config\nfrom ocr.core.utils.geometry_utils import calculate_cropbox, calculate_inverse_transform\n",
      "metadata": {
        "module": "ocr.core.utils.geometry_utils",
        "name": "calculate_inverse_transform",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/datasets/transforms.py",
      "line": 345,
      "column": 8,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "        \"\"\"\n        from PIL import Image as PILImage\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": "PILImage"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/schemas/storage.py",
      "line": 12,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/schemas/storage.py",
      "line": 16,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/schemas/storage.py",
      "line": 16,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/schemas/storage.py",
      "line": 16,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/interfaces.py",
      "line": 6,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/interfaces.py",
      "line": 8,
      "column": 0,
      "pattern": "from ocr.core.interfaces.losses import BaseLoss",
      "context": "Imports BaseLoss from ocr.core.interfaces.losses",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.interfaces.losses import BaseLoss\nfrom ocr.core.interfaces.models import BaseHead",
      "metadata": {
        "module": "ocr.core.interfaces.losses",
        "name": "BaseLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/interfaces.py",
      "line": 9,
      "column": 0,
      "pattern": "from ocr.core.interfaces.models import BaseHead",
      "context": "Imports BaseHead from ocr.core.interfaces.models",
      "category": "from_import",
      "code_snippet": "from ocr.core.interfaces.losses import BaseLoss\nfrom ocr.core.interfaces.models import BaseHead\n",
      "metadata": {
        "module": "ocr.core.interfaces.models",
        "name": "BaseHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 7,
      "column": 8,
      "pattern": "from ocr.features.detection.models.architectures.craft import CRAFT",
      "context": "Imports CRAFT from ocr.features.detection.models.architectures.craft",
      "category": "from_import",
      "code_snippet": "    if name == \"CRAFT\":\n        from ocr.features.detection.models.architectures.craft import CRAFT\n        return CRAFT",
      "metadata": {
        "module": "ocr.features.detection.models.architectures.craft",
        "name": "CRAFT",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 10,
      "column": 8,
      "pattern": "from ocr.features.detection.models.architectures.dbnet import DBNet",
      "context": "Imports DBNet from ocr.features.detection.models.architectures.dbnet",
      "category": "from_import",
      "code_snippet": "    elif name == \"DBNet\":\n        from ocr.features.detection.models.architectures.dbnet import DBNet\n        return DBNet",
      "metadata": {
        "module": "ocr.features.detection.models.architectures.dbnet",
        "name": "DBNet",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 13,
      "column": 8,
      "pattern": "from ocr.features.detection.models.architectures.dbnetpp import DBNetPP",
      "context": "Imports DBNetPP from ocr.features.detection.models.architectures.dbnetpp",
      "category": "from_import",
      "code_snippet": "    elif name == \"DBNetPP\":\n        from ocr.features.detection.models.architectures.dbnetpp import DBNetPP\n        return DBNetPP",
      "metadata": {
        "module": "ocr.features.detection.models.architectures.dbnetpp",
        "name": "DBNetPP",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 17,
      "column": 8,
      "pattern": "from ocr.features.detection.models.heads.craft_head import CRAFTHead",
      "context": "Imports CRAFTHead from ocr.features.detection.models.heads.craft_head",
      "category": "from_import",
      "code_snippet": "    elif name == \"CRAFTHead\":\n        from ocr.features.detection.models.heads.craft_head import CRAFTHead\n        return CRAFTHead",
      "metadata": {
        "module": "ocr.features.detection.models.heads.craft_head",
        "name": "CRAFTHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 20,
      "column": 8,
      "pattern": "from ocr.features.detection.models.heads.db_head import DBHead",
      "context": "Imports DBHead from ocr.features.detection.models.heads.db_head",
      "category": "from_import",
      "code_snippet": "    elif name == \"DBHead\":\n        from ocr.features.detection.models.heads.db_head import DBHead\n        return DBHead",
      "metadata": {
        "module": "ocr.features.detection.models.heads.db_head",
        "name": "DBHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 24,
      "column": 8,
      "pattern": "from ocr.features.detection.models.postprocess.craft_postprocess import CRAFTPostProcessor",
      "context": "Imports CRAFTPostProcessor from ocr.features.detection.models.postprocess.craft_postprocess",
      "category": "from_import",
      "code_snippet": "    elif name == \"CRAFTPostProcessor\":\n        from ocr.features.detection.models.postprocess.craft_postprocess import CRAFTPostProcessor\n        return CRAFTPostProcessor",
      "metadata": {
        "module": "ocr.features.detection.models.postprocess.craft_postprocess",
        "name": "CRAFTPostProcessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 27,
      "column": 8,
      "pattern": "from ocr.features.detection.models.postprocess.db_postprocess import DBPostProcessor",
      "context": "Imports DBPostProcessor from ocr.features.detection.models.postprocess.db_postprocess",
      "category": "from_import",
      "code_snippet": "    elif name == \"DBPostProcessor\":\n        from ocr.features.detection.models.postprocess.db_postprocess import DBPostProcessor\n        return DBPostProcessor",
      "metadata": {
        "module": "ocr.features.detection.models.postprocess.db_postprocess",
        "name": "DBPostProcessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 31,
      "column": 8,
      "pattern": "from ocr.features.detection.models.decoders.craft_decoder import CRAFTDecoder",
      "context": "Imports CRAFTDecoder from ocr.features.detection.models.decoders.craft_decoder",
      "category": "from_import",
      "code_snippet": "    elif name == \"CRAFTDecoder\":\n        from ocr.features.detection.models.decoders.craft_decoder import CRAFTDecoder\n        return CRAFTDecoder",
      "metadata": {
        "module": "ocr.features.detection.models.decoders.craft_decoder",
        "name": "CRAFTDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 34,
      "column": 8,
      "pattern": "from ocr.features.detection.models.decoders.dbpp_decoder import DBPPDecoder",
      "context": "Imports DBPPDecoder from ocr.features.detection.models.decoders.dbpp_decoder",
      "category": "from_import",
      "code_snippet": "    elif name == \"DBPPDecoder\":\n        from ocr.features.detection.models.decoders.dbpp_decoder import DBPPDecoder\n        return DBPPDecoder",
      "metadata": {
        "module": "ocr.features.detection.models.decoders.dbpp_decoder",
        "name": "DBPPDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 37,
      "column": 8,
      "pattern": "from ocr.features.detection.models.decoders.fpn_decoder import FPNDecoder",
      "context": "Imports FPNDecoder from ocr.features.detection.models.decoders.fpn_decoder",
      "category": "from_import",
      "code_snippet": "    elif name == \"FPNDecoder\":\n        from ocr.features.detection.models.decoders.fpn_decoder import FPNDecoder\n        return FPNDecoder",
      "metadata": {
        "module": "ocr.features.detection.models.decoders.fpn_decoder",
        "name": "FPNDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/__init__.py",
      "line": 41,
      "column": 8,
      "pattern": "from ocr.features.detection.models.encoders.craft_vgg import CRAFTVGG",
      "context": "Imports CRAFTVGG from ocr.features.detection.models.encoders.craft_vgg",
      "category": "from_import",
      "code_snippet": "    elif name == \"CRAFTVGG\":\n        from ocr.features.detection.models.encoders.craft_vgg import CRAFTVGG\n        return CRAFTVGG",
      "metadata": {
        "module": "ocr.features.detection.models.encoders.craft_vgg",
        "name": "CRAFTVGG",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/craft.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/craft.py",
      "line": 5,
      "column": 0,
      "pattern": "from ocr.core import registry",
      "context": "Imports registry from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import registry\nfrom ocr.features.detection.models.decoders.craft_decoder import CraftDecoder",
      "metadata": {
        "module": "ocr.core",
        "name": "registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/craft.py",
      "line": 6,
      "column": 0,
      "pattern": "from ocr.features.detection.models.decoders.craft_decoder import CraftDecoder",
      "context": "Imports CraftDecoder from ocr.features.detection.models.decoders.craft_decoder",
      "category": "from_import",
      "code_snippet": "from ocr.core import registry\nfrom ocr.features.detection.models.decoders.craft_decoder import CraftDecoder\nfrom ocr.features.detection.models.encoders.craft_vgg import CraftVGGEncoder",
      "metadata": {
        "module": "ocr.features.detection.models.decoders.craft_decoder",
        "name": "CraftDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/craft.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.features.detection.models.encoders.craft_vgg import CraftVGGEncoder",
      "context": "Imports CraftVGGEncoder from ocr.features.detection.models.encoders.craft_vgg",
      "category": "from_import",
      "code_snippet": "from ocr.features.detection.models.decoders.craft_decoder import CraftDecoder\nfrom ocr.features.detection.models.encoders.craft_vgg import CraftVGGEncoder\nfrom ocr.features.detection.models.heads.craft_head import CraftHead",
      "metadata": {
        "module": "ocr.features.detection.models.encoders.craft_vgg",
        "name": "CraftVGGEncoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/craft.py",
      "line": 8,
      "column": 0,
      "pattern": "from ocr.features.detection.models.heads.craft_head import CraftHead",
      "context": "Imports CraftHead from ocr.features.detection.models.heads.craft_head",
      "category": "from_import",
      "code_snippet": "from ocr.features.detection.models.encoders.craft_vgg import CraftVGGEncoder\nfrom ocr.features.detection.models.heads.craft_head import CraftHead\nfrom ocr.core.models.loss.craft_loss import CraftLoss",
      "metadata": {
        "module": "ocr.features.detection.models.heads.craft_head",
        "name": "CraftHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/craft.py",
      "line": 9,
      "column": 0,
      "pattern": "from ocr.core.models.loss.craft_loss import CraftLoss",
      "context": "Imports CraftLoss from ocr.core.models.loss.craft_loss",
      "category": "from_import",
      "code_snippet": "from ocr.features.detection.models.heads.craft_head import CraftHead\nfrom ocr.core.models.loss.craft_loss import CraftLoss\n",
      "metadata": {
        "module": "ocr.core.models.loss.craft_loss",
        "name": "CraftLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnet.py",
      "line": 3,
      "column": 0,
      "pattern": "from ocr.core import registry",
      "context": "Imports registry from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import registry\nfrom ocr.core.models.decoder.unet import UNetDecoder",
      "metadata": {
        "module": "ocr.core",
        "name": "registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnet.py",
      "line": 4,
      "column": 0,
      "pattern": "from ocr.core.models.decoder.unet import UNetDecoder",
      "context": "Imports UNetDecoder from ocr.core.models.decoder.unet",
      "category": "from_import",
      "code_snippet": "from ocr.core import registry\nfrom ocr.core.models.decoder.unet import UNetDecoder\nfrom ocr.core.models.encoder.timm_backbone import TimmBackbone",
      "metadata": {
        "module": "ocr.core.models.decoder.unet",
        "name": "UNetDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnet.py",
      "line": 5,
      "column": 0,
      "pattern": "from ocr.core.models.encoder.timm_backbone import TimmBackbone",
      "context": "Imports TimmBackbone from ocr.core.models.encoder.timm_backbone",
      "category": "from_import",
      "code_snippet": "from ocr.core.models.decoder.unet import UNetDecoder\nfrom ocr.core.models.encoder.timm_backbone import TimmBackbone\nfrom ocr.features.detection.models.heads.db_head import DBHead",
      "metadata": {
        "module": "ocr.core.models.encoder.timm_backbone",
        "name": "TimmBackbone",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnet.py",
      "line": 6,
      "column": 0,
      "pattern": "from ocr.features.detection.models.heads.db_head import DBHead",
      "context": "Imports DBHead from ocr.features.detection.models.heads.db_head",
      "category": "from_import",
      "code_snippet": "from ocr.core.models.encoder.timm_backbone import TimmBackbone\nfrom ocr.features.detection.models.heads.db_head import DBHead\nfrom ocr.core.models.loss.db_loss import DBLoss",
      "metadata": {
        "module": "ocr.features.detection.models.heads.db_head",
        "name": "DBHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnet.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.core.models.loss.db_loss import DBLoss",
      "context": "Imports DBLoss from ocr.core.models.loss.db_loss",
      "category": "from_import",
      "code_snippet": "from ocr.features.detection.models.heads.db_head import DBHead\nfrom ocr.core.models.loss.db_loss import DBLoss\n",
      "metadata": {
        "module": "ocr.core.models.loss.db_loss",
        "name": "DBLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnetpp.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnetpp.py",
      "line": 5,
      "column": 0,
      "pattern": "from ocr.core import registry",
      "context": "Imports registry from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import registry\nfrom ocr.features.detection.models.decoders.dbpp_decoder import DBPPDecoder",
      "metadata": {
        "module": "ocr.core",
        "name": "registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnetpp.py",
      "line": 6,
      "column": 0,
      "pattern": "from ocr.features.detection.models.decoders.dbpp_decoder import DBPPDecoder",
      "context": "Imports DBPPDecoder from ocr.features.detection.models.decoders.dbpp_decoder",
      "category": "from_import",
      "code_snippet": "from ocr.core import registry\nfrom ocr.features.detection.models.decoders.dbpp_decoder import DBPPDecoder\nfrom ocr.core.models.encoder.timm_backbone import TimmBackbone",
      "metadata": {
        "module": "ocr.features.detection.models.decoders.dbpp_decoder",
        "name": "DBPPDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnetpp.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.core.models.encoder.timm_backbone import TimmBackbone",
      "context": "Imports TimmBackbone from ocr.core.models.encoder.timm_backbone",
      "category": "from_import",
      "code_snippet": "from ocr.features.detection.models.decoders.dbpp_decoder import DBPPDecoder\nfrom ocr.core.models.encoder.timm_backbone import TimmBackbone\nfrom ocr.features.detection.models.heads.db_head import DBHead",
      "metadata": {
        "module": "ocr.core.models.encoder.timm_backbone",
        "name": "TimmBackbone",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnetpp.py",
      "line": 8,
      "column": 0,
      "pattern": "from ocr.features.detection.models.heads.db_head import DBHead",
      "context": "Imports DBHead from ocr.features.detection.models.heads.db_head",
      "category": "from_import",
      "code_snippet": "from ocr.core.models.encoder.timm_backbone import TimmBackbone\nfrom ocr.features.detection.models.heads.db_head import DBHead\nfrom ocr.core.models.loss.db_loss import DBLoss",
      "metadata": {
        "module": "ocr.features.detection.models.heads.db_head",
        "name": "DBHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/architectures/dbnetpp.py",
      "line": 9,
      "column": 0,
      "pattern": "from ocr.core.models.loss.db_loss import DBLoss",
      "context": "Imports DBLoss from ocr.core.models.loss.db_loss",
      "category": "from_import",
      "code_snippet": "from ocr.features.detection.models.heads.db_head import DBHead\nfrom ocr.core.models.loss.db_loss import DBLoss\n",
      "metadata": {
        "module": "ocr.core.models.loss.db_loss",
        "name": "DBLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/craft_decoder.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/craft_decoder.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/craft_decoder.py",
      "line": 8,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/craft_decoder.py",
      "line": 9,
      "column": 0,
      "pattern": "import torch.nn.functional",
      "context": "Imports module: torch.nn.functional",
      "category": "import",
      "code_snippet": "import torch.nn as nn\nimport torch.nn.functional as F\n",
      "metadata": {
        "module": "torch.nn.functional",
        "alias": "F"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/craft_decoder.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core import BaseDecoder",
      "context": "Imports BaseDecoder from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseDecoder\n",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/dbpp_decoder.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/dbpp_decoder.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/dbpp_decoder.py",
      "line": 8,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/dbpp_decoder.py",
      "line": 9,
      "column": 0,
      "pattern": "import torch.nn.functional",
      "context": "Imports module: torch.nn.functional",
      "category": "import",
      "code_snippet": "import torch.nn as nn\nimport torch.nn.functional as F\n",
      "metadata": {
        "module": "torch.nn.functional",
        "alias": "F"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/dbpp_decoder.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core import BaseDecoder",
      "context": "Imports BaseDecoder from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseDecoder\n",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/fpn_decoder.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/fpn_decoder.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/fpn_decoder.py",
      "line": 8,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/fpn_decoder.py",
      "line": 9,
      "column": 0,
      "pattern": "import torch.nn.functional",
      "context": "Imports module: torch.nn.functional",
      "category": "import",
      "code_snippet": "import torch.nn as nn\nimport torch.nn.functional as F\n",
      "metadata": {
        "module": "torch.nn.functional",
        "alias": "F"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/fpn_decoder.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core import BaseDecoder",
      "context": "Imports BaseDecoder from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseDecoder\n",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/encoders/craft_vgg.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/encoders/craft_vgg.py",
      "line": 7,
      "column": 0,
      "pattern": "import timm",
      "context": "Imports module: timm",
      "category": "import",
      "code_snippet": "\nimport timm\nimport torch",
      "metadata": {
        "module": "timm",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/encoders/craft_vgg.py",
      "line": 8,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import timm\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/encoders/craft_vgg.py",
      "line": 9,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\n",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/encoders/craft_vgg.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.core import BaseEncoder",
      "context": "Imports BaseEncoder from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseEncoder\n",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseEncoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/heads/craft_head.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/heads/craft_head.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/heads/craft_head.py",
      "line": 8,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\n",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/heads/craft_head.py",
      "line": 10,
      "column": 0,
      "pattern": "from ocr.core import BaseHead",
      "context": "Imports BaseHead from ocr.core",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core import BaseHead\nfrom ocr.features.detection.models.postprocess.craft_postprocess import CraftPostProcessor",
      "metadata": {
        "module": "ocr.core",
        "name": "BaseHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/heads/craft_head.py",
      "line": 11,
      "column": 0,
      "pattern": "from ocr.features.detection.models.postprocess.craft_postprocess import CraftPostProcessor",
      "context": "Imports CraftPostProcessor from ocr.features.detection.models.postprocess.craft_postprocess",
      "category": "from_import",
      "code_snippet": "from ocr.core import BaseHead\nfrom ocr.features.detection.models.postprocess.craft_postprocess import CraftPostProcessor\n",
      "metadata": {
        "module": "ocr.features.detection.models.postprocess.craft_postprocess",
        "name": "CraftPostProcessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/heads/db_head.py",
      "line": 17,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/heads/db_head.py",
      "line": 18,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\n",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/heads/db_head.py",
      "line": 20,
      "column": 0,
      "pattern": "from ocr.features.detection.interfaces import DetectionHead",
      "context": "Imports DetectionHead from ocr.features.detection.interfaces",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.detection.interfaces import DetectionHead\nfrom ocr.features.detection.models.postprocess.db_postprocess import DBPostProcessor",
      "metadata": {
        "module": "ocr.features.detection.interfaces",
        "name": "DetectionHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/heads/db_head.py",
      "line": 21,
      "column": 0,
      "pattern": "from ocr.features.detection.models.postprocess.db_postprocess import DBPostProcessor",
      "context": "Imports DBPostProcessor from ocr.features.detection.models.postprocess.db_postprocess",
      "category": "from_import",
      "code_snippet": "from ocr.features.detection.interfaces import DetectionHead\nfrom ocr.features.detection.models.postprocess.db_postprocess import DBPostProcessor\n",
      "metadata": {
        "module": "ocr.features.detection.models.postprocess.db_postprocess",
        "name": "DBPostProcessor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/craft_postprocess.py",
      "line": 3,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/craft_postprocess.py",
      "line": 7,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/craft_postprocess.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/craft_postprocess.py",
      "line": 10,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/db_postprocess.py",
      "line": 15,
      "column": 0,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\nimport cv2\nimport numpy as np",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/db_postprocess.py",
      "line": 16,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import cv2\nimport numpy as np\nimport pyclipper",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/db_postprocess.py",
      "line": 17,
      "column": 0,
      "pattern": "import pyclipper",
      "context": "Imports module: pyclipper",
      "category": "import",
      "code_snippet": "import numpy as np\nimport pyclipper\nimport torch",
      "metadata": {
        "module": "pyclipper",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/db_postprocess.py",
      "line": 18,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import pyclipper\nimport torch\nfrom shapely.geometry import Polygon",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/db_postprocess.py",
      "line": 19,
      "column": 0,
      "pattern": "from shapely.geometry import Polygon",
      "context": "Imports Polygon from shapely.geometry",
      "category": "from_import",
      "code_snippet": "import torch\nfrom shapely.geometry import Polygon\n",
      "metadata": {
        "module": "shapely.geometry",
        "name": "Polygon",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/postprocess/db_postprocess.py",
      "line": 21,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import is_config",
      "context": "Imports is_config from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import is_config\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "is_config",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/data/__init__.py",
      "line": 6,
      "column": 8,
      "pattern": "from ocr.features.kie.data.dataset import KIEDataset",
      "context": "Imports KIEDataset from ocr.features.kie.data.dataset",
      "category": "from_import",
      "code_snippet": "    if name == \"KIEDataset\":\n        from ocr.features.kie.data.dataset import KIEDataset\n        return KIEDataset",
      "metadata": {
        "module": "ocr.features.kie.data.dataset",
        "name": "KIEDataset",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/data/dataset.py",
      "line": 5,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nimport pandas as pd",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/data/dataset.py",
      "line": 6,
      "column": 0,
      "pattern": "import pandas",
      "context": "Imports module: pandas",
      "category": "import",
      "code_snippet": "import numpy as np\nimport pandas as pd\nimport torch",
      "metadata": {
        "module": "pandas",
        "alias": "pd"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/data/dataset.py",
      "line": 7,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import pandas as pd\nimport torch\nfrom PIL import Image",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/data/dataset.py",
      "line": 8,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import torch\nfrom PIL import Image\nfrom torch.utils.data import Dataset",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/data/dataset.py",
      "line": 9,
      "column": 0,
      "pattern": "from torch.utils.data import Dataset",
      "context": "Imports Dataset from torch.utils.data",
      "category": "from_import",
      "code_snippet": "from PIL import Image\nfrom torch.utils.data import Dataset\nfrom transformers import PreTrainedTokenizerBase, ProcessorMixin",
      "metadata": {
        "module": "torch.utils.data",
        "name": "Dataset",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/data/dataset.py",
      "line": 10,
      "column": 0,
      "pattern": "from transformers import PreTrainedTokenizerBase",
      "context": "Imports PreTrainedTokenizerBase from transformers",
      "category": "from_import",
      "code_snippet": "from torch.utils.data import Dataset\nfrom transformers import PreTrainedTokenizerBase, ProcessorMixin\n",
      "metadata": {
        "module": "transformers",
        "name": "PreTrainedTokenizerBase",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/data/dataset.py",
      "line": 10,
      "column": 0,
      "pattern": "from transformers import ProcessorMixin",
      "context": "Imports ProcessorMixin from transformers",
      "category": "from_import",
      "code_snippet": "from torch.utils.data import Dataset\nfrom transformers import PreTrainedTokenizerBase, ProcessorMixin\n",
      "metadata": {
        "module": "transformers",
        "name": "ProcessorMixin",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/data/dataset.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.features.kie.validation import KIEDataItem",
      "context": "Imports KIEDataItem from ocr.features.kie.validation",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.kie.validation import KIEDataItem\n",
      "metadata": {
        "module": "ocr.features.kie.validation",
        "name": "KIEDataItem",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from field_extractor import ExtractorConfig",
      "context": "Imports ExtractorConfig from field_extractor",
      "category": "from_import",
      "code_snippet": "\nfrom .field_extractor import (\n    ExtractorConfig,",
      "metadata": {
        "module": "field_extractor",
        "name": "ExtractorConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from field_extractor import ReceiptFieldExtractor",
      "context": "Imports ReceiptFieldExtractor from field_extractor",
      "category": "from_import",
      "code_snippet": "\nfrom .field_extractor import (\n    ExtractorConfig,",
      "metadata": {
        "module": "field_extractor",
        "name": "ReceiptFieldExtractor",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/__init__.py",
      "line": 11,
      "column": 0,
      "pattern": "from normalizers import normalize_currency",
      "context": "Imports normalize_currency from normalizers",
      "category": "from_import",
      "code_snippet": ")\nfrom .normalizers import (\n    normalize_currency,",
      "metadata": {
        "module": "normalizers",
        "name": "normalize_currency",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/__init__.py",
      "line": 11,
      "column": 0,
      "pattern": "from normalizers import normalize_date",
      "context": "Imports normalize_date from normalizers",
      "category": "from_import",
      "code_snippet": ")\nfrom .normalizers import (\n    normalize_currency,",
      "metadata": {
        "module": "normalizers",
        "name": "normalize_date",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/__init__.py",
      "line": 11,
      "column": 0,
      "pattern": "from normalizers import normalize_phone",
      "context": "Imports normalize_phone from normalizers",
      "category": "from_import",
      "code_snippet": ")\nfrom .normalizers import (\n    normalize_currency,",
      "metadata": {
        "module": "normalizers",
        "name": "normalize_phone",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/__init__.py",
      "line": 11,
      "column": 0,
      "pattern": "from normalizers import normalize_time",
      "context": "Imports normalize_time from normalizers",
      "category": "from_import",
      "code_snippet": ")\nfrom .normalizers import (\n    normalize_currency,",
      "metadata": {
        "module": "normalizers",
        "name": "normalize_time",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/__init__.py",
      "line": 17,
      "column": 0,
      "pattern": "from receipt_schema import LineItem",
      "context": "Imports LineItem from receipt_schema",
      "category": "from_import",
      "code_snippet": ")\nfrom .receipt_schema import (\n    LineItem,",
      "metadata": {
        "module": "receipt_schema",
        "name": "LineItem",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/__init__.py",
      "line": 17,
      "column": 0,
      "pattern": "from receipt_schema import ReceiptData",
      "context": "Imports ReceiptData from receipt_schema",
      "category": "from_import",
      "code_snippet": ")\nfrom .receipt_schema import (\n    LineItem,",
      "metadata": {
        "module": "receipt_schema",
        "name": "ReceiptData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/__init__.py",
      "line": 17,
      "column": 0,
      "pattern": "from receipt_schema import ReceiptMetadata",
      "context": "Imports ReceiptMetadata from receipt_schema",
      "category": "from_import",
      "code_snippet": ")\nfrom .receipt_schema import (\n    LineItem,",
      "metadata": {
        "module": "receipt_schema",
        "name": "ReceiptMetadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 16,
      "column": 0,
      "pattern": "from normalizers import normalize_business_number",
      "context": "Imports normalize_business_number from normalizers",
      "category": "from_import",
      "code_snippet": "\nfrom .normalizers import (\n    normalize_business_number,",
      "metadata": {
        "module": "normalizers",
        "name": "normalize_business_number",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 16,
      "column": 0,
      "pattern": "from normalizers import normalize_currency",
      "context": "Imports normalize_currency from normalizers",
      "category": "from_import",
      "code_snippet": "\nfrom .normalizers import (\n    normalize_business_number,",
      "metadata": {
        "module": "normalizers",
        "name": "normalize_currency",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 16,
      "column": 0,
      "pattern": "from normalizers import normalize_date",
      "context": "Imports normalize_date from normalizers",
      "category": "from_import",
      "code_snippet": "\nfrom .normalizers import (\n    normalize_business_number,",
      "metadata": {
        "module": "normalizers",
        "name": "normalize_date",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 16,
      "column": 0,
      "pattern": "from normalizers import normalize_phone",
      "context": "Imports normalize_phone from normalizers",
      "category": "from_import",
      "code_snippet": "\nfrom .normalizers import (\n    normalize_business_number,",
      "metadata": {
        "module": "normalizers",
        "name": "normalize_phone",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 16,
      "column": 0,
      "pattern": "from normalizers import normalize_time",
      "context": "Imports normalize_time from normalizers",
      "category": "from_import",
      "code_snippet": "\nfrom .normalizers import (\n    normalize_business_number,",
      "metadata": {
        "module": "normalizers",
        "name": "normalize_time",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 23,
      "column": 0,
      "pattern": "from receipt_schema import LineItem",
      "context": "Imports LineItem from receipt_schema",
      "category": "from_import",
      "code_snippet": ")\nfrom .receipt_schema import LineItem, ReceiptData, ReceiptMetadata\n",
      "metadata": {
        "module": "receipt_schema",
        "name": "LineItem",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 23,
      "column": 0,
      "pattern": "from receipt_schema import ReceiptData",
      "context": "Imports ReceiptData from receipt_schema",
      "category": "from_import",
      "code_snippet": ")\nfrom .receipt_schema import LineItem, ReceiptData, ReceiptMetadata\n",
      "metadata": {
        "module": "receipt_schema",
        "name": "ReceiptData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 23,
      "column": 0,
      "pattern": "from receipt_schema import ReceiptMetadata",
      "context": "Imports ReceiptMetadata from receipt_schema",
      "category": "from_import",
      "code_snippet": ")\nfrom .receipt_schema import LineItem, ReceiptData, ReceiptMetadata\n",
      "metadata": {
        "module": "receipt_schema",
        "name": "ReceiptMetadata",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/field_extractor.py",
      "line": 26,
      "column": 4,
      "pattern": "from layout.contracts import LayoutResult",
      "context": "Imports LayoutResult from layout.contracts",
      "category": "from_import",
      "code_snippet": "if TYPE_CHECKING:\n    from ..layout.contracts import LayoutResult\n",
      "metadata": {
        "module": "layout.contracts",
        "name": "LayoutResult",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/normalizers.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/receipt_schema.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/receipt_schema.py",
      "line": 13,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/receipt_schema.py",
      "line": 13,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/receipt_schema.py",
      "line": 13,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/vlm_extractor.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/vlm_extractor.py",
      "line": 14,
      "column": 0,
      "pattern": "import httpx",
      "context": "Imports module: httpx",
      "category": "import",
      "code_snippet": "\nimport httpx\nfrom PIL import Image",
      "metadata": {
        "module": "httpx",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/vlm_extractor.py",
      "line": 15,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import httpx\nfrom PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/inference/extraction/vlm_extractor.py",
      "line": 17,
      "column": 0,
      "pattern": "from receipt_schema import ReceiptData",
      "context": "Imports ReceiptData from receipt_schema",
      "category": "from_import",
      "code_snippet": "\nfrom .receipt_schema import ReceiptData\n",
      "metadata": {
        "module": "receipt_schema",
        "name": "ReceiptData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/lightning/callbacks/kie_wandb_image_logging.py",
      "line": 4,
      "column": 0,
      "pattern": "import lightning.pytorch",
      "context": "Imports module: lightning.pytorch",
      "category": "import",
      "code_snippet": "\nimport lightning.pytorch as pl\nimport torch",
      "metadata": {
        "module": "lightning.pytorch",
        "alias": "pl"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/lightning/callbacks/kie_wandb_image_logging.py",
      "line": 5,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import lightning.pytorch as pl\nimport torch\nfrom PIL import Image, ImageDraw",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/lightning/callbacks/kie_wandb_image_logging.py",
      "line": 6,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import torch\nfrom PIL import Image, ImageDraw\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/lightning/callbacks/kie_wandb_image_logging.py",
      "line": 6,
      "column": 0,
      "pattern": "from PIL import ImageDraw",
      "context": "Imports ImageDraw from PIL",
      "category": "from_import",
      "code_snippet": "import torch\nfrom PIL import Image, ImageDraw\n",
      "metadata": {
        "module": "PIL",
        "name": "ImageDraw",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/lightning/callbacks/kie_wandb_image_logging.py",
      "line": 8,
      "column": 0,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "\nimport wandb\n",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/models/__init__.py",
      "line": 6,
      "column": 8,
      "pattern": "from ocr.features.kie.models.model import LayoutLMv3Wrapper",
      "context": "Imports LayoutLMv3Wrapper from ocr.features.kie.models.model",
      "category": "from_import",
      "code_snippet": "    if name == \"LayoutLMv3Wrapper\":\n        from ocr.features.kie.models.model import LayoutLMv3Wrapper\n        return LayoutLMv3Wrapper",
      "metadata": {
        "module": "ocr.features.kie.models.model",
        "name": "LayoutLMv3Wrapper",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/models/__init__.py",
      "line": 9,
      "column": 8,
      "pattern": "from ocr.features.kie.models.model import LiLTWrapper",
      "context": "Imports LiLTWrapper from ocr.features.kie.models.model",
      "category": "from_import",
      "code_snippet": "    elif name == \"LiLTWrapper\":\n        from ocr.features.kie.models.model import LiLTWrapper\n        return LiLTWrapper",
      "metadata": {
        "module": "ocr.features.kie.models.model",
        "name": "LiLTWrapper",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/models/model.py",
      "line": 1,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/models/model.py",
      "line": 2,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\nfrom transformers import LayoutLMv3ForTokenClassification, LiltForTokenClassification",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/models/model.py",
      "line": 3,
      "column": 0,
      "pattern": "from transformers import LayoutLMv3ForTokenClassification",
      "context": "Imports LayoutLMv3ForTokenClassification from transformers",
      "category": "from_import",
      "code_snippet": "import torch.nn as nn\nfrom transformers import LayoutLMv3ForTokenClassification, LiltForTokenClassification\n",
      "metadata": {
        "module": "transformers",
        "name": "LayoutLMv3ForTokenClassification",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/models/model.py",
      "line": 3,
      "column": 0,
      "pattern": "from transformers import LiltForTokenClassification",
      "context": "Imports LiltForTokenClassification from transformers",
      "category": "from_import",
      "code_snippet": "import torch.nn as nn\nfrom transformers import LayoutLMv3ForTokenClassification, LiltForTokenClassification\n",
      "metadata": {
        "module": "transformers",
        "name": "LiltForTokenClassification",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/models/model.py",
      "line": 5,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import ensure_dict",
      "context": "Imports ensure_dict from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.config_utils import ensure_dict\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "ensure_dict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/trainer.py",
      "line": 1,
      "column": 0,
      "pattern": "import lightning.pytorch",
      "context": "Imports module: lightning.pytorch",
      "category": "import",
      "code_snippet": "import lightning.pytorch as pl\nimport torch",
      "metadata": {
        "module": "lightning.pytorch",
        "alias": "pl"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/trainer.py",
      "line": 2,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import lightning.pytorch as pl\nimport torch\nfrom seqeval.metrics import classification_report, f1_score",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/trainer.py",
      "line": 3,
      "column": 0,
      "pattern": "from seqeval.metrics import classification_report",
      "context": "Imports classification_report from seqeval.metrics",
      "category": "from_import",
      "code_snippet": "import torch\nfrom seqeval.metrics import classification_report, f1_score\nfrom torch.utils.data import DataLoader",
      "metadata": {
        "module": "seqeval.metrics",
        "name": "classification_report",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/trainer.py",
      "line": 3,
      "column": 0,
      "pattern": "from seqeval.metrics import f1_score",
      "context": "Imports f1_score from seqeval.metrics",
      "category": "from_import",
      "code_snippet": "import torch\nfrom seqeval.metrics import classification_report, f1_score\nfrom torch.utils.data import DataLoader",
      "metadata": {
        "module": "seqeval.metrics",
        "name": "f1_score",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/trainer.py",
      "line": 4,
      "column": 0,
      "pattern": "from torch.utils.data import DataLoader",
      "context": "Imports DataLoader from torch.utils.data",
      "category": "from_import",
      "code_snippet": "from seqeval.metrics import classification_report, f1_score\nfrom torch.utils.data import DataLoader\nfrom transformers import get_linear_schedule_with_warmup",
      "metadata": {
        "module": "torch.utils.data",
        "name": "DataLoader",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/trainer.py",
      "line": 5,
      "column": 0,
      "pattern": "from transformers import get_linear_schedule_with_warmup",
      "context": "Imports get_linear_schedule_with_warmup from transformers",
      "category": "from_import",
      "code_snippet": "from torch.utils.data import DataLoader\nfrom transformers import get_linear_schedule_with_warmup\n",
      "metadata": {
        "module": "transformers",
        "name": "get_linear_schedule_with_warmup",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/trainer.py",
      "line": 8,
      "column": 0,
      "pattern": "from ocr.core.utils.config_utils import ensure_dict",
      "context": "Imports ensure_dict from ocr.core.utils.config_utils",
      "category": "from_import",
      "code_snippet": "# Use centralized config utilities\nfrom ocr.core.utils.config_utils import ensure_dict\n",
      "metadata": {
        "module": "ocr.core.utils.config_utils",
        "name": "ensure_dict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/validation.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/validation.py",
      "line": 11,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "\nimport torch\nfrom pydantic import BaseModel, ConfigDict, field_validator",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/validation.py",
      "line": 12,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/validation.py",
      "line": 12,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/kie/validation.py",
      "line": 12,
      "column": 0,
      "pattern": "from pydantic import field_validator",
      "context": "Imports field_validator from pydantic",
      "category": "from_import",
      "code_snippet": "import torch\nfrom pydantic import BaseModel, ConfigDict, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "field_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.features.layout.inference.contracts import BoundingBox",
      "context": "Imports BoundingBox from ocr.features.layout.inference.contracts",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.layout.inference.contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "ocr.features.layout.inference.contracts",
        "name": "BoundingBox",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.features.layout.inference.contracts import LayoutResult",
      "context": "Imports LayoutResult from ocr.features.layout.inference.contracts",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.layout.inference.contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "ocr.features.layout.inference.contracts",
        "name": "LayoutResult",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.features.layout.inference.contracts import TextBlock",
      "context": "Imports TextBlock from ocr.features.layout.inference.contracts",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.layout.inference.contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "ocr.features.layout.inference.contracts",
        "name": "TextBlock",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.features.layout.inference.contracts import TextElement",
      "context": "Imports TextElement from ocr.features.layout.inference.contracts",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.layout.inference.contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "ocr.features.layout.inference.contracts",
        "name": "TextElement",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.features.layout.inference.contracts import TextLine",
      "context": "Imports TextLine from ocr.features.layout.inference.contracts",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.layout.inference.contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "ocr.features.layout.inference.contracts",
        "name": "TextLine",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/__init__.py",
      "line": 14,
      "column": 0,
      "pattern": "from ocr.features.layout.inference.grouper import LineGrouper",
      "context": "Imports LineGrouper from ocr.features.layout.inference.grouper",
      "category": "from_import",
      "code_snippet": ")\nfrom ocr.features.layout.inference.grouper import (\n    LineGrouper,",
      "metadata": {
        "module": "ocr.features.layout.inference.grouper",
        "name": "LineGrouper",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/__init__.py",
      "line": 14,
      "column": 0,
      "pattern": "from ocr.features.layout.inference.grouper import LineGrouperConfig",
      "context": "Imports LineGrouperConfig from ocr.features.layout.inference.grouper",
      "category": "from_import",
      "code_snippet": ")\nfrom ocr.features.layout.inference.grouper import (\n    LineGrouper,",
      "metadata": {
        "module": "ocr.features.layout.inference.grouper",
        "name": "LineGrouperConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/__init__.py",
      "line": 14,
      "column": 0,
      "pattern": "from ocr.features.layout.inference.grouper import create_text_element",
      "context": "Imports create_text_element from ocr.features.layout.inference.grouper",
      "category": "from_import",
      "code_snippet": ")\nfrom ocr.features.layout.inference.grouper import (\n    LineGrouper,",
      "metadata": {
        "module": "ocr.features.layout.inference.grouper",
        "name": "create_text_element",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from contracts import BoundingBox",
      "context": "Imports BoundingBox from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "BoundingBox",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from contracts import LayoutResult",
      "context": "Imports LayoutResult from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "LayoutResult",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from contracts import TextBlock",
      "context": "Imports TextBlock from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "TextBlock",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from contracts import TextElement",
      "context": "Imports TextElement from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "TextElement",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from contracts import TextLine",
      "context": "Imports TextLine from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "TextLine",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/__init__.py",
      "line": 14,
      "column": 0,
      "pattern": "from grouper import LineGrouper",
      "context": "Imports LineGrouper from grouper",
      "category": "from_import",
      "code_snippet": ")\nfrom .grouper import LineGrouper, LineGrouperConfig\n",
      "metadata": {
        "module": "grouper",
        "name": "LineGrouper",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/__init__.py",
      "line": 14,
      "column": 0,
      "pattern": "from grouper import LineGrouperConfig",
      "context": "Imports LineGrouperConfig from grouper",
      "category": "from_import",
      "code_snippet": ")\nfrom .grouper import LineGrouper, LineGrouperConfig\n",
      "metadata": {
        "module": "grouper",
        "name": "LineGrouperConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/contracts.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/contracts.py",
      "line": 12,
      "column": 0,
      "pattern": "from pydantic import BaseModel",
      "context": "Imports BaseModel from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "BaseModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/contracts.py",
      "line": 12,
      "column": 0,
      "pattern": "from pydantic import ConfigDict",
      "context": "Imports ConfigDict from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "ConfigDict",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/contracts.py",
      "line": 12,
      "column": 0,
      "pattern": "from pydantic import Field",
      "context": "Imports Field from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "Field",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/contracts.py",
      "line": 12,
      "column": 0,
      "pattern": "from pydantic import field_validator",
      "context": "Imports field_validator from pydantic",
      "category": "from_import",
      "code_snippet": "\nfrom pydantic import BaseModel, ConfigDict, Field, field_validator\n",
      "metadata": {
        "module": "pydantic",
        "name": "field_validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/grouper.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/grouper.py",
      "line": 12,
      "column": 0,
      "pattern": "from contracts import BoundingBox",
      "context": "Imports BoundingBox from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "BoundingBox",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/grouper.py",
      "line": 12,
      "column": 0,
      "pattern": "from contracts import LayoutResult",
      "context": "Imports LayoutResult from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "LayoutResult",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/grouper.py",
      "line": 12,
      "column": 0,
      "pattern": "from contracts import TextBlock",
      "context": "Imports TextBlock from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "TextBlock",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/grouper.py",
      "line": 12,
      "column": 0,
      "pattern": "from contracts import TextElement",
      "context": "Imports TextElement from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "TextElement",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/layout/inference/grouper.py",
      "line": 12,
      "column": 0,
      "pattern": "from contracts import TextLine",
      "context": "Imports TextLine from contracts",
      "category": "from_import",
      "code_snippet": "\nfrom .contracts import (\n    BoundingBox,",
      "metadata": {
        "module": "contracts",
        "name": "TextLine",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/callbacks/wandb_image_logging.py",
      "line": 1,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "from __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/callbacks/wandb_image_logging.py",
      "line": 7,
      "column": 0,
      "pattern": "import lightning.pytorch",
      "context": "Imports module: lightning.pytorch",
      "category": "import",
      "code_snippet": "\nimport lightning.pytorch as pl\nimport numpy as np",
      "metadata": {
        "module": "lightning.pytorch",
        "alias": "pl"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/callbacks/wandb_image_logging.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "import lightning.pytorch as pl\nimport numpy as np\nimport torch",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/callbacks/wandb_image_logging.py",
      "line": 9,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import numpy as np\nimport torch\n",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/callbacks/wandb_image_logging.py",
      "line": 64,
      "column": 8,
      "pattern": "import wandb",
      "context": "Imports module: wandb",
      "category": "import",
      "code_snippet": "        # Lazy imports to reduce module load time\n        import wandb\n        from PIL import Image",
      "metadata": {
        "module": "wandb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/callbacks/wandb_image_logging.py",
      "line": 65,
      "column": 8,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "        import wandb\n        from PIL import Image\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/callbacks/wandb_image_logging.py",
      "line": 68,
      "column": 12,
      "pattern": "from ocr.core.utils.text_rendering import put_text_utf8",
      "context": "Imports put_text_utf8 from ocr.core.utils.text_rendering",
      "category": "from_import",
      "code_snippet": "        try:\n            from ocr.core.utils.text_rendering import put_text_utf8\n        except ImportError:",
      "metadata": {
        "module": "ocr.core.utils.text_rendering",
        "name": "put_text_utf8",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/callbacks/wandb_image_logging.py",
      "line": 218,
      "column": 12,
      "pattern": "import cv2",
      "context": "Imports module: cv2",
      "category": "import",
      "code_snippet": "\n            import cv2\n            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)",
      "metadata": {
        "module": "cv2",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/data/__init__.py",
      "line": 3,
      "column": 0,
      "pattern": "from tokenizer import KoreanOCRTokenizer",
      "context": "Imports KoreanOCRTokenizer from tokenizer",
      "category": "from_import",
      "code_snippet": "\nfrom .tokenizer import KoreanOCRTokenizer\nfrom .lmdb_dataset import LMDBRecognitionDataset",
      "metadata": {
        "module": "tokenizer",
        "name": "KoreanOCRTokenizer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/data/__init__.py",
      "line": 4,
      "column": 0,
      "pattern": "from lmdb_dataset import LMDBRecognitionDataset",
      "context": "Imports LMDBRecognitionDataset from lmdb_dataset",
      "category": "from_import",
      "code_snippet": "from .tokenizer import KoreanOCRTokenizer\nfrom .lmdb_dataset import LMDBRecognitionDataset\n",
      "metadata": {
        "module": "lmdb_dataset",
        "name": "LMDBRecognitionDataset",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/data/lmdb_dataset.py",
      "line": 8,
      "column": 0,
      "pattern": "import lmdb",
      "context": "Imports module: lmdb",
      "category": "import",
      "code_snippet": "\nimport lmdb\nimport torch",
      "metadata": {
        "module": "lmdb",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/data/lmdb_dataset.py",
      "line": 9,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import lmdb\nimport torch\nfrom PIL import Image",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/data/lmdb_dataset.py",
      "line": 10,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "import torch\nfrom PIL import Image\nfrom torch.utils.data import Dataset",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/data/lmdb_dataset.py",
      "line": 11,
      "column": 0,
      "pattern": "from torch.utils.data import Dataset",
      "context": "Imports Dataset from torch.utils.data",
      "category": "from_import",
      "code_snippet": "from PIL import Image\nfrom torch.utils.data import Dataset\n",
      "metadata": {
        "module": "torch.utils.data",
        "name": "Dataset",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/backends/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/backends/paddleocr_recognizer.py",
      "line": 7,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/backends/paddleocr_recognizer.py",
      "line": 13,
      "column": 4,
      "pattern": "from ocr.features.recognition.inference.recognizer import RecognitionInput",
      "context": "Imports RecognitionInput from ocr.features.recognition.inference.recognizer",
      "category": "from_import",
      "code_snippet": "if TYPE_CHECKING:\n    from ocr.features.recognition.inference.recognizer import RecognitionInput, RecognitionOutput, RecognizerConfig\n",
      "metadata": {
        "module": "ocr.features.recognition.inference.recognizer",
        "name": "RecognitionInput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/backends/paddleocr_recognizer.py",
      "line": 13,
      "column": 4,
      "pattern": "from ocr.features.recognition.inference.recognizer import RecognitionOutput",
      "context": "Imports RecognitionOutput from ocr.features.recognition.inference.recognizer",
      "category": "from_import",
      "code_snippet": "if TYPE_CHECKING:\n    from ocr.features.recognition.inference.recognizer import RecognitionInput, RecognitionOutput, RecognizerConfig\n",
      "metadata": {
        "module": "ocr.features.recognition.inference.recognizer",
        "name": "RecognitionOutput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/backends/paddleocr_recognizer.py",
      "line": 13,
      "column": 4,
      "pattern": "from ocr.features.recognition.inference.recognizer import RecognizerConfig",
      "context": "Imports RecognizerConfig from ocr.features.recognition.inference.recognizer",
      "category": "from_import",
      "code_snippet": "if TYPE_CHECKING:\n    from ocr.features.recognition.inference.recognizer import RecognitionInput, RecognitionOutput, RecognizerConfig\n",
      "metadata": {
        "module": "ocr.features.recognition.inference.recognizer",
        "name": "RecognizerConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/backends/paddleocr_recognizer.py",
      "line": 15,
      "column": 0,
      "pattern": "from ocr.features.recognition.inference.recognizer import BaseRecognizer",
      "context": "Imports BaseRecognizer from ocr.features.recognition.inference.recognizer",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.recognition.inference.recognizer import BaseRecognizer, RecognitionOutput\n",
      "metadata": {
        "module": "ocr.features.recognition.inference.recognizer",
        "name": "BaseRecognizer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/backends/paddleocr_recognizer.py",
      "line": 15,
      "column": 0,
      "pattern": "from ocr.features.recognition.inference.recognizer import RecognitionOutput",
      "context": "Imports RecognitionOutput from ocr.features.recognition.inference.recognizer",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.features.recognition.inference.recognizer import BaseRecognizer, RecognitionOutput\n",
      "metadata": {
        "module": "ocr.features.recognition.inference.recognizer",
        "name": "RecognitionOutput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/backends/paddleocr_recognizer.py",
      "line": 47,
      "column": 12,
      "pattern": "from paddleocr import PaddleOCR",
      "context": "Imports PaddleOCR from paddleocr",
      "category": "from_import",
      "code_snippet": "        try:\n            from paddleocr import PaddleOCR\n        except ImportError as e:",
      "metadata": {
        "module": "paddleocr",
        "name": "PaddleOCR",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/recognizer.py",
      "line": 10,
      "column": 0,
      "pattern": "from __future__ import annotations",
      "context": "Imports annotations from __future__",
      "category": "from_import",
      "code_snippet": "\nfrom __future__ import annotations\n",
      "metadata": {
        "module": "__future__",
        "name": "annotations",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/recognizer.py",
      "line": 18,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/inference/recognizer.py",
      "line": 228,
      "column": 12,
      "pattern": "from ocr.features.recognition.inference.backends.paddleocr_recognizer import PaddleOCRRecognizer",
      "context": "Imports PaddleOCRRecognizer from ocr.features.recognition.inference.backends.paddleocr_recognizer",
      "category": "from_import",
      "code_snippet": "        elif self.config.backend == RecognizerBackend.PADDLEOCR:\n            from ocr.features.recognition.inference.backends.paddleocr_recognizer import PaddleOCRRecognizer\n",
      "metadata": {
        "module": "ocr.features.recognition.inference.backends.paddleocr_recognizer",
        "name": "PaddleOCRRecognizer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/__init__.py",
      "line": 6,
      "column": 8,
      "pattern": "from ocr.features.recognition.models.architecture import PARSeq",
      "context": "Imports PARSeq from ocr.features.recognition.models.architecture",
      "category": "from_import",
      "code_snippet": "    if name == \"PARSeq\":\n        from ocr.features.recognition.models.architecture import PARSeq\n        return PARSeq",
      "metadata": {
        "module": "ocr.features.recognition.models.architecture",
        "name": "PARSeq",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/__init__.py",
      "line": 9,
      "column": 8,
      "pattern": "from ocr.features.recognition.models.architecture import register_parseq_components",
      "context": "Imports register_parseq_components from ocr.features.recognition.models.architecture",
      "category": "from_import",
      "code_snippet": "    elif name == \"register_parseq_components\":\n        from ocr.features.recognition.models.architecture import register_parseq_components\n        return register_parseq_components",
      "metadata": {
        "module": "ocr.features.recognition.models.architecture",
        "name": "register_parseq_components",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/__init__.py",
      "line": 12,
      "column": 8,
      "pattern": "from ocr.features.recognition.models.decoder import PARSeqDecoder",
      "context": "Imports PARSeqDecoder from ocr.features.recognition.models.decoder",
      "category": "from_import",
      "code_snippet": "    elif name == \"PARSeqDecoder\":\n        from ocr.features.recognition.models.decoder import PARSeqDecoder\n        return PARSeqDecoder",
      "metadata": {
        "module": "ocr.features.recognition.models.decoder",
        "name": "PARSeqDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/__init__.py",
      "line": 15,
      "column": 8,
      "pattern": "from ocr.features.recognition.models.head import PARSeqHead",
      "context": "Imports PARSeqHead from ocr.features.recognition.models.head",
      "category": "from_import",
      "code_snippet": "    elif name == \"PARSeqHead\":\n        from ocr.features.recognition.models.head import PARSeqHead\n        return PARSeqHead",
      "metadata": {
        "module": "ocr.features.recognition.models.head",
        "name": "PARSeqHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/architecture.py",
      "line": 1,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import torch\nfrom ocr.core.models.architecture import OCRModel",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/architecture.py",
      "line": 2,
      "column": 0,
      "pattern": "from ocr.core.models.architecture import OCRModel",
      "context": "Imports OCRModel from ocr.core.models.architecture",
      "category": "from_import",
      "code_snippet": "import torch\nfrom ocr.core.models.architecture import OCRModel\nfrom ocr.core import registry",
      "metadata": {
        "module": "ocr.core.models.architecture",
        "name": "OCRModel",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/architecture.py",
      "line": 3,
      "column": 0,
      "pattern": "from ocr.core import registry",
      "context": "Imports registry from ocr.core",
      "category": "from_import",
      "code_snippet": "from ocr.core.models.architecture import OCRModel\nfrom ocr.core import registry\nfrom ocr.features.recognition.models.decoder import PARSeqDecoder",
      "metadata": {
        "module": "ocr.core",
        "name": "registry",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/architecture.py",
      "line": 4,
      "column": 0,
      "pattern": "from ocr.features.recognition.models.decoder import PARSeqDecoder",
      "context": "Imports PARSeqDecoder from ocr.features.recognition.models.decoder",
      "category": "from_import",
      "code_snippet": "from ocr.core import registry\nfrom ocr.features.recognition.models.decoder import PARSeqDecoder\nfrom ocr.features.recognition.models.head import PARSeqHead",
      "metadata": {
        "module": "ocr.features.recognition.models.decoder",
        "name": "PARSeqDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/architecture.py",
      "line": 5,
      "column": 0,
      "pattern": "from ocr.features.recognition.models.head import PARSeqHead",
      "context": "Imports PARSeqHead from ocr.features.recognition.models.head",
      "category": "from_import",
      "code_snippet": "from ocr.features.recognition.models.decoder import PARSeqDecoder\nfrom ocr.features.recognition.models.head import PARSeqHead\nfrom ocr.core.models.encoder.timm_backbone import TimmBackbone",
      "metadata": {
        "module": "ocr.features.recognition.models.head",
        "name": "PARSeqHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/architecture.py",
      "line": 6,
      "column": 0,
      "pattern": "from ocr.core.models.encoder.timm_backbone import TimmBackbone",
      "context": "Imports TimmBackbone from ocr.core.models.encoder.timm_backbone",
      "category": "from_import",
      "code_snippet": "from ocr.features.recognition.models.head import PARSeqHead\nfrom ocr.core.models.encoder.timm_backbone import TimmBackbone\nfrom ocr.core.models.loss.cross_entropy_loss import CrossEntropyLoss",
      "metadata": {
        "module": "ocr.core.models.encoder.timm_backbone",
        "name": "TimmBackbone",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/architecture.py",
      "line": 7,
      "column": 0,
      "pattern": "from ocr.core.models.loss.cross_entropy_loss import CrossEntropyLoss",
      "context": "Imports CrossEntropyLoss from ocr.core.models.loss.cross_entropy_loss",
      "category": "from_import",
      "code_snippet": "from ocr.core.models.encoder.timm_backbone import TimmBackbone\nfrom ocr.core.models.loss.cross_entropy_loss import CrossEntropyLoss\n",
      "metadata": {
        "module": "ocr.core.models.loss.cross_entropy_loss",
        "name": "CrossEntropyLoss",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/decoder.py",
      "line": 2,
      "column": 0,
      "pattern": "import torch",
      "context": "Imports module: torch",
      "category": "import",
      "code_snippet": "import math\nimport torch\nimport torch.nn as nn",
      "metadata": {
        "module": "torch",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/decoder.py",
      "line": 3,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch\nimport torch.nn as nn\nfrom ocr.core.interfaces.models import BaseDecoder",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/decoder.py",
      "line": 4,
      "column": 0,
      "pattern": "from ocr.core.interfaces.models import BaseDecoder",
      "context": "Imports BaseDecoder from ocr.core.interfaces.models",
      "category": "from_import",
      "code_snippet": "import torch.nn as nn\nfrom ocr.core.interfaces.models import BaseDecoder\n",
      "metadata": {
        "module": "ocr.core.interfaces.models",
        "name": "BaseDecoder",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/head.py",
      "line": 1,
      "column": 0,
      "pattern": "import torch.nn",
      "context": "Imports module: torch.nn",
      "category": "import",
      "code_snippet": "import torch.nn as nn\nfrom ocr.core.interfaces.models import BaseHead",
      "metadata": {
        "module": "torch.nn",
        "alias": "nn"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/head.py",
      "line": 2,
      "column": 0,
      "pattern": "from ocr.core.interfaces.models import BaseHead",
      "context": "Imports BaseHead from ocr.core.interfaces.models",
      "category": "from_import",
      "code_snippet": "import torch.nn as nn\nfrom ocr.core.interfaces.models import BaseHead\n",
      "metadata": {
        "module": "ocr.core.interfaces.models",
        "name": "BaseHead",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/__init__.py",
      "line": 10,
      "column": 0,
      "pattern": "from dataset import SyntheticDatasetGenerator",
      "context": "Imports SyntheticDatasetGenerator from dataset",
      "category": "from_import",
      "code_snippet": "\nfrom .dataset import SyntheticDatasetGenerator\nfrom .generators import BackgroundGenerator, TextGenerator, TextRenderer",
      "metadata": {
        "module": "dataset",
        "name": "SyntheticDatasetGenerator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/__init__.py",
      "line": 11,
      "column": 0,
      "pattern": "from generators import BackgroundGenerator",
      "context": "Imports BackgroundGenerator from generators",
      "category": "from_import",
      "code_snippet": "from .dataset import SyntheticDatasetGenerator\nfrom .generators import BackgroundGenerator, TextGenerator, TextRenderer\nfrom .models import SyntheticImage, TextRegion",
      "metadata": {
        "module": "generators",
        "name": "BackgroundGenerator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/__init__.py",
      "line": 11,
      "column": 0,
      "pattern": "from generators import TextGenerator",
      "context": "Imports TextGenerator from generators",
      "category": "from_import",
      "code_snippet": "from .dataset import SyntheticDatasetGenerator\nfrom .generators import BackgroundGenerator, TextGenerator, TextRenderer\nfrom .models import SyntheticImage, TextRegion",
      "metadata": {
        "module": "generators",
        "name": "TextGenerator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/__init__.py",
      "line": 11,
      "column": 0,
      "pattern": "from generators import TextRenderer",
      "context": "Imports TextRenderer from generators",
      "category": "from_import",
      "code_snippet": "from .dataset import SyntheticDatasetGenerator\nfrom .generators import BackgroundGenerator, TextGenerator, TextRenderer\nfrom .models import SyntheticImage, TextRegion",
      "metadata": {
        "module": "generators",
        "name": "TextRenderer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/__init__.py",
      "line": 12,
      "column": 0,
      "pattern": "from models import SyntheticImage",
      "context": "Imports SyntheticImage from models",
      "category": "from_import",
      "code_snippet": "from .generators import BackgroundGenerator, TextGenerator, TextRenderer\nfrom .models import SyntheticImage, TextRegion\nfrom .utils import augment_existing_dataset, create_synthetic_dataset, setup_augmentation_pipeline",
      "metadata": {
        "module": "models",
        "name": "SyntheticImage",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/__init__.py",
      "line": 12,
      "column": 0,
      "pattern": "from models import TextRegion",
      "context": "Imports TextRegion from models",
      "category": "from_import",
      "code_snippet": "from .generators import BackgroundGenerator, TextGenerator, TextRenderer\nfrom .models import SyntheticImage, TextRegion\nfrom .utils import augment_existing_dataset, create_synthetic_dataset, setup_augmentation_pipeline",
      "metadata": {
        "module": "models",
        "name": "TextRegion",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/__init__.py",
      "line": 13,
      "column": 0,
      "pattern": "from utils import augment_existing_dataset",
      "context": "Imports augment_existing_dataset from utils",
      "category": "from_import",
      "code_snippet": "from .models import SyntheticImage, TextRegion\nfrom .utils import augment_existing_dataset, create_synthetic_dataset, setup_augmentation_pipeline\n",
      "metadata": {
        "module": "utils",
        "name": "augment_existing_dataset",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/__init__.py",
      "line": 13,
      "column": 0,
      "pattern": "from utils import create_synthetic_dataset",
      "context": "Imports create_synthetic_dataset from utils",
      "category": "from_import",
      "code_snippet": "from .models import SyntheticImage, TextRegion\nfrom .utils import augment_existing_dataset, create_synthetic_dataset, setup_augmentation_pipeline\n",
      "metadata": {
        "module": "utils",
        "name": "create_synthetic_dataset",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/__init__.py",
      "line": 13,
      "column": 0,
      "pattern": "from utils import setup_augmentation_pipeline",
      "context": "Imports setup_augmentation_pipeline from utils",
      "category": "from_import",
      "code_snippet": "from .models import SyntheticImage, TextRegion\nfrom .utils import augment_existing_dataset, create_synthetic_dataset, setup_augmentation_pipeline\n",
      "metadata": {
        "module": "utils",
        "name": "setup_augmentation_pipeline",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 11,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom omegaconf import DictConfig",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 12,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom omegaconf import DictConfig\nfrom PIL import Image, ImageDraw, ImageFont",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 13,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "from omegaconf import DictConfig\nfrom PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 13,
      "column": 0,
      "pattern": "from PIL import ImageDraw",
      "context": "Imports ImageDraw from PIL",
      "category": "from_import",
      "code_snippet": "from omegaconf import DictConfig\nfrom PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "ImageDraw",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 13,
      "column": 0,
      "pattern": "from PIL import ImageFont",
      "context": "Imports ImageFont from PIL",
      "category": "from_import",
      "code_snippet": "from omegaconf import DictConfig\nfrom PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "ImageFont",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 16,
      "column": 4,
      "pattern": "import albumentations",
      "context": "Imports module: albumentations",
      "category": "import",
      "code_snippet": "try:\n    import albumentations as A\n",
      "metadata": {
        "module": "albumentations",
        "alias": "A"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 23,
      "column": 0,
      "pattern": "from ocr.core.utils.logging import logger",
      "context": "Imports logger from ocr.core.utils.logging",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.logging import logger\n",
      "metadata": {
        "module": "ocr.core.utils.logging",
        "name": "logger",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 25,
      "column": 0,
      "pattern": "from generators import BackgroundGenerator",
      "context": "Imports BackgroundGenerator from generators",
      "category": "from_import",
      "code_snippet": "\nfrom .generators import BackgroundGenerator, TextGenerator, TextRenderer\nfrom .models import SyntheticImage",
      "metadata": {
        "module": "generators",
        "name": "BackgroundGenerator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 25,
      "column": 0,
      "pattern": "from generators import TextGenerator",
      "context": "Imports TextGenerator from generators",
      "category": "from_import",
      "code_snippet": "\nfrom .generators import BackgroundGenerator, TextGenerator, TextRenderer\nfrom .models import SyntheticImage",
      "metadata": {
        "module": "generators",
        "name": "TextGenerator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 25,
      "column": 0,
      "pattern": "from generators import TextRenderer",
      "context": "Imports TextRenderer from generators",
      "category": "from_import",
      "code_snippet": "\nfrom .generators import BackgroundGenerator, TextGenerator, TextRenderer\nfrom .models import SyntheticImage",
      "metadata": {
        "module": "generators",
        "name": "TextRenderer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/dataset.py",
      "line": 26,
      "column": 0,
      "pattern": "from models import SyntheticImage",
      "context": "Imports SyntheticImage from models",
      "category": "from_import",
      "code_snippet": "from .generators import BackgroundGenerator, TextGenerator, TextRenderer\nfrom .models import SyntheticImage\n",
      "metadata": {
        "module": "models",
        "name": "SyntheticImage",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/__init__.py",
      "line": 6,
      "column": 0,
      "pattern": "from background import BackgroundGenerator",
      "context": "Imports BackgroundGenerator from background",
      "category": "from_import",
      "code_snippet": "\nfrom .background import BackgroundGenerator\nfrom .renderer import TextRenderer",
      "metadata": {
        "module": "background",
        "name": "BackgroundGenerator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/__init__.py",
      "line": 7,
      "column": 0,
      "pattern": "from renderer import TextRenderer",
      "context": "Imports TextRenderer from renderer",
      "category": "from_import",
      "code_snippet": "from .background import BackgroundGenerator\nfrom .renderer import TextRenderer\nfrom .text import TextGenerator",
      "metadata": {
        "module": "renderer",
        "name": "TextRenderer",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/__init__.py",
      "line": 8,
      "column": 0,
      "pattern": "from text import TextGenerator",
      "context": "Imports TextGenerator from text",
      "category": "from_import",
      "code_snippet": "from .renderer import TextRenderer\nfrom .text import TextGenerator\n",
      "metadata": {
        "module": "text",
        "name": "TextGenerator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/background.py",
      "line": 9,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom omegaconf import DictConfig",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/background.py",
      "line": 10,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom omegaconf import DictConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/renderer.py",
      "line": 8,
      "column": 0,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "\nimport numpy as np\nfrom omegaconf import DictConfig",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/renderer.py",
      "line": 9,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "import numpy as np\nfrom omegaconf import DictConfig\nfrom PIL import Image, ImageDraw, ImageFont",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/renderer.py",
      "line": 10,
      "column": 0,
      "pattern": "from PIL import Image",
      "context": "Imports Image from PIL",
      "category": "from_import",
      "code_snippet": "from omegaconf import DictConfig\nfrom PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "Image",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/renderer.py",
      "line": 10,
      "column": 0,
      "pattern": "from PIL import ImageDraw",
      "context": "Imports ImageDraw from PIL",
      "category": "from_import",
      "code_snippet": "from omegaconf import DictConfig\nfrom PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "ImageDraw",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/renderer.py",
      "line": 10,
      "column": 0,
      "pattern": "from PIL import ImageFont",
      "context": "Imports ImageFont from PIL",
      "category": "from_import",
      "code_snippet": "from omegaconf import DictConfig\nfrom PIL import Image, ImageDraw, ImageFont\n",
      "metadata": {
        "module": "PIL",
        "name": "ImageFont",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/renderer.py",
      "line": 12,
      "column": 0,
      "pattern": "from models import TextRegion",
      "context": "Imports TextRegion from models",
      "category": "from_import",
      "code_snippet": "\nfrom ..models import TextRegion\n",
      "metadata": {
        "module": "models",
        "name": "TextRegion",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/generators/text.py",
      "line": 9,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "\nfrom omegaconf import DictConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/models.py",
      "line": 10,
      "column": 4,
      "pattern": "import numpy",
      "context": "Imports module: numpy",
      "category": "import",
      "code_snippet": "if TYPE_CHECKING:\n    import numpy as np\n",
      "metadata": {
        "module": "numpy",
        "alias": "np"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/utils.py",
      "line": 8,
      "column": 0,
      "pattern": "from omegaconf import DictConfig",
      "context": "Imports DictConfig from omegaconf",
      "category": "from_import",
      "code_snippet": "\nfrom omegaconf import DictConfig\n",
      "metadata": {
        "module": "omegaconf",
        "name": "DictConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/utils.py",
      "line": 11,
      "column": 4,
      "pattern": "import albumentations",
      "context": "Imports module: albumentations",
      "category": "import",
      "code_snippet": "try:\n    import albumentations as A\n",
      "metadata": {
        "module": "albumentations",
        "alias": "A"
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/utils.py",
      "line": 18,
      "column": 0,
      "pattern": "from ocr.core.utils.logging import logger",
      "context": "Imports logger from ocr.core.utils.logging",
      "category": "from_import",
      "code_snippet": "\nfrom ocr.core.utils.logging import logger\n",
      "metadata": {
        "module": "ocr.core.utils.logging",
        "name": "logger",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/synthetic_data/utils.py",
      "line": 20,
      "column": 0,
      "pattern": "from dataset import SyntheticDatasetGenerator",
      "context": "Imports SyntheticDatasetGenerator from dataset",
      "category": "from_import",
      "code_snippet": "\nfrom .dataset import SyntheticDatasetGenerator\n",
      "metadata": {
        "module": "dataset",
        "name": "SyntheticDatasetGenerator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import VALID_EXIF_ORIENTATIONS",
      "context": "Imports VALID_EXIF_ORIENTATIONS from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "VALID_EXIF_ORIENTATIONS",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import BatchSample",
      "context": "Imports BatchSample from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "BatchSample",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import CollateOutput",
      "context": "Imports CollateOutput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "CollateOutput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import DatasetSample",
      "context": "Imports DatasetSample from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "DatasetSample",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import LightningStepPrediction",
      "context": "Imports LightningStepPrediction from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "LightningStepPrediction",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import MetricConfig",
      "context": "Imports MetricConfig from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "MetricConfig",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import ModelOutput",
      "context": "Imports ModelOutput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ModelOutput",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import PolygonArray",
      "context": "Imports PolygonArray from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "PolygonArray",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import ValidatedTensorData",
      "context": "Imports ValidatedTensorData from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "ValidatedTensorData",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import validate_predictions",
      "context": "Imports validate_predictions from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "validate_predictions",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 12,
      "column": 0,
      "pattern": "from ocr.core.validation import validator",
      "context": "Imports validator from ocr.core.validation",
      "category": "from_import",
      "code_snippet": "# Re-export all classes\nfrom ocr.core.validation import (\n    VALID_EXIF_ORIENTATIONS,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "validator",
        "alias": null
      }
    },
    {
      "file": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation/models.py",
      "line": 25,
      "column": 0,
      "pattern": "from ocr.core.validation import LoaderTransformOutput",
      "context": "Imports LoaderTransformOutput from ocr.core.validation",
      "category": "from_import",
      "code_snippet": ")\nfrom ocr.core.validation import (\n    LoaderTransformOutput as TransformOutput,",
      "metadata": {
        "module": "ocr.core.validation",
        "name": "LoaderTransformOutput",
        "alias": "TransformOutput"
      }
    }
  ],
  "summary": {
    "files_analyzed": 227,
    "total_findings": 1115,
    "errors": 0,
    "skipped": 0
  },
  "total_findings": 1115
}
