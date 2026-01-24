"""
Preflight Infrastructure Check Module

Verifies connectivity to critical services:
1. Redis (State/Config Cache)
2. RabbitMQ (Message Broker)
3. Ollama (Local LLM Inference)
"""

import socket
import logging
import os
import requests
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class PreflightCheck:
    """Checks availability of core infrastructure services."""

    def __init__(self):
        self.redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "rabbitmq")
        self.rabbitmq_port = int(os.getenv("RABBITMQ_PORT", 5672))
        
        # Ollama usually runs on host.docker.internal or localhost
        self.ollama_host = os.getenv("OLLAMA_HOST", "host.docker.internal") 
        self.ollama_port = int(os.getenv("OLLAMA_PORT", 11434))

    def _check_socket(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if a TCP socket is open."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    def _check_ollama_http(self) -> bool:
        """Check if Ollama is responsive via HTTP."""
        url = f"http://{self.ollama_host}:{self.ollama_port}/api/version"
        try:
            response = requests.get(url, timeout=2.0)
            return response.status_code == 200
        except requests.RequestException:
            # Fallback: socket check only if HTTP endpoint differs or auth issues
            return self._check_socket(self.ollama_host, self.ollama_port)

    def check_all(self) -> Dict[str, bool]:
        """Run all checks and return status dict."""
        status = {
            "redis": self._check_socket(self.redis_host, self.redis_port),
            "rabbitmq": self._check_socket(self.rabbitmq_host, self.rabbitmq_port),
            "ollama": self._check_ollama_http()
        }
        return status

    def print_report(self):
        """Run checks and print a formatted report to stdout."""
        print("🔍 checking infrastructure...")
        results = self.check_all()
        
        all_ok = True
        
        # Redis
        if results["redis"]:
            print(f"✅ Redis       ({self.redis_host}:{self.redis_port}) is ONLINE")
        else:
            print(f"❌ Redis       ({self.redis_host}:{self.redis_port}) is OFFLINE")
            all_ok = False

        # RabbitMQ
        if results["rabbitmq"]:
            print(f"✅ RabbitMQ    ({self.rabbitmq_host}:{self.rabbitmq_port}) is ONLINE")
        else:
            print(f"❌ RabbitMQ    ({self.rabbitmq_host}:{self.rabbitmq_port}) is OFFLINE")
            all_ok = False

        # Ollama
        if results["ollama"]:
            print(f"✅ Ollama LLM  ({self.ollama_host}:{self.ollama_port}) is ONLINE")
        else:
            print(f"❌ Ollama LLM  ({self.ollama_host}:{self.ollama_port}) is OFFLINE")
            print("   -> Ensure 'ollama serve' is running and reachable.")
            all_ok = False

        if all_ok:
            print("\n🚀 System is ready for verify/inference.")
            return True
        else:
            print("\n⚠️  System Check Failed. Some services are unavailable.")
            return False

if __name__ == "__main__":
    checker = PreflightCheck()
    success = checker.print_report()
    import sys
    sys.exit(0 if success else 1)
