"""
Health Monitoring Module for AgentQMS Middleware

Provides comprehensive health checks for policy enforcement,
logging infrastructure, and metrics collection.
"""
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_config import setup_middleware_logger

logger = setup_middleware_logger("health")


def check_policies_health() -> dict[str, Any]:
    """Comprehensive health check for middleware policies.

    Returns:
        Health status dictionary with check results
    """
    from .policies import (
        ComplianceInterceptor,
        FileOperationInterceptor,
        ProactiveFeedbackInterceptor,
        RedundancyInterceptor,
        StandardsInterceptor,
    )

    results = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    # Define policies to test
    policies = {
        "ComplianceInterceptor": ComplianceInterceptor,
        "RedundancyInterceptor": RedundancyInterceptor,
        "StandardsInterceptor": StandardsInterceptor,
        "FileOperationInterceptor": FileOperationInterceptor,
        "ProactiveFeedbackInterceptor": ProactiveFeedbackInterceptor,
    }

    # Test each policy
    for policy_name, policy_class in policies.items():
        try:
            # Instantiate policy
            policy = policy_class()

            # Test validation with safe test data
            policy.validate("test_tool", {"test": "data"})

            results["checks"][policy_name] = "ok"
            logger.debug(f"Health check passed for {policy_name}")

        except Exception as e:
            results["status"] = "degraded"
            results["checks"][policy_name] = {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
            }
            logger.warning(
                f"Health check failed for {policy_name}",
                extra={"extra": {"error": str(e), "error_type": type(e).__name__}},
            )

    # Check logging infrastructure
    try:
        from AgentQMS.tools.utils.paths import get_project_root

        log_dir = get_project_root() / "outputs" / "logs" / "middleware"
        if log_dir.exists() and log_dir.is_dir():
            results["checks"]["logging_infrastructure"] = "ok"
        else:
            results["status"] = "degraded"
            results["checks"]["logging_infrastructure"] = {
                "status": "failed",
                "error": f"Log directory does not exist: {log_dir}",
            }
    except Exception as e:
        results["status"] = "degraded"
        results["checks"]["logging_infrastructure"] = {
            "status": "failed",
            "error": str(e),
        }

    # Check metrics directory
    try:
        from AgentQMS.tools.utils.paths import get_project_root

        metrics_dir = get_project_root() / "outputs" / "metrics"
        if metrics_dir.exists() and metrics_dir.is_dir():
            results["checks"]["metrics_directory"] = "ok"
        else:
            results["status"] = "degraded"
            results["checks"]["metrics_directory"] = {
                "status": "failed",
                "error": f"Metrics directory does not exist: {metrics_dir}",
            }
    except Exception as e:
        results["status"] = "degraded"
        results["checks"]["metrics_directory"] = {
            "status": "failed",
            "error": str(e),
        }

    logger.info(
        f"Health check completed with status: {results['status']}",
        extra={"extra": {"total_checks": len(results["checks"])}},
    )

    return results


def get_health_status() -> dict[str, Any]:
    """Get current health status of middleware.

    Returns:
        Health status summary
    """
    health = check_policies_health()

    # Calculate summary statistics
    total_checks = len(health["checks"])
    passed_checks = sum(1 for v in health["checks"].values() if v == "ok")
    failed_checks = total_checks - passed_checks

    return {
        "status": health["status"],
        "timestamp": health["timestamp"],
        "summary": {
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed": failed_checks,
        },
        "details": health["checks"],
    }


class HealthMonitor:
    """Legacy health monitor class for backwards compatibility."""

    def __init__(self):
        self.agents = {}  # Registry of monitored agents
        self.last_check = 0
        self.check_interval = 30
        logger.warning("HealthMonitor class is deprecated. Use check_policies_health() instead.")

    def register_agent(self, agent_id: str, ping_callback):
        self.agents[agent_id] = {
            "callback": ping_callback,
            "failures": 0,
            "status": "healthy",
        }

    def check_agents(self):
        now = time.time()
        if now - self.last_check < self.check_interval:
            return

        self.last_check = now
        for agent_id, data in self.agents.items():
            try:
                # specific ping logic or callback
                response = data["callback"]()
                if not response:
                    self._handle_failure(agent_id, "No response")
                else:
                    data["failures"] = 0
                    data["status"] = "healthy"
            except TimeoutError:
                self._handle_failure(agent_id, "Timeout")
            except Exception as e:
                self._handle_failure(agent_id, str(e))

    def _handle_failure(self, agent_id, reason):
        print(f"🚨 ALERT: Agent {agent_id} failed: {reason}")
        self.agents[agent_id]["failures"] += 1
        self.agents[agent_id]["status"] = "unhealthy"
        # self.restart_agent(agent_id) # Logic to restart

