"""
Health Monitoring Module
Fix #4: Silent Failure Detection
"""
import time
import logging

logger = logging.getLogger(__name__)

class HealthMonitor:
    def __init__(self):
        self.agents = {}  # Registry of monitored agents
        self.last_check = 0
        self.check_interval = 30

    def register_agent(self, agent_id: str, ping_callback):
        self.agents[agent_id] = {
            "callback": ping_callback,
            "failures": 0,
            "status": "healthy"
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
