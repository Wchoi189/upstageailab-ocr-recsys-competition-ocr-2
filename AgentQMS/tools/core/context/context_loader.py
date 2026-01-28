"""
Context Loader Core

Manages intelligent context loading based on user messages and tasks.
Integrates with ContextSuggester to automatically load relevant bundles.
"""
import logging
from dataclasses import dataclass, field

# Import from utilities (ContextSuggester)
from AgentQMS.tools.utilities.suggest_context import ContextSuggester

# Import from core (Context Bundle Loader)
from AgentQMS.tools.core.context.context_bundle import get_context_bundle

logger = logging.getLogger(__name__)

@dataclass
class SessionState:
    """Mock session state for context tracking."""
    # In a real integration, this might be passed from the host application
    loaded_bundles: set[str] = field(default_factory=set)
    memory_footprint_mb: float = 0.0
    active_topics: list[str] = field(default_factory=list)
    message_history: list[str] = field(default_factory=list)
    bundle_last_used: dict[str, int] = field(default_factory=dict) # timestamp/turn count

class ContextLoader:
    def __init__(self, settings: dict = None):
        self.settings = settings or {}
        # Get config from settings or defaults
        self.enabled = self.settings.get("context_integration", {}).get("enabled", False)
        self.threshold = self.settings.get("context_integration", {}).get("auto_load_threshold", 5)
        self.window_size = self.settings.get("context_integration", {}).get("window_size", 5)
        self.persistence_turns = self.settings.get("context_integration", {}).get("persistence_turns", 3)

        self.suggester = ContextSuggester()
        self.turn_counter = 0
        self.analytics_enabled = self.settings.get("context_integration", {}).get("analytics_enabled", True)

    def process_message(self, message: str, session: SessionState) -> list[str]:
        """
        Process a user message, suggest context, and load if above threshold.
        Returns list of newly loaded bundles.
        """
        if not self.enabled:
            return []

        self.turn_counter += 1

        # Update history
        session.message_history.append(message)
        if len(session.message_history) > self.window_size:
            session.message_history.pop(0)

        # Analyze window
        # We focus on the latest message for immediate needs, but could use history for stability.
        # For now, let's analyze the latest message primarily, but maybe concatenate if short?
        # Let's try analyzing the concatenated last 2 messages for better context.
        recent_context = "\n".join(session.message_history[-2:])

        # Get suggestions
        result = self.suggester.suggest(recent_context)
        suggestions = result.get("suggestions", [])

        newly_loaded = []
        active_bundles_this_turn = set()

        for suggestion in suggestions:
            score = suggestion.get("score", 0)
            bundle_name = suggestion.get("context_bundle")

            if not bundle_name:
                continue

            # If high score, consider it active
            if score >= self.threshold:
                active_bundles_this_turn.add(bundle_name)
                session.bundle_last_used[bundle_name] = self.turn_counter

                if bundle_name not in session.loaded_bundles:
                    self._load_bundle(bundle_name, session, recent_context)
                    newly_loaded.append(bundle_name)

        # Unload stale bundles
        self._unload_stale_bundles(session)

        return newly_loaded

    def _load_bundle(self, bundle_name: str, session: SessionState, task_description: str):
        """Load a specific bundle and update session state."""
        try:
            # We use get_context_bundle to resolve files
            # We pass bundle_name as task_type to force loading that specific bundle
            files = get_context_bundle(task_description=task_description, task_type=bundle_name)

            session.loaded_bundles.add(bundle_name)

            # Simulate memory impact (e.g. 0.5 MB per bundle for now)
            # In a real system, we would measure the size of 'files' content
            session.memory_footprint_mb += 0.5

            logger.info(f"Automatically loaded context bundle: {bundle_name} (Score > {self.threshold})")
            logger.info(f"Bundle files: {len(files)}")

            if self.analytics_enabled:
                self._log_analytics("load", bundle_name)

        except Exception as e:
            logger.error(f"Failed to load bundle {bundle_name}: {e}")

    def _unload_stale_bundles(self, session: SessionState):
        """Unload bundles that haven't been relevant for persistence_turns."""
        to_unload = []
        for bundle in session.loaded_bundles:
            last_used = session.bundle_last_used.get(bundle, 0)
            if (self.turn_counter - last_used) > self.persistence_turns:
                to_unload.append(bundle)

        for bundle in to_unload:
            session.loaded_bundles.remove(bundle)
            session.memory_footprint_mb = max(0.0, session.memory_footprint_mb - 0.5)
            logger.info(f"Automatically unloaded stale bundle: {bundle}")
            if self.analytics_enabled:
                self._log_analytics("unload", bundle)

    def force_load_bundle(self, bundle_name: str, session: SessionState):
        """Manually force load a context bundle."""
        if bundle_name not in session.loaded_bundles:
            self._load_bundle(bundle_name, session, task_description=f"Manual load: {bundle_name}")
            # Mark as recently used so it doesn't immediately unload
            session.bundle_last_used[bundle_name] = self.turn_counter
            return True
        return False

    def force_unload_bundle(self, bundle_name: str, session: SessionState):
        """Manually force unload a context bundle."""
        if bundle_name in session.loaded_bundles:
            session.loaded_bundles.remove(bundle_name)
            session.memory_footprint_mb = max(0.0, session.memory_footprint_mb - 0.5)
            logger.info(f"Manually unloaded bundle: {bundle_name}")
            if self.analytics_enabled:
                self._log_analytics("manual_unload", bundle_name)
            return True
        return False

    def _log_analytics(self, action: str, bundle_name: str):
        """Log bundle usage analytics."""
        import json
        import datetime

        try:
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "action": action,
                "bundle_name": bundle_name,
                "turn": self.turn_counter
            }

            # Since we don't have a guaranteed writable logs dir passed in,
            # we'll log to structured logger and let the logging system handle file IO,
            # or try to write to a local log file if we are in the project.

            # Use standard logger for now, but with structured data
            logger.info(f"ANALYTICS: {json.dumps(entry)}")

        except Exception as e:
            logger.error(f"Failed to log analytics: {e}")
