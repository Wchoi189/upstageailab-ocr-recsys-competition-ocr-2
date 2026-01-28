import pytest
from unittest.mock import patch
from AgentQMS.tools.core.context.context_loader import ContextLoader, SessionState

class TestContextIntegration:

    @pytest.fixture
    def settings(self):
        return {
            "context_integration": {
                "enabled": True,
                "auto_load_threshold": 5
            }
        }

    @pytest.fixture
    def session(self):
        return SessionState()

    def test_context_loader_initialization(self, settings):
        loader = ContextLoader(settings)
        assert loader.enabled is True
        assert loader.threshold == 5

    @patch("AgentQMS.tools.core.context.context_loader.ContextSuggester")
    @patch("AgentQMS.tools.core.context.context_loader.get_context_bundle")
    def test_process_message_loads_bundle(self, mock_get_bundle, mock_suggester_class, settings, session):
        # Setup mocks
        mock_suggester = mock_suggester_class.return_value
        mock_suggester.suggest.return_value = {
            "suggestions": [
                {"context_bundle": "test-bundle", "score": 10}
            ]
        }
        mock_get_bundle.return_value = ["file1.py", "file2.py"]

        loader = ContextLoader(settings)

        # Action
        newly_loaded = loader.process_message("test message", session)

        # Assert assertions
        assert "test-bundle" in newly_loaded
        assert "test-bundle" in session.loaded_bundles
        assert session.memory_footprint_mb > 0
        mock_get_bundle.assert_called_with(task_description="test message", task_type="test-bundle")

    @patch("AgentQMS.tools.core.context.context_loader.ContextSuggester")
    def test_process_message_ignores_low_score(self, mock_suggester_class, settings, session):
        # Setup mocks
        mock_suggester = mock_suggester_class.return_value
        mock_suggester.suggest.return_value = {
            "suggestions": [
                {"context_bundle": "weak-bundle", "score": 2}
            ]
        }

        loader = ContextLoader(settings)

        # Action
        newly_loaded = loader.process_message("test message", session)

        # Assertions
        assert "weak-bundle" not in newly_loaded
        assert "weak-bundle" not in session.loaded_bundles

    @patch("AgentQMS.tools.core.context.context_loader.ContextSuggester")
    def test_bundle_unloading(self, mock_suggester_class, settings, session):
        settings["context_integration"]["persistence_turns"] = 1
        settings["context_integration"]["enabled"] = True

        mock_suggester = mock_suggester_class.return_value
        loader = ContextLoader(settings)

        # Turn 1: High relevance
        mock_suggester.suggest.return_value = {
            "suggestions": [{"context_bundle": "test-bundle", "score": 10}]
        }
        with patch("AgentQMS.tools.core.context.context_loader.get_context_bundle") as mock_get_bundle:
            mock_get_bundle.return_value = []
            loader.process_message("relevance high", session)
            assert "test-bundle" in session.loaded_bundles
            assert session.bundle_last_used["test-bundle"] == 1

        # Turn 2: Low relevance (but persistence keeps it)
        mock_suggester.suggest.return_value = {
            "suggestions": [{"context_bundle": "test-bundle", "score": 2}]
        }
        loader.process_message("relevance low", session)
        assert "test-bundle" in session.loaded_bundles # Still loaded

        # Turn 3: Low relevance (persistence expired: turn 3 - last used 1 = 2 > 1)
        loader.process_message("relevance zero", session) # Turn 3
        # Wait, if score is 2, it is < threshold, so last_used is NOT updated.
        # last_used remains 1. Turn is 3. Diff is 2. Persistence is 1. UNLOAD.

        assert "test-bundle" not in session.loaded_bundles
        assert session.memory_footprint_mb == 0.0

    @patch("AgentQMS.tools.core.context.context_loader.get_context_bundle")
    def test_manual_overrides(self, mock_get_bundle, settings, session):
        mock_get_bundle.return_value = ["file.py"]
        loader = ContextLoader(settings)
        loader.turn_counter = 10

        # Force load
        success = loader.force_load_bundle("force-bundle", session)
        assert success
        assert "force-bundle" in session.loaded_bundles
        assert session.bundle_last_used["force-bundle"] == 10

        # Force unload
        success = loader.force_unload_bundle("force-bundle", session)
        assert success
        assert "force-bundle" not in session.loaded_bundles

    @patch("AgentQMS.tools.core.context.context_loader.ContextSuggester")
    def test_process_message_disabled(self, mock_suggester_class, session):

        settings = {"context_integration": {"enabled": False}}
        loader = ContextLoader(settings)

        newly_loaded = loader.process_message("test message", session)

        assert len(newly_loaded) == 0
        mock_suggester_class.return_value.suggest.assert_not_called()
