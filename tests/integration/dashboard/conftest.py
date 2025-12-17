import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../apps/agentqms-dashboard/backend")))

from server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def clean_artifacts():
    """Cleanup artifacts created during tests."""
    yield
    # Cleanup logic if needed, e.g. delete test files
    # For now, we rely on the 'delete' endpoint or manual cleanup in tests
    pass
