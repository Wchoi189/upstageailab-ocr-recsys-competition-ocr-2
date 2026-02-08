import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add backend to path
from ocr.core.utils.path_utils import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT / "apps" / "agentqms-dashboard" / "backend"))  # noqa: path-hack

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
