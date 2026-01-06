import sys
from unittest.mock import MagicMock
import pytest

from omegaconf import OmegaConf

from ocr.utils.config_utils import ensure_dict, is_config


@pytest.fixture(scope="module", autouse=True)
def mock_heavy_imports():
    """Mock dependencies to avoid importing heavy model libraries."""
    # Store original modules
    original_modules = {}
    modules_to_mock = ["ocr.models.core", "ocr.models.core.registry"]

    for module_name in modules_to_mock:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]

    # Apply mocks
    mock_registry = MagicMock()
    sys.modules["ocr.models.core"] = MagicMock()
    sys.modules["ocr.models.core.registry"] = mock_registry

    yield

    # Cleanup: restore original modules
    for module_name in modules_to_mock:
        if module_name in original_modules:
            sys.modules[module_name] = original_modules[module_name]
        else:
            sys.modules.pop(module_name, None)



def test_is_config():
    # Test valid configs
    assert is_config({})
    assert is_config(OmegaConf.create({}))
    assert is_config({"a": 1})
    assert is_config(OmegaConf.create({"a": 1}))

    # Test invalid configs
    assert not is_config([])
    assert not is_config(OmegaConf.create([]))
    assert not is_config("string")
    assert not is_config(123)
    assert not is_config(None)

def test_ensure_dict_simple():
    cfg = OmegaConf.create({"a": 1, "b": "test"})
    res = ensure_dict(cfg)
    assert isinstance(res, dict)
    assert res == {"a": 1, "b": "test"}

def test_ensure_dict_nested():
    cfg = OmegaConf.create({"a": {"b": 2}})
    res = ensure_dict(cfg)
    assert isinstance(res, dict)
    assert isinstance(res["a"], dict)
    assert res == {"a": {"b": 2}}

def test_ensure_dict_list():
    cfg = OmegaConf.create({"a": [1, 2, {"c": 3}]})
    res = ensure_dict(cfg)
    assert isinstance(res, dict)
    assert isinstance(res["a"], list)
    assert res["a"][2] == {"c": 3}
    assert isinstance(res["a"][2], dict)

def test_ensure_dict_idempotent():
    # Calling ensure_dict on a dict should return a dict
    native = {"a": 1}
    assert ensure_dict(native) == native
    assert ensure_dict(native) is not native # Should copy? actually my implementation might not copy if simple dict recursion creates new dict.
    # Let's check implementation behavior:
    # if isinstance(cfg, dict): return {k: ensure_dict(v) ...} -> creates new dict.

    # Check ListConfig
    lst = OmegaConf.create([1, 2])
    res_lst = ensure_dict(lst)
    assert isinstance(res_lst, list)
    assert res_lst == [1, 2]

def test_ensure_dict_mixed():
    # Mixed native dict and DictConfig
    mixed = {"a": OmegaConf.create({"b": 1})}
    res = ensure_dict(mixed)
    assert isinstance(res["a"], dict)
    assert res["a"]["b"] == 1
