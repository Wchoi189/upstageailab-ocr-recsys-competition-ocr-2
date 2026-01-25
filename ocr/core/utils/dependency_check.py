"""
Central module for handling optional external dependencies.

Pattern: Wrap external imports in try/except to prevent import failures
from blocking core functionality.
"""
from typing import Optional, Any
import sys

# Track which dependencies are available
AVAILABLE_DEPS = {}

def safe_import(module_name: str, package_name: Optional[str] = None) -> Optional[Any]:
    """
    Safely import an optional dependency.
    
    Args:
        module_name: Python module name (e.g., 'tiktoken')
        package_name: PyPI package name if different (e.g., 'tiktoken')
    
    Returns:
        Module if available, None otherwise
    """
    if module_name in AVAILABLE_DEPS:
        return AVAILABLE_DEPS[module_name]
    
    try:
        module = __import__(module_name)
        AVAILABLE_DEPS[module_name] = module
        return module
    except ImportError:
        AVAILABLE_DEPS[module_name] = None
        pkg = package_name or module_name
        print(f"⚠️  Optional dependency '{pkg}' not available. Install with: uv pip install {pkg}", 
              file=sys.stderr)
        return None

def require_dependency(module_name: str, package_name: Optional[str] = None) -> Any:
    """
    Import a required dependency, raising error if not available.
    
    Args:
        module_name: Python module name
        package_name: PyPI package name if different
    
    Returns:
        Module object
    
    Raises:
        ImportError: If dependency is not available
    """
    module = safe_import(module_name, package_name)
    if module is None:
        pkg = package_name or module_name
        raise ImportError(
            f"Required dependency '{pkg}' not installed. "
            f"Install with: uv pip install {pkg}"
        )
    return module

# Common optional dependencies
def get_tiktoken() -> Optional[Any]:
    """Get tiktoken module if available (for token counting)."""
    return safe_import('tiktoken')

def get_boto3() -> Optional[Any]:
    """Get boto3 module if available (for AWS operations)."""
    return safe_import('boto3')

def get_deep_translator() -> Optional[Any]:
    """Get deep_translator module if available (for translations)."""
    return safe_import('deep_translator', 'deep-translator')

def check_all_optional_deps() -> dict[str, bool]:
    """
    Check availability of all known optional dependencies.
    
    Returns:
        Dict mapping dependency name to availability status
    """
    deps = {
        'tiktoken': get_tiktoken() is not None,
        'boto3': get_boto3() is not None,
        'deep_translator': get_deep_translator() is not None,
    }
    return deps

if __name__ == "__main__":
    print("Checking optional dependencies...")
    print("=" * 60)
    
    status = check_all_optional_deps()
    for dep, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"{status_icon} {dep:20s} {'Available' if available else 'Not installed'}")
    
    print("=" * 60)
    available_count = sum(status.values())
    print(f"{available_count}/{len(status)} optional dependencies available")
