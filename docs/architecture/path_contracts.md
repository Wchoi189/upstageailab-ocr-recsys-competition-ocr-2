# Path Management Standards

This document outlines the professional standards for handling file paths, configuration, and environment variables within the project. Adhering to these standards ensures portability, testability, and adherence to the [12-Factor App](https://12factor.net/) methodology.

## 1. The Golden Rule: No Hardcoded Paths

**Anti-Pattern:**
```python
# ❌ BAD: Hardcoded assumption about where code lives
DATA_DIR = Path(__file__).parent.parent / "data"
LOG_DIR = "/var/log/myapp"
```

**Standard:**
All structural paths (data, logs, configuration) must be **injectable** via configuration or environment variables.

```python
# ✅ GOOD: defaulted to a sane relative path, but overridable
class Settings(BaseSettings):
    data_dir: Path = Field(default=Path("data"), env="DATA_DIR")
```

## 2. Configuration Management

We use `pydantic-settings` to manage configuration. This provides type safety, validation, and easy environment variable overrides.

### Example Implementation

```python
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    # Application Root (defaults to current working dir)
    app_root: Path = Field(default=Path.cwd(), env="APP_ROOT")

    # Derived Paths
    @property
    def data_dir(self) -> Path:
        return self.app_root / "data"

    @property
    def log_dir(self) -> Path:
        return self.app_root / "logs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = AppSettings()
```

## 3. Project Root Resolution

When you absolutely must resolve the project root dynamically (e.g., in a script or tool), use a robust discovery mechanism rather than fragile parent chains actions.

**Anti-Pattern:**
```python
# ❌ Fragile: Breaks if file moves depth
ROOT = Path(__file__).parent.parent.parent
```

**Standard:**
Use a marker file (like `pyproject.toml` or `.git`) to locate the root.

```python
def get_project_root() -> Path:
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (missing pyproject.toml)")
```

## 4. Docker & Deployment Contracts

Containers should not rely on the structure of the host machine. We use **Volume Mounts** and standard internal paths.

-   **Internal Path**: `/app` or `/workspace`
-   **Data Mount**: `/app/data`
-   **Log Mount**: `/app/logs`

The application code inside the container simply reads `DATA_DIR` (env var) or defaults to `./data`. It does not care if that maps to `/mnt/ssd/data` or `C:\Users\Data` on the host.

## 5. Summary of Best Practices

1.  **Use `pathlib.Path`** exclusively (no `os.path.join`).
2.  **Define paths in one place** (e.g., `src/config.py` or `system/paths.py`).
3.  **Prioritize Environment Variables** for overrides.
4.  **Use Relative Paths** by default for development convenience.
5.  **Validate Paths** on startup (fail early if a required directory is missing).
