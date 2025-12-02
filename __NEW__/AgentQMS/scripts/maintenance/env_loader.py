"""
Environment Loader Utility (minimal)

Loads environment variables from .env and .env.local, and returns UPSTAGE_API_KEY.
"""

import os
from pathlib import Path


class EnvLoader:
    _loaded: bool = False

    @staticmethod
    def load_env() -> None:
        if EnvLoader._loaded:
            return
        root = Path(__file__).resolve().parents[2]
        for name in [".env.local"]:
            print(f"Loading environment variables from {name}")
            path = root / name
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        EnvLoader._loaded = True

    @staticmethod
    def get_api_key(key_name: str = "UPSTAGE_API_KEY") -> str | None:
        EnvLoader.load_env()
        value = os.environ.get(key_name)
        if value:
            return value
        try:
            import streamlit as st  # optional fallback

            return st.secrets.get(key_name)  # type: ignore[attr-defined]
        except Exception:
            return None
