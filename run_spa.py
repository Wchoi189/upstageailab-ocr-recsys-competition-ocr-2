#!/usr/bin/env python3
"""
SPA Development Server Runner

Launches the FastAPI backend and optionally proxies to a frontend dev server.
"""

import subprocess
import sys
from pathlib import Path

from ocr.utils.path_utils import get_path_resolver, setup_project_paths

setup_project_paths()


def run_api_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Run the FastAPI playground API server."""
    from services.playground_api.app import app

    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def run_frontend_dev_server():
    """Run the frontend Vite dev server (if package.json exists)."""
    frontend_dir = get_path_resolver().config.project_root / "frontend"
    if not (frontend_dir / "package.json").exists():
        print("⚠️  Frontend package.json not found. Skipping frontend server.")
        print("   Create frontend/package.json and install dependencies to enable.")
        return

    print("🚀 Starting Vite dev server...")
    subprocess.run(["npm", "run", "dev"], cwd=frontend_dir, check=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the playground SPA development servers")
    parser.add_argument("--api-only", action="store_true", help="Only run the API server")
    parser.add_argument("--frontend-only", action="store_true", help="Only run the frontend dev server")
    parser.add_argument("--host", default="127.0.0.1", help="API server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable API server auto-reload")

    args = parser.parse_args()

    if args.frontend_only:
        run_frontend_dev_server()
    elif args.api_only:
        run_api_server(host=args.host, port=args.port, reload=not args.no_reload)
    else:
        print("🚀 Starting both API and frontend servers...")
        print("   API: http://127.0.0.1:8000")
        print("   Frontend: http://localhost:5173 (if configured)")
        print("\n   Use --api-only or --frontend-only to run separately.\n")

        # For now, just run API. Frontend can be started separately.
        run_api_server(host=args.host, port=args.port, reload=not args.no_reload)

