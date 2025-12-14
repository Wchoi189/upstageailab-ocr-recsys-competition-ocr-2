"""Tracking database status endpoint."""
import os
import sys

from fastapi import APIRouter, Query

# Ensure AgentQMS is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

router = APIRouter(prefix="/api/v1/tracking", tags=["tracking"])


@router.get("/status")
async def get_tracking_status(kind: str = Query("all", description="Kind: plan, experiment, debug, refactor, or all")):
    """Get tracking database status for plans, experiments, debug sessions, or refactors."""
    try:
        from AgentQMS.agent_tools.utilities.tracking.query import get_status

        status_text = get_status(kind)
        return {
            "kind": kind,
            "status": status_text,
            "success": True
        }
    except Exception as e:
        return {
            "kind": kind,
            "status": "",
            "success": False,
            "error": str(e)
        }
