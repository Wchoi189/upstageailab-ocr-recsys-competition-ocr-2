"""Tools execution endpoint."""
import os
import subprocess

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/tools", tags=["tools"])


class ToolExecRequest(BaseModel):
    tool_id: str
    args: dict


@router.post("/exec")
async def execute_tool(request: ToolExecRequest):
    """Execute an AgentQMS tool via make command."""
    # Map tool IDs to make targets
    tool_commands = {
        "validate": ["make", "-C", "AgentQMS/interface", "validate"],
        "compliance": ["make", "-C", "AgentQMS/interface", "compliance"],
        "boundary": ["make", "-C", "AgentQMS/interface", "boundary"],
        "discover": ["make", "-C", "AgentQMS/interface", "discover"],
        "status": ["make", "-C", "AgentQMS/interface", "status"],
    }

    if request.tool_id not in tool_commands:
        return {
            "success": False,
            "error": f"Unknown tool: {request.tool_id}",
            "output": ""
        }

    try:
        # Get workspace root (4 levels up from backend/routes/tools.py)
        # backend/routes/tools.py -> backend -> agentqms-dashboard -> apps -> workspace
        workspace_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..")
        )

        cmd = tool_commands[request.tool_id]

        # Verify workspace root exists
        agentqms_path = os.path.join(workspace_root, "AgentQMS/interface")
        if not os.path.exists(agentqms_path):
            return {
                "success": False,
                "error": f"AgentQMS path not found at {agentqms_path}",
                "output": ""
            }

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=workspace_root  # Run from workspace root
        )

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Tool execution timed out",
            "output": ""
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output": ""
        }
