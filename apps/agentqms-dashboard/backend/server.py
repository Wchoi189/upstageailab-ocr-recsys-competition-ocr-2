import os
import sys

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import local utils
# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add workspace root to path for AgentQMS imports
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, workspace_root)

import fs_utils
from routes import artifacts, compliance, system, tools, tracking

# Initialize FastAPI app
app = FastAPI(
    title="AgentQMS Dashboard Bridge",
    description="Backend bridge for AgentQMS Manager Dashboard",
    version="0.1.0"
)

# Include Routers
app.include_router(artifacts.router)
app.include_router(compliance.router)
app.include_router(system.router)
app.include_router(tracking.router)
app.include_router(tools.router)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class WriteRequest(BaseModel):
    path: str
    content: str

class ToolExecRequest(BaseModel):
    tool_id: str
    args: dict

@app.get("/status")
async def get_status():
    """Health check endpoint."""
    return {
        "status": "online",
        "version": "0.1.0",
        "cwd": os.getcwd(),
        "agentqms_root": os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    }

@app.get("/fs/list")
async def list_files(path: str = Query(..., description="Path to list files from")):
    """List files in a directory."""
    try:
        # Security check: prevent escaping root if necessary, but for now allow relative
        full_path = os.path.abspath(path)
        return {"path": path, "items": fs_utils.list_files(full_path)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Directory not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fs/read")
async def read_file(path: str = Query(..., description="Path to read file from")):
    """Read file content."""
    try:
        full_path = os.path.abspath(path)
        content = fs_utils.read_file(full_path)
        return {"path": path, "content": content, "encoding": "utf-8"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fs/write")
async def write_file(request: WriteRequest):
    """Write content to a file."""
    try:
        # Implement write logic in fs_utils or here
        # For now, simple write
        with open(request.path, 'w', encoding='utf-8') as f:
            f.write(request.content)
            bytes_written = f.tell()
        return {"success": True, "bytes_written": bytes_written}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/exec")
async def execute_tool(request: ToolExecRequest):
    """Execute an AgentQMS tool via make command."""
    import subprocess

    # Map tool IDs to make targets
    tool_commands = {
        "validate": ["make", "-C", "AgentQMS/interface", "validate"],
        "compliance": ["make", "-C", "AgentQMS/interface", "compliance"],
        "boundary": ["make", "-C", "AgentQMS/interface", "boundary"],
    }

    if request.tool_id not in tool_commands:
        return {
            "tool_id": request.tool_id,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"Unknown tool_id: {request.tool_id}. Available: {list(tool_commands.keys())}"
        }

    try:
        result = subprocess.run(
            tool_commands[request.tool_id],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.abspath(".")
        )

        return {
            "tool_id": request.tool_id,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "tool_id": request.tool_id,
            "exit_code": 124,
            "stdout": "",
            "stderr": "Tool execution timed out after 30 seconds"
        }
    except Exception as e:
        return {
            "tool_id": request.tool_id,
            "exit_code": 1,
            "stdout": "",
            "stderr": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
