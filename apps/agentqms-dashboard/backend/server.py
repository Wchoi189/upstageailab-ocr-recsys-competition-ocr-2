from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Initialize FastAPI app
app = FastAPI(
    title="AgentQMS Dashboard Bridge",
    description="Backend bridge for AgentQMS Manager Dashboard",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def get_status():
    """Health check endpoint."""
    return {
        "status": "online",
        "version": "0.1.0",
        "cwd": os.getcwd(),
        "agentqms_root": os.path.abspath(os.path.join(os.getcwd(), "../../..")) # Approximate root
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
