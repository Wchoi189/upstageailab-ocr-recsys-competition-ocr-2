from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from AgentQMS.tools.core.context_loader import ContextLoader, SessionState

# Use /api/v1 prefix to match system.py if needed,
# but usually prefix is defined in the router or include.
# system.py has explicit prefix. We will follow that.
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Mock session storage - in production this would be Redis/DB
# Key: session_id, Value: SessionState
sessions: dict[str, SessionState] = {}

# Initialize loader with default settings or load from actual settings
loader = ContextLoader({
    "context_integration": {
        "enabled": True,
        "auto_load_threshold": 5,
        "analytics_enabled": True
    }
})

class ContextRequest(BaseModel):
    session_id: str
    message: str
    force_reload: bool | None = False

@router.post("/process_context")
async def process_context(data: ContextRequest):
    """
    Process a chat message to detect and load relevant context bundles.
    Does NOT generate a response, only manages context state.
    """
    # Create session if not exists
    if data.session_id not in sessions:
        sessions[data.session_id] = SessionState()

    session = sessions[data.session_id]

    if data.force_reload:
        # Clear session loaded bundles (optional logic)
        session.loaded_bundles.clear()

    # Process message through ContextLoader
    newly_loaded = loader.process_message(data.message, session)

    return {
        "session_id": data.session_id,
        "loaded_bundles": list(session.loaded_bundles),
        "newly_loaded": newly_loaded,
        "memory_footprint_mb": session.memory_footprint_mb,
        "turn_counter": loader.turn_counter
    }

@router.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """Get current context state for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return {
        "session_id": session_id,
        "loaded_bundles": list(session.loaded_bundles),
        "memory_footprint_mb": session.memory_footprint_mb,
        "message_history_length": len(session.message_history)
    }

@router.post("/session/{session_id}/bundle/{bundle_name}")
async def force_load_bundle(session_id: str, bundle_name: str):
    """Manually force load a bundle."""
    if session_id not in sessions:
        sessions[session_id] = SessionState()

    session = sessions[session_id]
    success = loader.force_load_bundle(bundle_name, session)

    return {
        "success": success,
        "bundle": bundle_name,
        "loaded_bundles": list(session.loaded_bundles)
    }
