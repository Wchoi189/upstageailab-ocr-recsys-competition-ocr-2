import shutil
import os
import sys
import argparse
import datetime
import json
import yaml
from pathlib import Path

# Import validation functions
try:
    from project_compass.src.validation import validate_session_content, validate_session_name
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COMPASS_DIR = PROJECT_ROOT / "project_compass"
ACTIVE_CONTEXT_DIR = COMPASS_DIR / "active_context"
HISTORY_DIR = COMPASS_DIR / "history"
SESSIONS_DIR = HISTORY_DIR / "sessions"
TASK_FILE = PROJECT_ROOT / "task.md"  # Assuming task.md is in root for agent use, but artifacts might be elsewhere.
# Re-checking task.md location: The user has artifacts in ~/.gemini/.../brain/.../task.md
# BUT, usually agents act on files in the workspace. The task.md artifact is a special case.
# Wait, the user's `task.md` in the context of "Project Compass" usually refers to a file AI Agents use.
# Let's check if there is a 'task.md' in the workspace or if we should rely on the artifact path.
# The user's request mentioned: "The work area needs to stay low-memory footprint... Any long content or many artifacts if there is any should be stored outside the project_compass directory."
# However, "context" usually implies the active tasks.
# Let's assume for now we only manage `active_context` files and maybe a `task.md` if it exists in a known location.
# Current session files: active_context/current_session.yml, active_context/blockers.yml.
# We will focus on `active_context` first.


def get_current_session_id():
    session_file = ACTIVE_CONTEXT_DIR / "current_session.yml"
    if not session_file.exists():
        return None
    try:
        with open(session_file) as f:
            data = yaml.safe_load(f)
            return data.get("session_id")
    except Exception:
        return None


def get_last_exported_session_id():
    """Retrieves the original_session_id from the most recent export."""
    if not SESSIONS_DIR.exists():
        return None

    sessions = [d for d in SESSIONS_DIR.iterdir() if d.is_dir()]
    if not sessions:
        return None

    # Sort by name (timestamped) to get latest
    latest_session = max(sessions, key=lambda p: p.name)

    manifest_path = latest_session / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path) as f:
            data = json.load(f)
            return data.get("original_session_id")
    except Exception:
        return None


def export_session(note=None, force=False):
    """Archives the current active context to history/sessions.

    CRITICAL FIX (2026-01-08): Creates timestamped copy of session_handover.md
    to preserve session-specific state. Previous implementation lost data by
    copying from a single overwritten file.

    UPDATE (2026-01-19): Added content and naming validation to prevent
    empty session exports and enforce naming standards.
    """
    session_id = get_current_session_id()
    if not session_id:
        print("Error: No active session found in active_context/current_session.yml")
        sys.exit(1)

    # NEW: Validate session name
    name_valid, name_error = validate_session_name(session_id)
    if not name_valid:
        print("❌ INVALID SESSION NAME:")
        print(f"  {name_error}")
        if not force:
            print("\n💡 Fix: Update session_id in active_context/current_session.yml")
            print("   Use format: domain-action-target (e.g., 'hydra-refactor-domains')")
            print("   Or use --force to bypass naming validation.")
            sys.exit(1)
        else:
            print("⚠️  Continuing due to --force flag")

    # NEW: Validate session content
    session_handover_path = COMPASS_DIR / "session_handover.md"
    current_session_path = ACTIVE_CONTEXT_DIR / "current_session.yml"

    content_valid, content_errors = validate_session_content(
        session_handover_path,
        current_session_path
    )

    if not content_valid:
        print("❌ EMPTY SESSION DETECTED:")
        for error in content_errors:
            print(f"  ✗ {error}")

        if not force:
            print("\n💡 Before exporting:")
            print("  1. Update session_handover.md with session summary")
            print("  2. Ensure current_session.yml has real objective")
            print("  3. Document key decisions and outcomes")
            print("\nUse --force to bypass content validation (not recommended).")
            sys.exit(1)
        else:
            print("⚠️  Exporting potentially empty session due to --force flag")

    if not force:
        last_id = get_last_exported_session_id()
        if last_id and last_id == session_id:
            print(f"❌ BLOCKING EXPORT: Stale Session ID detected ('{session_id}').")
            print("   This session ID matches the last exported session.")
            print("   Protocol Violation: You MUST update 'active_context/current_session.yml' before exporting.")
            print("   Use --force to override this check.")
            sys.exit(1)


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Folder name: YYYYMMDD_HHMMSS_session_id
    session_folder_name = f"{timestamp}_{session_id}"
    dest_dir = SESSIONS_DIR / session_folder_name

    if dest_dir.exists():
        print(f"Warning: Session directory {dest_dir} already exists.")
        # We proceed to overwrite or maybe fail? Unique timestamp prevents this usually.

    os.makedirs(dest_dir, exist_ok=True)

    # 1. Copy active_context
    shutil.copytree(ACTIVE_CONTEXT_DIR, dest_dir / "active_context", dirs_exist_ok=True)

    # 2. Copy session_handover.md with timestamp to preserve session state
    handover_file = COMPASS_DIR / "session_handover.md"
    if handover_file.exists():
        # Copy to timestamped filename to preserve this session's state
        timestamped_handover = dest_dir / f"session_handover_{timestamp}.md"
        shutil.copy2(handover_file, timestamped_handover)
        print(f"✓ Preserved session handover: session_handover_{timestamp}.md")
    else:
        print("Warning: No session_handover.md found. Creating placeholder.")
        placeholder = dest_dir / f"session_handover_{timestamp}.md"
        with open(placeholder, "w") as f:
            f.write(f"# Session Handover\n\nSession ID: {session_id}\nExported: {timestamp}\n\nNo handover document was created for this session.\n")

    # 3. Manifest
    manifest = {"original_session_id": session_id, "exported_at": timestamp, "note": note or ""}
    with open(dest_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Session exported to: {dest_dir}")

    # Auto-clear workspace after export
    _reset_workspace()

    return dest_dir


def list_sessions():
    if not SESSIONS_DIR.exists():
        print("No sessions found.")
        return

    sessions = sorted(SESSIONS_DIR.iterdir(), key=os.path.getmtime, reverse=True)
    print(f"{'TIMESTAMP & ID':<40} {'NOTE'}")
    print("-" * 60)
    for sess in sessions:
        if sess.is_dir():
            note = ""
            manifest_path = sess / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        m = json.load(f)
                        note = m.get("note", "")
                except:
                    pass
            print(f"{sess.name:<40} {note}")


def import_session(session_folder_name):
    """Restores a session from history/sessions to active_context."""
    src_dir = SESSIONS_DIR / session_folder_name
    if not src_dir.exists():
        # Try fuzzy match?
        print(f"Error: Session {session_folder_name} not found in {SESSIONS_DIR}")
        sys.exit(1)

    print(f"Restoring session from {src_dir}...")

    # Safety: Backup current if it looks important?
    # For now, we assume user knows what they are doing or we could do a quick auto-export.
    # Let's do a quick auto-backup to a 'trash' or 'temp' if needed, but user asked for simple.

    # Nuke current active_context content?
    # Or just overwrite. Overwrite is safer to keep untracked files, but technically we want a clean state.
    # Let's clean ACTIVE_CONTEXT_DIR first to avoid mixing.
    for item in ACTIVE_CONTEXT_DIR.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Copy back
    src_active = src_dir / "active_context"
    if src_active.exists():
        shutil.copytree(src_active, ACTIVE_CONTEXT_DIR, dirs_exist_ok=True)
        print("Restored active_context.")
    else:
        print("Warning: No active_context found in archived session.")

    # Copy back session_handover.md (find timestamped version)
    # Look for session_handover_*.md files
    handover_files = list(src_dir.glob("session_handover_*.md"))
    if handover_files:
        # Use the first (should only be one)
        src_handover = handover_files[0]
        shutil.copy2(src_handover, COMPASS_DIR / "session_handover.md")
        print(f"Restored session_handover.md from {src_handover.name}")
    else:
        print("Warning: No session_handover file found in archived session.")


def _reset_workspace():
    """Clears active context and initializes a fresh template."""
    print("Clearing active context...")
    if ACTIVE_CONTEXT_DIR.exists():
        for item in ACTIVE_CONTEXT_DIR.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        ACTIVE_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear session_handover.md
    handover_file = COMPASS_DIR / "session_handover.md"
    with open(handover_file, "w") as f:
        f.write("# Session Handover\n\nNo active session.\n")

    # Create template files
    session_file = ACTIVE_CONTEXT_DIR / "current_session.yml"
    today = datetime.date.today().isoformat()
    template = f"""---
# Project Compass Active Session
session_id: "new-session-{datetime.datetime.now().strftime('%H%M%S')}"
status: "active"
started_date: "{today}"
completed_date: null

objective: |
  [Enter objective here]

implementation_plan: null
source_walkthrough: null

phases: {{}}
"""
    with open(session_file, "w") as f:
        f.write(template)

    print(f"Created fresh session template at {session_file}")
    print("Session cleared. Ready for new context.")


def new_session(session_id=None):
    """Clears active context for a fresh start."""
    # Auto-export current before clearing
    current_id = get_current_session_id()
    if current_id:
        print(f"Auto-exporting current session {current_id}...")
        export_session(note="Auto-save before new session")
    else:
        # If no session, just reset
        _reset_workspace()


def main():
    parser = argparse.ArgumentParser(description="Project Compass Session Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export
    export_parser = subparsers.add_parser("export", help="Archive current session")
    export_parser.add_argument("--note", "-n", help="Note for the manifest")
    export_parser.add_argument("--force", "-f", action="store_true", help="Bypass stale session check")

    # Import
    import_parser = subparsers.add_parser("import", help="Restore a session")
    import_parser.add_argument("session_name", help="Name of the session folder to import")

    # List
    list_parser = subparsers.add_parser("list", help="List archived sessions")

    # New
    new_parser = subparsers.add_parser("new", help="Start fresh session (archives current)")

    args = parser.parse_args()

    if args.command == "export":
        export_session(args.note, args.force)
    elif args.command == "list":
        list_sessions()
    elif args.command == "import":
        import_session(args.session_name)
    elif args.command == "new":
        new_session()


if __name__ == "__main__":
    main()
