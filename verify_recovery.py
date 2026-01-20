
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
COMPASS_DIR = PROJECT_ROOT / "project_compass"
ACTIVE_CONTEXT_DIR = COMPASS_DIR / "active_context"
SESSION_HANDOVER = COMPASS_DIR / "session_handover.md"
CURRENT_SESSION = ACTIVE_CONTEXT_DIR / "current_session.yml"

def test_recovery():
    print("🧪 Starting Session Recovery Test")

    # 1. Setup Mock Environment
    if not COMPASS_DIR.exists():
        COMPASS_DIR.mkdir()
    if not ACTIVE_CONTEXT_DIR.exists():
        ACTIVE_CONTEXT_DIR.mkdir()

    # Create valid session_handover.md
    handover_content = """# Session Handover
```yaml
session_id: test-recovery-session-01
objective:
  primary_goal: Testing auto-recovery
```
Some content here.
"""
    with open(SESSION_HANDOVER, "w") as f:
        f.write(handover_content)

    # Ensure current_session.yml is DELETED
    if CURRENT_SESSION.exists():
        CURRENT_SESSION.unlink()

    print("✅ Setup complete: session_handover.md exists, current_session.yml deleted.")

    # 2. Run Export (using the library directly to avoid subprocess complexity for now)
    # We need to import the modified script.
    # Since it's in scripts/session_manager.py, we add it to path.
    sys.path.append(str(PROJECT_ROOT / "project_compass" / "scripts"))
    import session_manager

    # Mock sys.exit to catch it if it fails
    original_exit = sys.exit
    def mock_exit(code):
        if code != 0:
            raise RuntimeError(f"sys.exit called with code {code}")
    sys.exit = mock_exit

    try:
        # We use force=True to bypass extensive validation which might require more mocking
        # The goal is to test the RECOVERY logic, which happens *before* validation.
        # However, validation logic *also* reads current_session.yml.
        # So if recovery works, validation should pass (or at least find the file).

        print("🚀 Running export_session()...")
        # Note: We can't easily mock the 'shutil.copytree' calls completely without more work,
        # but the script will likely try to export to history/sessions.
        # We should clean that up later or accept it.

        session_manager.export_session(note="Automated Test", force=True)
        print("✅ Export function returned successfully.")

        # 3. Verify Recovery
        if CURRENT_SESSION.exists():
            print("✅ VERIFIED: current_session.yml was automatically restored.")
            with open(CURRENT_SESSION) as f:
                content = f.read()
                if "test-recovery-session-01" in content:
                     print("✅ VERIFIED: Correct session_id in restored file.")
                else:
                     print("❌ FAILED: Incorrect content in restored file.")
        else:
            print("❌ FAILED: current_session.yml was NOT restored.")

    except Exception as e:
        print(f"❌ Test Failed: {e}")
    finally:
        sys.exit = original_exit
        # Cleanup if needed?
        # Maybe leave artifacts for inspection.
        pass

if __name__ == "__main__":
    test_recovery()
