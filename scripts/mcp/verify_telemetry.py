#!/usr/bin/env python3
"""
Verify Telemetry Middleware Integration
Tests both Redundancy and Compliance policies.
"""
import sys
import shutil
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from AgentQMS.middleware.telemetry import PolicyViolation
from AgentQMS.middleware.policies import RedundancyInterceptor, ComplianceInterceptor

def test_compliance_policy():
    print("Testing ComplianceInterceptor...")
    interceptor = ComplianceInterceptor()

    # 1. Test Python execution violation
    try:
        interceptor.validate("run_command", {"CommandLine": "python script.py"})
        print("❌ Failed to detect 'python script.py'")
        sys.exit(1)
    except PolicyViolation as e:
        print(f"✅ Detected python violation: {e.feedback_to_ai}")
        assert "uv run python" in e.feedback_to_ai

    # 2. Test valid uv run
    try:
        interceptor.validate("run_command", {"CommandLine": "uv run python script.py"})
        print("✅ Correctly allowed 'uv run python'")
    except PolicyViolation:
        print("❌ Falsely flagged 'uv run python'")
        sys.exit(1)

    # 3. Test sys.path violation
    try:
        interceptor.validate("write_to_file", {"CodeContent": "import sys; sys.path.append('..')"})
        print("❌ Failed to detect sys.path.append")
        sys.exit(1)
    except PolicyViolation as e:
        print(f"✅ Detected sys.path violation: {e.feedback_to_ai}")

def test_redundancy_policy():
    print("\nTesting RedundancyInterceptor...")
    interceptor = RedundancyInterceptor()

    # Setup mock .gemini directory
    gemini_dir = project_root / ".gemini"
    gemini_dir.mkdir(exist_ok=True)

    # Mock an existing implementation plan in some subdirectory
    # e.g. .gemini/brain/uuid/implementation_plan.md
    mock_brain = gemini_dir / "brain" / "mock-uuid"
    mock_brain.mkdir(parents=True, exist_ok=True)
    (mock_brain / "implementation_plan.md").touch()

    try:
        # Trigger redundancy check
        interceptor.validate("create_artifact", {"artifact_type": "implementation_plan"})
        print("❌ Failed to detect redundant implementation plan")
        # cleanup
        shutil.rmtree(gemini_dir)
        sys.exit(1)
    except PolicyViolation as e:
        print(f"✅ Detected redundancy: {e.feedback_to_ai}")
        assert "already managed" in e.feedback_to_ai

    # Clean up
    # We don't want to delete real .gemini if it existed before, but for this env it likely didn't
    # or we should be careful.
    # For safety in this test, we only delete the test file we created.
    (mock_brain / "implementation_plan.md").unlink()
    if not any(mock_brain.iterdir()):
        mock_brain.rmdir()

def main():
    try:
        test_compliance_policy()
        test_redundancy_policy()
        print("\n🎉 All middleware tests passed!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
