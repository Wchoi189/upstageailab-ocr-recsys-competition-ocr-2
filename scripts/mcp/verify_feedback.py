
import asyncio
import sys




try:
    from scripts.mcp.unified_server import call_tool
    from AgentQMS.tools.utils.system.paths import get_project_root
except ImportError as e:
    print(f"FAILED to import required modules: {e}")
    sys.exit(1)

async def verify():
    print("--- Verifying log_process_feedback ---")
    args = {
        "severity": "low",
        "category": "tool_friction",
        "observation": "Test observation from verification script",
        "suggestion": "Test suggestion"
    }

    try:
        result = await call_tool("log_process_feedback", args)
        print(f"Result: {result[0].text}")
    except Exception as e:
        print(f"❌ call_tool failed: {e}")
        return

    # Check file content
    # Use standard path resolution from AgentQMS
    project_root = get_project_root()
    feedback_file = project_root / "project_compass/feedback_log.md"
    if feedback_file.exists():
        content = feedback_file.read_text()
        if "Test observation" in content:
            print("✅ Feedback logged to file successfully")
        else:
            print("❌ Feedback file exists but content missing")
            print(f"Content: {content}")
    else:
         print("❌ Feedback file not created")

if __name__ == "__main__":
    asyncio.run(verify())
