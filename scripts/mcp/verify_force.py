import sys
from AgentQMS.middleware.policies import ComplianceInterceptor, PolicyViolation

def test_force_override():
    print("Testing Force Override for ComplianceInterceptor...")
    interceptor = ComplianceInterceptor()

    # Mock a tool call with bad code
    code_with_violation = """
import sys
sys.path.insert(0, '/bad/path')
print('test')
    """

    try:
        # Pass force=True
        interceptor.validate("run_command", {
            "CommandLine": "python script.py",
            "CodeContent": code_with_violation,
            "force": True
        })
        print("✅ PASSED: Policy allowed violation when force=True")
    except PolicyViolation as e:
        print(f"❌ FAILED: Policy rejected even with force=True: {e.feedback_to_ai}")
        sys.exit(1)

if __name__ == "__main__":
    test_force_override()
