import sys
from AgentQMS.middleware.policies import ComplianceInterceptor, PolicyViolation

def test_insert_violation():
    print("Testing ComplianceInterceptor against sys.path.insert...")
    interceptor = ComplianceInterceptor()

    # Mock a tool call with bad code
    code_with_violation = """
import sys
sys.path.insert(0, '/bad/path')
print('test')
    """

    try:
        interceptor.validate("run_command", {"CommandLine": "python script.py", "CodeContent": code_with_violation})
        print("❌ FAILED: Policy allowed sys.path.insert")
        sys.exit(1)
    except PolicyViolation as e:
        print(f"✅ PASSED: Caught violation: {e.message}")
        print(f"   Feedback: {e.feedback_to_ai}")

if __name__ == "__main__":
    test_insert_violation()
