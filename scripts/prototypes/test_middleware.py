
from AgentQMS.middleware.policies import FileOperationInterceptor, PolicyViolation

def test_interceptor():
    interceptor = FileOperationInterceptor()

    # Test 1: Writing to forbidden config dir
    try:
        interceptor.validate("write_to_file", {
            "TargetFile": "/workspaces/project/AgentQMS/config/bad.yaml",
            "CodeContent": "foo: bar"
        })
        print("FAIL: Should have blocked AgentQMS/config write")
    except PolicyViolation as e:
        print(f"PASS: Blocked AgentQMS/config write: {e}")

    # Test 2: Writing valid standard
    try:
        interceptor.validate("write_to_file", {
            "TargetFile": "/workspaces/project/AgentQMS/standards/my-standard.yaml",
            "CodeContent": "foo: bar"
        })
        print("PASS: Allowed standard write")
    except PolicyViolation as e:
        print(f"FAIL: Blocked valid standard write: {e}")

if __name__ == "__main__":
    test_interceptor()
