import sys
from AgentQMS.middleware.policies import StandardsInterceptor, PolicyViolation

def test_standards_compliance():
    print("Testing StandardsInterceptor...")
    interceptor = StandardsInterceptor()

    # Happy Path: Valid ADS content
    valid_content = """
ads_version: "1.0"
type: rule_set
agent: all
tier: 2
priority: medium
validates_with: something
compliance_status: unknown
memory_footprint: 100
    """
    try:
        interceptor.validate("write_to_file", {
            "TargetFile": "/path/to/AgentQMS/standards/my-standard.yaml",
            "CodeContent": valid_content
        })
        print("✅ Valid content passed.")
    except PolicyViolation:
        print("❌ Valid content FAILED (Unexpected)")
        sys.exit(1)

    # Bad Path: Missing keys
    bad_content = """
title: Some Standard
version: 1.0
    """
    try:
        interceptor.validate("write_to_file", {
            "TargetFile": "/path/to/AgentQMS/standards/my-standard.yaml",
            "CodeContent": bad_content
        })
        print("❌ Invalid content PASSED (Policy failed)")
        sys.exit(1)
    except PolicyViolation as e:
        print(f"✅ Invalid content correctly REJECTED: {str(e)}")

    # Force Override
    try:
        interceptor.validate("write_to_file", {
            "TargetFile": "/path/to/AgentQMS/standards/my-standard.yaml",
            "CodeContent": bad_content,
            "force": True
        })
        print("✅ Force override passed.")
    except PolicyViolation:
        print("❌ Force override FAILED")
        sys.exit(1)


if __name__ == "__main__":
    test_standards_compliance()
