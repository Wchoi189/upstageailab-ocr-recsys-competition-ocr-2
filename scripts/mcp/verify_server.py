
import sys
import asyncio





try:
    # # from scripts.mcp.unified_server import  # TODO: Update path  # TODO: Update path RESOURCES_CONFIG, URI_MAP, PATH_MAP, list_tools, read_resource
except ImportError as e:
    print(f"FAILED to import unified_server: {e}")
    sys.exit(1)

def verify():
    print("--- Verifying Resources ---")
    print(f"Total Resources: {len(RESOURCES_CONFIG)}")
    print(f"URI_MAP Size: {len(URI_MAP)}")
    print(f"PATH_MAP Size: {len(PATH_MAP)}")

    # Check for specific resources
    compass_uri = "compass://compass.json"
    if compass_uri in URI_MAP:
        print(f"✅ Found {compass_uri}")
    else:
        print(f"❌ Missing {compass_uri}")

    # Check for virtual resource (should have path=None and not crash)
    templates_uri = "agentqms://templates/list"
    if templates_uri in URI_MAP:
        res = URI_MAP[templates_uri]
        if res.get("path") is None:
             print(f"✅ Found virtual resource {templates_uri} with path=None")
        else:
             print(f"❌ Virtual resource {templates_uri} has unexpected path: {res.get('path')}")
    else:
        print(f"❌ Missing {templates_uri}")

    print("\n--- Verifying Tools ---")
    tools = asyncio.run(list_tools())
    print(f"Total Tools Loaded: {len(tools)}")

    # Check for a specific tool
    if any(t.name == "validate_artifact" for t in tools):
        print("✅ Found validate_artifact tool")
    else:
         print("❌ Missing validate_artifact tool")

    print("\n--- Verifying Read Resource (Logic) ---")
    # We won't actually call it to avoid side effects or I/O if possible,
    # but we can check if the function is defined and inspects the maps.
    # Actually, let's call it for a virtual resource to test the helper dispatch
    try:
        results = asyncio.run(read_resource(templates_uri))
        print(f"✅ Successfully read virtual resource: {len(results)} chunks")
    except Exception as e:
        print(f"❌ Failed to read virtual resource: {e}")

if __name__ == "__main__":
    verify()
