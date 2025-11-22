#!/usr/bin/env python3
"""
Agent Tools Discovery Helper
Shows available tools and their locations
"""

from pathlib import Path


def show_tools():
    base_dir = Path(__file__).parent.parent  # Go up one level to agent_tools
    print("🔍 Available Agent Tools:")
    print()
    print("📁 Architecture:")
    print("   This directory contains the Implementation Layer.")
    print("   Agent interface layer: ../agent/")
    print("   See index.md for details.")
    print()

    categories = ["core", "compliance", "documentation", "utilities", "maintenance", "ocr"]
    category_descriptions = {
        "core": "Essential automation tools",
        "compliance": "Compliance and validation tools",
        "documentation": "Documentation management tools",
        "utilities": "Helper functions",
        "maintenance": "One-time fixes and maintenance",
        "ocr": "OCR-specific tools (checkpoint management, training, etc.)",
    }

    for category in categories:
        cat_dir = base_dir / category
        if cat_dir.exists():
            print(f"📁 {category.upper()}: {category_descriptions.get(category, '')}")
            tools = sorted([t for t in cat_dir.glob("*.py") if t.name != "__init__.py"])
            if tools:
                for tool in tools:
                    tool_path = f"python scripts/agent_tools/{category}/{tool.name}"
                    # Highlight artifact_workflow.py as critical for all artifacts
                    if tool.name == "artifact_workflow.py":
                        print(f"   ⭐ {tool_path}  # Use for ALL artifacts including summaries")
                    else:
                        print(f"   {tool_path}")
            else:
                print("   (no tools found)")
            print()

    # Show deprecated directory info
    deprecated_dir = base_dir / "_deprecated"
    if deprecated_dir.exists():
        deprecated_tools = list(deprecated_dir.glob("*.py"))
        if deprecated_tools:
            print("⚠️  _deprecated/: Legacy tools (not recommended for use)")
            print(f"   ({len(deprecated_tools)} deprecated scripts)")
            print()

    print("💡 Usage:")
    print("   For agents: cd agent/ && make help")
    print("   For humans: Use Python scripts directly")
    print("   See README.md for detailed usage information")
    print()
    print("⚠️  BEFORE CREATING NEW SCRIPTS:")
    print("   1. Check existing scripts above")
    print("   2. Update existing script instead of creating new one")
    print("   3. Pre-commit hook blocks duplicate scripts")
    print()


if __name__ == "__main__":
    show_tools()
