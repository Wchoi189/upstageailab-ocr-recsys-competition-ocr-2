#!/usr/bin/env python3
"""
Agent Tools Discovery Helper

Shows available tools and their locations in the AgentQMS framework.
This is the canonical implementation in agent_tools.
"""

from pathlib import Path


def show_tools():
    """Display all available agent tools organized by category."""
    base_dir = Path(__file__).parent.parent  # Go up to agent_tools
    print("🔍 Available Agent Tools:")
    print()
    print("📁 Architecture:")
    print("   Implementation Layer: AgentQMS/agent_tools/ (canonical)")
    print("   Legacy Shim Layer: AgentQMS/toolkit/ (compatibility only)")
    print("   Agent Interface: AgentQMS/interface/")
    print("   See AgentQMS/knowledge/meta/MAINTAINERS.md for details.")
    print()

    categories = ["core", "compliance", "documentation", "utilities", "audit"]
    category_descriptions = {
        "core": "Essential automation tools (artifact creation, context bundles)",
        "compliance": "Compliance and validation tools",
        "documentation": "Documentation management tools",
        "utilities": "Helper functions and tracking",
        "audit": "Audit framework tools",
    }

    for category in categories:
        cat_dir = base_dir / category
        if cat_dir.exists():
            print(f"📁 {category.upper()}: {category_descriptions.get(category, '')}")
            tools = sorted([t for t in cat_dir.glob("*.py") if t.name != "__init__.py"])
            if tools:
                for tool in tools:
                    print(f"   python AgentQMS/agent_tools/{category}/{tool.name}")
            else:
                print("   (no tools found)")
            print()

    print("💡 Usage:")
    print("   For agents: cd AgentQMS/interface/ && make help")
    print("   Direct CLI: PYTHONPATH=. python AgentQMS/agent_tools/<category>/<tool>.py")
    print("   See README.md for detailed usage information")
    print()


def main():
    """Main entry point."""
    show_tools()


if __name__ == "__main__":
    main()

