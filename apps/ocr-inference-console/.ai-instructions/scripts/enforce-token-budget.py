#!/usr/bin/env python3
"""Enforce token budget for AI documentation.

Ensures .ai-instructions/ stays under 1,000 tokens total.
"""

import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("⚠️  tiktoken not installed - skipping token budget check")
    print("   Install with: pip install tiktoken")
    sys.exit(0)

AI_DOCS = Path(__file__).parent.parent

# Token budgets per file type
BUDGETS = {
    "INDEX.yaml": 100,
    "quickstart.yaml": 150,
    "architecture/*.yaml": 200,
    "contracts/*.yaml": 200,
    "workflows/*.yaml": 150,
}

TOTAL_BUDGET = 1000


def count_tokens(text: str) -> int:
    """Count tokens using GPT-4 tokenizer."""
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))


def check_file_budget(file_path: Path, budget: int) -> bool:
    """Check if file is within token budget."""
    text = file_path.read_text()
    tokens = count_tokens(text)

    status = "✅" if tokens <= budget else "❌"
    print(f"{status} {file_path.name}: {tokens}/{budget} tokens")

    return tokens <= budget


def main():
    """Check token budgets for all files."""
    print("📊 Checking token budgets for .ai-instructions/\n")

    all_passed = True
    total_tokens = 0

    # Check INDEX.yaml
    index_file = AI_DOCS / "INDEX.yaml"
    if index_file.exists():
        tokens = count_tokens(index_file.read_text())
        total_tokens += tokens
        if not check_file_budget(index_file, BUDGETS["INDEX.yaml"]):
            all_passed = False

    # Check quickstart.yaml
    quickstart_file = AI_DOCS / "quickstart.yaml"
    if quickstart_file.exists():
        tokens = count_tokens(quickstart_file.read_text())
        total_tokens += tokens
        if not check_file_budget(quickstart_file, BUDGETS["quickstart.yaml"]):
            all_passed = False

    # Check architecture files
    for file in (AI_DOCS / "architecture").glob("*.yaml"):
        tokens = count_tokens(file.read_text())
        total_tokens += tokens
        if not check_file_budget(file, BUDGETS["architecture/*.yaml"]):
            all_passed = False

    # Check contracts files
    for file in (AI_DOCS / "contracts").glob("*.yaml"):
        tokens = count_tokens(file.read_text())
        total_tokens += tokens
        if not check_file_budget(file, BUDGETS["contracts/*.yaml"]):
            all_passed = False

    # Check workflows files
    if (AI_DOCS / "workflows").exists():
        for file in (AI_DOCS / "workflows").glob("*.yaml"):
            tokens = count_tokens(file.read_text())
            total_tokens += tokens
            if not check_file_budget(file, BUDGETS["workflows/*.yaml"]):
                all_passed = False

    # Check total budget
    print(f"\n📈 Total: {total_tokens}/{TOTAL_BUDGET} tokens")

    if total_tokens > TOTAL_BUDGET:
        print(f"❌ Exceeds total budget by {total_tokens - TOTAL_BUDGET} tokens")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All token budgets satisfied!")
        return 0
    else:
        print("❌ Token budget exceeded - make documentation more concise")
        return 1


if __name__ == "__main__":
    sys.exit(main())
