#!/usr/bin/env python3
"""
Grok-powered Artifact Fixer.

This script uses the xAI API to intelligently fix artifact violations that are too complex
for regex-based autofixers (e.g., incorrect categories, missing summaries, schema compliance).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from AgentQMS.tools.utils.paths import get_project_root
from AgentQMS.tools.compliance.validate_artifacts import ArtifactValidator

# Load environment variables from .env.local
try:
    load_dotenv(get_project_root() / ".env.local")
except Exception:
    pass


class GrokFixer:
    def __init__(self, api_key: str | None = None, model: str = "grok-4-1-fast-non-reasoning", dry_run: bool = False):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY environment variable is required.")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",
        )
        self.model = model
        self.dry_run = dry_run
        self.project_root = get_project_root()
        self.token_usage = {"input": 0, "output": 0}

    def run_validation(self) -> list[dict[str, Any]]:
        """Run artifact validation to identify issues."""
        print("🔍 Running validation...")
        validator = ArtifactValidator()
        return validator.validate_all()

    def filter_fixable_violations(self, violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter for violations that involve frontmatter or content issues suitable for LLM fixing."""
        fixable = []
        for item in violations:
            if item.get("valid"):
                continue

            # check if errors relate to frontmatter or semantic content
            errors = item.get("errors", [])
            is_relevant = any("Frontmatter" in e or "Category" in e or "Type" in e or "Schema" in e or "Summary" in e for e in errors)

            if is_relevant:
                fixable.append(item)
        return fixable

    def fix_artifact(self, file_path: Path, errors: list[str]) -> bool:
        """Attempt to fix a single artifact using Grok."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  ❌ Failed to read {file_path.name}: {e}")
            return False

        print(f"\n🤖 Fixing {file_path.name}...")
        for err in errors:
            print(f"    - {err}")

        prompt = f"""
You are an expert documentation maintainer for a software project.
I have a Markdown artifact that has failed validation check.
Please fix the frontmatter and content to comply with the rules.

ERRORS:
{json.dumps(errors, indent=2)}

FILE CONTENT:
```markdown
{content}
```

RULES:
1. Valid categories: development, architecture, evaluation, compliance, code_quality, reference, planning, research, troubleshooting, governance, meeting, security
2. Valid statuses: active, draft, completed, archived, deprecated, approved, deferred, pending, rejected
3. Valid types: implementation_plan, assessment, audit, design, research, template, bug_report, session_note, completion_summary, vlm_report, change_request, decision_record, meeting_notes, specification, ocr_experiment_report
4. Dates must be in 'YYYY-MM-DD HH:MM (KST)' format.
5. Frontmatter must start and end with '---'.

INSTRUCTIONS:
- Return ONLY the fully corrected markdown content.
- Do not add explanation text.
- Do not use markdown code blocks in your response (just return the raw text).
- fix the Frontmatter strictly.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant specialized in documentation compliance."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            if response.usage:
                self.token_usage["input"] += response.usage.prompt_tokens
                self.token_usage["output"] += response.usage.completion_tokens

            fixed_content = response.choices[0].message.content.strip()

            # Clean up potential markdown formatting from LLM
            if fixed_content.startswith("```markdown"):
                fixed_content = fixed_content.replace("```markdown", "", 1)
            if fixed_content.startswith("```"):
                fixed_content = fixed_content.replace("```", "", 1)
            if fixed_content.endswith("```"):
                fixed_content = fixed_content[:-3]

            fixed_content = fixed_content.strip()

            if self.dry_run:
                print("  [dry-run] Would write fixed content to file.")
                # print(f"Sample:\n{fixed_content[:200]}...")
            else:
                file_path.write_text(fixed_content, encoding="utf-8")
                print("  ✅ Fixed content written.")

            return True

        except Exception as e:
            print(f"  ❌ API Error: {e}")
            return False

    def process(self, limit: int = 5):
        violations = self.run_validation()
        fixable = self.filter_fixable_violations(violations)

        print(f"📋 Found {len(fixable)} LLM-fixable violations (Total violations: {len([v for v in violations if not v['valid']])})")

        count = 0
        for item in fixable:
            if count >= limit:
                break

            file_path = self.project_root / item["file"]
            if self.fix_artifact(file_path, item["errors"]):
                count += 1

        print(f"\n✨ Processed {count} files.")
        print("📊 Token Usage:")
        print(f"   Input:  {self.token_usage['input']}")
        print(f"   Output: {self.token_usage['output']}")
        print(f"   Total:  {self.token_usage['input'] + self.token_usage['output']}")


def main():
    parser = argparse.ArgumentParser(description="Grok-based Artifact Fixer")
    parser.add_argument("--limit", type=int, default=5, help="Max files to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--model", type=str, default="grok-4-1-fast-non-reasoning", help="Model to use")

    args = parser.parse_args()

    try:
        fixer = GrokFixer(model=args.model, dry_run=args.dry_run)
        fixer.process(limit=args.limit)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
