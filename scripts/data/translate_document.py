#!/usr/bin/env python3
"""
Document Translator using Upstage Solar API.

Translates Markdown documents from English to Korean while preserving structure.
Supports recursive translation of linked local Markdown files.
"""

import argparse
import os
import re
import re
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")


class DocumentTranslator:
    def __init__(self, api_key: str | None = None, model: str = "solar-pro", dry_run: bool = False, recursive: bool = False):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY environment variable is required.")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.upstage.ai/v1/solar",
        )
        self.model = model
        self.dry_run = dry_run
        self.recursive = recursive
        self.token_usage = {"input": 0, "output": 0}
        self.visited: set[Path] = set()
        self.link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")

    def normalize_artifact_name(self, file_path: Path) -> Path:
        """Normalize artifact filenames to match AgentQMS conventions.

        AgentQMS uses underscores as separators throughout artifact filenames.
        The descriptive part uses hyphens for word separation (kebab-case).

        Pattern: YYYY-MM-DD_HHMM_TYPE_ID_descriptive-name
        - Underscores separate structural parts (date, time, type, ID)
        - Hyphens separate words within the descriptive name

        Examples:
            - 2026-01-04_1730_BUG_003_config_precedence_leak.md
              -> 2026-01-04_1730_BUG_003_config-precedence-leak.md
            - 2026-01-03_1200_assessment_my_long_name.md
              -> 2026-01-03_1200_assessment_my-long-name.md
        """
        # Only normalize if this is an artifact file (in docs/artifacts/)
        if "docs/artifacts" not in str(file_path):
            return file_path

        # Parse the filename
        stem = file_path.stem

        # Split by underscores to find structural parts
        parts = stem.split("_")

        # Need at least: date, time, type
        if len(parts) < 3:
            return file_path

        date_part = parts[0]  # YYYY-MM-DD
        time_part = parts[1]  # HHMM
        type_part = parts[2]  # BUG, assessment, etc.

        # For bug reports: YYYY-MM-DD_HHMM_BUG_NNN_description
        if len(parts) >= 5 and type_part.upper() == "BUG" and parts[3].isdigit():
            bug_id = parts[3]
            # Everything after BUG_NNN: convert underscores to hyphens
            descriptive = "-".join(parts[4:])
            normalized_stem = f"{date_part}_{time_part}_{type_part}_{bug_id}_{descriptive}"
        # For other artifacts: YYYY-MM-DD_HHMM_type_description
        elif len(parts) >= 4:
            # Everything after type: convert underscores to hyphens
            descriptive = "-".join(parts[3:])
            normalized_stem = f"{date_part}_{time_part}_{type_part}_{descriptive}"
        else:
            return file_path

        return file_path.with_stem(normalized_stem)

    def update_links(self, content: str) -> str:
        """Update links in translated content to point to .ko.md files."""

        def replace_link(match):
            text = match.group(1)
            path = match.group(2)
            if path.startswith(("http", "https", "/")):
                return match.group(0)

            # If it looks like a local markdown file, rename extension
            if path.lower().endswith(".md") and not path.lower().endswith(".ko.md"):
                new_path = path[:-3] + ".ko.md"
                return f"[{text}]({new_path})"
            return match.group(0)

        return self.link_pattern.sub(replace_link, content)

    def translate_file(self, file_path: Path, output_path: Path) -> bool:
        file_path = file_path.resolve()
        if file_path in self.visited:
            return True
        self.visited.add(file_path)

        # Skip if it's already a translation file
        if file_path.name.endswith(".ko.md"):
            return True

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  ❌ Failed to read {file_path.name}: {e}")
            return False

        print(f"\n🌐 Translating {file_path.name}...")

        # If recursive, find valid links before translation
        child_files = []
        if self.recursive:
            child_files = self.extract_links(content, file_path)
            if child_files:
                print(f"   Found {len(child_files)} linked documents to process.")

        system_prompt = (
            "You are a professional technical translator specialized in AI and Software Engineering contexts. "
            "Translate the following Markdown document from English to Korean. "
            "Strictly follow these rules:\n"
            "1. Preserve ALL Markdown formatting, links, image tags, and code blocks exactly.\n"
            "2. Do NOT translate code inside code blocks.\n"
            "3. Do NOT translate URLs, file paths, or link targets in markdown links [text](URL). Only translate the link text.\n"
            "4. Do NOT translate frontmatter keys (e.g., 'title:', 'date:'), only translate values if they are text.\n"
            "5. Do NOT translate file names, directory names, or paths.\n"
            "6. Output ONLY the translated content, no conversational filler.\n"
            "7. Maintain a professional, clear tone."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
            )

            translated_content = response.choices[0].message.content.strip()

            if hasattr(response, "usage") and response.usage:
                self.token_usage["input"] += response.usage.prompt_tokens
                self.token_usage["output"] += response.usage.completion_tokens

            # Clean up potential markdown formatting from LLM
            if translated_content.startswith("```markdown"):
                translated_content = translated_content.replace("```markdown", "", 1)
            elif translated_content.startswith("```"):
                translated_content = translated_content.replace("```", "", 1)

            if translated_content.endswith("```"):
                translated_content = translated_content[:-3]

            # Post-processing: Update links in the translated content
            translated_content = self.update_links(translated_content)

            if self.dry_run:
                print("  [dry-run] Would write translated content.")
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(translated_content, encoding="utf-8")
                print(f"  ✅ Saved to {output_path.name}")

            # Recursively process children
            for child in child_files:
                # Calculate child output path (same dir as child, .ko.md extension)
                if child.name == "README.md":
                    child_output = child.parent / "README.ko.md"
                else:
                    child_output = self.normalize_artifact_name(child.with_suffix(".ko.md"))

                self.translate_file(child, child_output)

            return True

        except Exception as e:
            print(f"  ❌ API Error: {e}")
            return False

    def print_usage(self):
        print("\n📊 Token Usage:")
        print(f"   Input:  {self.token_usage['input']}")
        print(f"   Output: {self.token_usage['output']}")
        print(f"   Total:  {self.token_usage['input'] + self.token_usage['output']}")


def main():
    parser = argparse.ArgumentParser(description="Upstage Solar Document Translator")
    parser.add_argument("files", nargs="+", help="Files to translate")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively translate linked local markdown files")
    parser.add_argument("--model", type=str, default="solar-pro", help="Model to use")

    args = parser.parse_args()

    translator = DocumentTranslator(model=args.model, dry_run=args.dry_run, recursive=args.recursive)

    success = True
    for file_str in args.files:
        file_path = Path(file_str)
        if not file_path.exists():
            print(f"Warning: File {file_path} not found.")
            continue

        # Skip if it's already a translation file
        if file_path.name.endswith(".ko.md"):
            print(f"Skipping {file_path.name} (already a translation).")
            continue

        # Determine output path
        if file_path.name == "README.md":
            output_path = file_path.parent / "README.ko.md"
        else:
            output_path = translator.normalize_artifact_name(file_path.with_suffix(".ko.md"))

        if not translator.translate_file(file_path, output_path):
            success = False

    translator.print_usage()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
