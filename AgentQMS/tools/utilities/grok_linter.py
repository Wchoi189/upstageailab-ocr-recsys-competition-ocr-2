#!/usr/bin/env python3
"""
Grok-Powered Linting Autofix Tool

Uses xAI's Grok API to intelligently fix linting errors reported by ruff.
Integrates with the AgentQMS workflow for automated code quality improvements.

Usage:
    python grok_linter.py --input lint_errors.json [--dry-run] [--limit N]

    # Or via ruff directly:
    ruff check . --output-format=json | python grok_linter.py --stdin
"""

import argparse
import json
from typing import Any
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import requests
import yaml


class CostTracker:
    """Track API usage costs for Grok API calls."""

    # Grok-4 pricing (as of 2026-01-06)
    INPUT_COST_PER_1M = 0.20  # $0.20 per 1M input tokens
    OUTPUT_COST_PER_1M = 0.50  # $0.50 per 1M output tokens

    def __init__(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = 0
        self.file_costs: list[dict[str, Any]] = []

    def record_usage(self, file_path: str, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a file."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.api_calls += 1
        self.file_costs.append(
            {"file": file_path, "input_tokens": input_tokens, "output_tokens": output_tokens, "timestamp": datetime.now().isoformat()}
        )

    def get_estimated_cost(self) -> float:
        """Calculate estimated cost in USD."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        return input_cost + output_cost

    def get_summary(self) -> dict:
        """Get cost summary as dictionary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "api_calls": self.api_calls,
            "estimated_cost_usd": round(self.get_estimated_cost(), 4),
            "file_costs": self.file_costs,
        }

    def save_report(self, path: str) -> None:
        """Save cost report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)


class GrokLinter:
    """AI-powered linting fix tool using xAI's Grok API."""

    DEFAULT_EXCLUSIONS = [
        "ocr/",
        "ocr-etl-pipeline/",
        "tests/ocr/",
        "runners/",
    ]

    def __init__(self, api_key: str, dry_run: bool = False, verbose: bool = False, config_path: str | None = None):
        self.api_key = api_key
        self.dry_run = dry_run
        self.verbose = verbose
        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-4-1-fast-non-reasoning"  # Grok-4 for cost efficiency
        self.fixes_applied = 0
        self.files_modified: list[str] = []
        self.files_excluded: list[str] = []
        self.cost_tracker = CostTracker()

        # Load configuration
        self.config = self._load_config(config_path)
        self.exclusions = self.config.get("exclusions", self.DEFAULT_EXCLUSIONS)
        self.max_cost = self.config.get("max_cost_per_run", 1.0)
        self.cost_report_path = self.config.get("cost_report_path", "/tmp/grok_linter_cost_report.json")

    def _load_config(self, config_path: str | None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try default location
            config_path = ".grok-linter.yaml"

        if not os.path.exists(config_path):
            return {}

        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.log(f"Warning: Could not load config from {config_path}: {e}", "WARN")
            return {}

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def should_exclude_file(self, file_path: str) -> bool:
        """Check if file should be excluded from AI fixes."""
        for pattern in self.exclusions:
            if pattern in file_path:
                self.log(f"⚠️  EXCLUDED (high-risk): {file_path} (matches '{pattern}')", "WARN")
                self.files_excluded.append(file_path)
                return True
        return False

    def group_errors_by_file(self, errors: list[dict]) -> dict[str, list[dict]]:
        """Group linting errors by file path."""
        grouped = defaultdict(list)
        for error in errors:
            file_path = error.get("filename", error.get("file"))
            if file_path:
                grouped[file_path].append(error)
        return dict(grouped)

    def call_grok_api(self, prompt: str, file_path: str, max_retries: int = 3) -> str | None:
        """Call Grok API with retry logic and cost tracking."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python code fixer. Your task is to fix linting errors "
                        "reported by ruff. Provide ONLY the corrected code without explanations. "
                        "Preserve all functionality and only fix the specific linting issues mentioned."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,  # Low temperature for consistent fixes
            "max_tokens": 4000,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()

                    # Extract token usage
                    usage = result.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)

                    # Record cost
                    self.cost_tracker.record_usage(file_path, input_tokens, output_tokens)

                    if self.verbose:
                        self.log(f"Tokens: {input_tokens} in, {output_tokens} out", "DEBUG")

                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    # Rate limit - exponential backoff
                    wait_time = 2**attempt
                    self.log(f"Rate limited. Waiting {wait_time}s...", "WARN")
                    time.sleep(wait_time)
                else:
                    self.log(f"API error {response.status_code}: {response.text}", "ERROR")
                    return None

            except Exception as e:
                self.log(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}", "ERROR")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        return None

    def create_fix_prompt(self, file_path: str, file_content: str, errors: list[dict]) -> str:
        """Create a prompt for Grok to fix the linting errors."""
        error_descriptions = []
        for error in errors:
            location = error.get("location", {})
            row = location.get("row", "?")
            col = location.get("column", "?")
            code = error.get("code", "")
            message = error.get("message", "")
            error_descriptions.append(f"- Line {row}, Col {col}: [{code}] {message}")

        errors_text = "\n".join(error_descriptions)

        prompt = f"""Fix the following linting errors in this Python file:

File: {file_path}

Linting Errors:
{errors_text}

Current Code:
```python
{file_content}
```

Provide the corrected code. Output ONLY the fixed Python code without any markdown formatting, explanations, or comments about the changes."""

        return prompt

    def apply_fix(self, file_path: str, fixed_content: str) -> bool:
        """Apply the fix to the file."""
        try:
            # Clean up the response - remove markdown code blocks if present
            if fixed_content.startswith("```python"):
                fixed_content = fixed_content.split("```python", 1)[1]
            if fixed_content.endswith("```"):
                fixed_content = fixed_content.rsplit("```", 1)[0]
            fixed_content = fixed_content.strip()

            if self.dry_run:
                self.log(f"[DRY-RUN] Would write to {file_path}", "INFO")
                if self.verbose:
                    print("\n" + "=" * 80)
                    print(f"Fixed content for {file_path}:")
                    print("=" * 80)
                    print(fixed_content)
                    print("=" * 80 + "\n")
                return True
            else:
                # Write the fixed content
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(fixed_content)
                    if not fixed_content.endswith("\n"):
                        f.write("\n")
                self.log(f"Applied fix to {file_path}", "SUCCESS")
                self.files_modified.append(file_path)
                return True

        except Exception as e:
            self.log(f"Failed to apply fix to {file_path}: {e}", "ERROR")
            return False

    def verify_fix(self, file_path: str) -> bool:
        """Verify the fix by running ruff on the file."""
        import subprocess

        try:
            result = subprocess.run(["ruff", "check", file_path], capture_output=True, text=True, timeout=10)
            # Exit code 0 means no errors
            return result.returncode == 0
        except Exception as e:
            self.log(f"Verification failed for {file_path}: {e}", "WARN")
            return False

    def process_file(self, file_path: str, errors: list[dict]) -> bool:
        """Process a single file with linting errors."""
        # Check if file should be excluded
        if self.should_exclude_file(file_path):
            return False

        self.log(f"Processing {file_path} ({len(errors)} errors)")

        # Read current file content
        try:
            with open(file_path, encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            self.log(f"Failed to read {file_path}: {e}", "ERROR")
            return False

        # Create fix prompt
        prompt = self.create_fix_prompt(file_path, file_content, errors)

        if self.verbose:
            self.log(f"Prompt length: {len(prompt)} chars", "DEBUG")

        # Call Grok API
        self.log(f"Calling Grok API for {file_path}...")
        fixed_content = self.call_grok_api(prompt, file_path)

        if not fixed_content:
            self.log(f"Failed to get fix from Grok for {file_path}", "ERROR")
            return False

        # Apply the fix
        if self.apply_fix(file_path, fixed_content):
            self.fixes_applied += 1

            # Verify if not in dry-run mode
            if not self.dry_run:
                if self.verify_fix(file_path):
                    self.log(f"✓ Verified: {file_path} now passes linting", "SUCCESS")
                else:
                    self.log(f"⚠ Warning: {file_path} may still have issues", "WARN")

            return True

        return False

    def run(self, errors: list[dict], limit: int | None = None) -> None:
        """Run the linting fix process."""
        self.log(f"Starting Grok Linter (dry_run={self.dry_run})")
        self.log(f"Exclusions active: {', '.join(self.exclusions)}")

        # Group errors by file
        grouped_errors = self.group_errors_by_file(errors)
        self.log(f"Found errors in {len(grouped_errors)} files")

        # Process each file
        files_to_process = list(grouped_errors.items())
        if limit:
            files_to_process = files_to_process[:limit]
            self.log(f"Limiting to {limit} files")

        for file_path, file_errors in files_to_process:
            self.process_file(file_path, file_errors)
            # Small delay to avoid rate limiting
            time.sleep(0.5)

            # Check cost limit
            current_cost = self.cost_tracker.get_estimated_cost()
            if current_cost > self.max_cost:
                self.log(f"⚠️  Cost limit reached: ${current_cost:.4f} > ${self.max_cost}", "WARN")
                self.log("Stopping to prevent excessive costs", "WARN")
                break

        # Cost summary
        cost_summary = self.cost_tracker.get_summary()

        # Summary
        self.log("=" * 80)
        self.log(f"Summary: {self.fixes_applied} files processed")
        if self.files_modified:
            self.log(f"Modified files: {', '.join(self.files_modified)}")
        if self.files_excluded:
            self.log(f"Excluded files (high-risk): {len(self.files_excluded)}")

        # Cost report
        self.log("\n" + "Cost Report:")
        self.log(f"  Total API calls: {cost_summary['api_calls']}")
        self.log(f"  Input tokens: {cost_summary['total_input_tokens']:,}")
        self.log(f"  Output tokens: {cost_summary['total_output_tokens']:,}")
        self.log(f"  Total tokens: {cost_summary['total_tokens']:,}")
        self.log(f"  Estimated cost: ${cost_summary['estimated_cost_usd']:.4f} USD")

        # Save cost report
        if self.config.get("save_cost_report", True):
            self.cost_tracker.save_report(self.cost_report_path)
            self.log(f"\nCost report saved to: {self.cost_report_path}")

        self.log("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-powered linting autofix using Grok")
    parser.add_argument("--input", type=str, help="Path to ruff JSON output file")
    parser.add_argument("--stdin", action="store_true", help="Read JSON from stdin")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just show what would be done")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--api-key", type=str, help="xAI API key (or set XAI_API_KEY env var)")
    parser.add_argument("--config", type=str, help="Path to config file (default: .grok-linter.yaml)")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not found. Set via --api-key or XAI_API_KEY environment variable.")
        sys.exit(1)

    # Load errors
    if args.stdin:
        errors = json.load(sys.stdin)
    elif args.input:
        with open(args.input) as f:
            errors = json.load(f)
    else:
        print("Error: Must specify --input or --stdin")
        parser.print_help()
        sys.exit(1)

    # Run the fixer
    linter = GrokLinter(api_key=api_key, dry_run=args.dry_run, verbose=args.verbose, config_path=args.config)
    linter.run(errors, limit=args.limit)


if __name__ == "__main__":
    main()
