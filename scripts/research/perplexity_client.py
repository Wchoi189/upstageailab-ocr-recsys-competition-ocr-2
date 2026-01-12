#!/usr/bin/env python3
"""
Perplexity API Client
---------------------
A simple CLI tool to query the Perplexity API for research purposes.
Results are printed to stdout and optionally saved to data/research.

Usage:
    uv run python scripts/research/perplexity_client.py --query "Your research question"
    uv run python scripts/research/perplexity_client.py -q "Question" -m llama-3.1-sonar-huge-128k-online
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import requests

# Try to load .env.local if available
try:
    from dotenv import load_dotenv

    load_dotenv(".env.local")
except ImportError:
    pass

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

DEFAULT_MODEL = "sonar"
API_URL = "https://api.perplexity.ai/chat/completions"


def query_perplexity(query: str, model: str = DEFAULT_MODEL) -> str:
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY environment variable is not set. Please check .env.local")

    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful and precise research assistant. Provide detailed, well-cited answers in Markdown format.",
            },
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract content from response
        content = data["choices"][0]["message"]["content"]
        # Citations are usually available in the 'citations' field of the response body if supported by the model/endpoint
        # But for chat completions, they are often embedded or returned separately.
        # The standard chat completion format puts the text in message.content.

        return content

    except requests.exceptions.RequestException as e:
        print(f"Error querying Perplexity API: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print(f"Details: {e.response.text}", file=sys.stderr)
        sys.exit(1)


def save_research(query: str, content: str, output_dir: str = "data/research"):
    """Saves the research result to a markdown file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize query for filename (first 30 chars, alphanumeric only-ish)
    slug = "".join(x for x in query[:40] if x.isalnum() or x in " -_").strip().replace(" ", "_")
    filename = f"{timestamp}_{slug}.md"

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Research Query: {query}\n")
        f.write(f"**Date:** {datetime.datetime.now()}\n\n")
        f.write("---\n\n")
        f.write(content)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Query Perplexity API for research.")
    parser.add_argument("-q", "--query", required=True, help="The research question to ask.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-save", action="store_true", help="Do not save the output to a file.")

    args = parser.parse_args()

    print(f"🔍 Researching: {args.query} ...", file=sys.stderr)

    result = query_perplexity(args.query, args.model)

    print("\n" + "=" * 80 + "\n")
    print(result)
    print("\n" + "=" * 80 + "\n")

    if not args.no_save:
        filepath = save_research(args.query, result)
        print(f"📝 Result saved to: {filepath}", file=sys.stderr)


if __name__ == "__main__":
    main()
