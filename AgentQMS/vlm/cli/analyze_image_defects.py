#!/usr/bin/env python3
"""CLI tool for analyzing image defects using VLM.

Main entry point for VLM image analysis from command line.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from AgentQMS.vlm.core.client import VLMClient
from AgentQMS.vlm.core.config import get_config
from AgentQMS.vlm.core.contracts import AnalysisMode, AnalysisRequest
from AgentQMS.vlm.integrations.reports import ReportIntegrator


def main():
    """Main CLI entry point."""
    config = get_config()

    parser = argparse.ArgumentParser(
        description="Analyze images for defects using Vision Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        nargs="+",
        help="Path(s) to image file(s) to analyze",
    )

    # Analysis mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["defect", "input", "compare", "full", "bug_001"],
        default="defect",
        help="Analysis mode (default: defect)",
    )

    # Optional arguments
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="Path to comparison image (for compare mode)",
    )
    parser.add_argument(
        "--via-annotations",
        type=Path,
        help="Path to VIA annotations JSON file",
    )
    parser.add_argument(
        "--initial-description",
        type=str,
        help="User's initial description of the image",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=Path,
        help="Path to few-shot examples JSON file",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Path to analysis template file",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )

    # Integration options
    parser.add_argument(
        "--auto-populate",
        action="store_true",
        help="Auto-populate incident report or assessment",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Experiment ID for integration",
    )
    parser.add_argument(
        "--incident-report",
        type=Path,
        help="Path to incident report to populate",
    )

    # Backend options
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openrouter", "solar_pro2", "cli"],
        help="Preferred backend (default: auto-select)",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=config.image.max_resolution,
        help=f"Maximum image resolution (default: {config.image.max_resolution})",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "compare" and not args.compare_with:
        parser.error("--compare-with is required for compare mode")

    # Create analysis request
    try:
        request = AnalysisRequest(
            mode=AnalysisMode(args.mode),
            image_paths=args.image,
            compare_with=args.compare_with,
            via_annotations=args.via_annotations,
            initial_description=args.initial_description,
            few_shot_examples=args.few_shot_examples,
            template=args.template,
            output_format=args.output_format,
            auto_populate=args.auto_populate,
            experiment_id=args.experiment_id,
            incident_report=args.incident_report,
            backend_preference=args.backend,
            max_resolution=args.max_resolution,
        )
    except Exception as e:
        print(f"Error: Invalid request parameters: {e}", file=sys.stderr)
        sys.exit(1)

    # Create VLM client
    try:
        client = VLMClient(backend_preference=args.backend)
    except Exception as e:
        print(f"Error: Failed to initialize VLM client: {e}", file=sys.stderr)
        sys.exit(1)

    # Perform analysis
    try:
        result = client.analyze(request)
    except Exception as e:
        print(f"Error: Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Helper: build metadata summary from result
    def _build_metadata(result_obj) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        # Base fields from result / metadata with safe fallbacks
        md = result_obj.metadata or {}
        image_paths = result_obj.image_paths or []
        primary_image = image_paths[0] if image_paths else None
        # Resolve absolute path when possible
        image_path_str = None
        image_id = None
        if primary_image is not None:
            try:
                image_path_str = str(primary_image.resolve())
            except Exception:
                image_path_str = str(primary_image)
            image_id = primary_image.name

        meta["type"] = f"{result_obj.mode.value}_analysis"
        meta["experiment_id"] = md.get("experiment_id", None)
        meta["image_path"] = image_path_str
        meta["image_id"] = image_id
        meta["model"] = md.get("model_name", None)
        meta["backend"] = result_obj.backend_used
        meta["timestamp"] = md.get("timestamp", None)
        meta["tags"] = md.get("tags", [])
        # Status can be filled by downstream processors; default unknown
        meta["status"] = md.get("status", "unknown")
        return meta

    # Format output
    if args.output_format == "json":
        output = json.dumps(result.model_dump(), indent=2)
    elif args.output_format == "markdown":
        meta = _build_metadata(result)
        # YAML frontmatter with absolute image path and model info
        frontmatter_lines = ["---"]
        for key, value in meta.items():
            if value is None:
                continue
            frontmatter_lines.append(f"{key}: {json.dumps(value)}")
        frontmatter_lines.append("---")
        frontmatter = "\n".join(frontmatter_lines)

        output = (
            f"{frontmatter}\n\n"
            f"# Analysis Result\n\n"
            f"**Mode:** {result.mode.value}\n"
            f"**Backend:** {result.backend_used}\n"
            f"**Processing Time:** {result.processing_time_seconds:.2f}s\n\n"
            f"## Analysis\n\n"
            f"{result.analysis_text}\n"
        )
    else:  # text
        meta = _build_metadata(result)
        header_lines = [
            f"Mode: {result.mode.value}",
            f"Backend: {result.backend_used}",
        ]
        if meta.get("model"):
            header_lines.append(f"Model: {meta['model']}")
        if meta.get("image_path"):
            header_lines.append(f"Image: {meta['image_path']}")
        if meta.get("timestamp"):
            header_lines.append(f"Timestamp: {meta['timestamp']}")
        header = "\n".join(header_lines)
        output = f"{header}\n\n{result.analysis_text}\n"

    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"Analysis saved to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Auto-populate report if requested
    if args.auto_populate and args.incident_report:
        try:
            integrator = ReportIntegrator()
            integrator.populate_report(args.incident_report, [result], args.via_annotations)
            print(f"Report populated: {args.incident_report}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to populate report: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
