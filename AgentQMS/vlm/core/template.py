"""Image Analysis Template Framework.

Structured framework for organizing image references and analysis results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_qms.vlm.core.contracts import AnalysisResult, VIAAnnotation
from agent_qms.vlm.utils.paths import get_path_resolver


class ImageAnalysisTemplate:
    """Template for organizing image analysis results."""

    def __init__(self, title: str = "Image Analysis"):
        """Initialize template.

        Args:
            title: Template title
        """
        self.title = title
        self.images: List[Dict[str, Any]] = []
        self.few_shot_examples: List[Dict[str, Any]] = []
        self.analysis_results: List[AnalysisResult] = []
        self.metadata: Dict[str, Any] = {
            "created": datetime.now().isoformat(),
        }

    def add_image(
        self,
        image_id: str,
        image_path: Path,
        initial_description: Optional[str] = None,
        vlm_analysis: Optional[str] = None,
        via_annotations: Optional[Path] = None,
    ) -> None:
        """Add an image reference to the template.

        Args:
            image_id: Unique identifier for the image
            image_path: Path to the image file
            initial_description: User's initial description
            vlm_analysis: VLM-generated analysis text
            via_annotations: Path to VIA annotations file
        """
        image_entry = {
            "id": image_id,
            "path": str(image_path),
            "initial_description": initial_description,
            "vlm_analysis": vlm_analysis,
            "via_annotations": str(via_annotations) if via_annotations else None,
        }
        self.images.append(image_entry)

    def add_few_shot_example(
        self,
        image_id: str,
        description: str,
        analysis: str,
    ) -> None:
        """Add a few-shot example.

        Args:
            image_id: Image identifier
            description: Description of the image
            analysis: Example analysis text
        """
        example = {
            "image_id": image_id,
            "description": description,
            "analysis": analysis,
        }
        self.few_shot_examples.append(example)

    def add_analysis_result(self, result: AnalysisResult) -> None:
        """Add an analysis result.

        Args:
            result: Analysis result to add
        """
        self.analysis_results.append(result)

    def to_markdown(self) -> str:
        """Convert template to markdown format.

        Returns:
            Markdown-formatted template
        """
        lines = [f"# {self.title}\n"]

        # Images section
        if self.images:
            lines.append("## Reference Images\n")
            lines.append("| Image ID | Path | Initial Description | VLM Analysis | Annotations |")
            lines.append("|----------|------|---------------------|--------------|-------------|")

            for img in self.images:
                initial_desc = img.get("initial_description", "") or ""
                vlm_analysis = img.get("vlm_analysis", "") or ""
                annotations = img.get("via_annotations", "") or ""

                # Truncate long text for table
                initial_desc = initial_desc[:50] + "..." if len(initial_desc) > 50 else initial_desc
                vlm_analysis = vlm_analysis[:50] + "..." if len(vlm_analysis) > 50 else vlm_analysis

                lines.append(
                    f"| {img['id']} | {img['path']} | {initial_desc} | {vlm_analysis} | {annotations} |"
                )
            lines.append("")

        # Few-shot examples section
        if self.few_shot_examples:
            lines.append("## Few-Shot Examples\n")
            for i, example in enumerate(self.few_shot_examples, 1):
                lines.append(f"### Example {i}")
                lines.append(f"- **Image ID:** {example['image_id']}")
                lines.append(f"- **Description:** {example['description']}")
                lines.append(f"- **Analysis:** {example['analysis']}")
                lines.append("")

        # Analysis results section
        if self.analysis_results:
            lines.append("## Analysis Results\n")
            for i, result in enumerate(self.analysis_results, 1):
                lines.append(f"### Result {i}")
                lines.append(f"- **Mode:** {result.mode.value}")
                lines.append(f"- **Backend:** {result.backend_used}")
                lines.append(f"- **Processing Time:** {result.processing_time_seconds:.2f}s")
                lines.append(f"- **Analysis:**\n{result.analysis_text}")
                lines.append("")

        # Metadata section
        if self.metadata:
            lines.append("## Metadata\n")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "title": self.title,
            "images": self.images,
            "few_shot_examples": self.few_shot_examples,
            "analysis_results": [
                {
                    "mode": r.mode.value,
                    "image_paths": [str(p) for p in r.image_paths],
                    "analysis_text": r.analysis_text,
                    "structured_data": r.structured_data,
                    "backend_used": r.backend_used,
                    "processing_time_seconds": r.processing_time_seconds,
                    "metadata": r.metadata,
                }
                for r in self.analysis_results
            ],
            "metadata": self.metadata,
        }

    def save(self, output_path: Path, format: str = "markdown") -> None:
        """Save template to file.

        Args:
            output_path: Path to save template
            format: Output format ('markdown' or 'json')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            output_path.write_text(self.to_markdown())
        elif format == "json":
            output_path.write_text(json.dumps(self.to_dict(), indent=2))
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, template_path: Path) -> "ImageAnalysisTemplate":
        """Load template from file.

        Args:
            template_path: Path to template file

        Returns:
            Loaded template instance
        """
        if template_path.suffix == ".json":
            data = json.loads(template_path.read_text())
            template = cls(title=data.get("title", "Image Analysis"))
            template.images = data.get("images", [])
            template.few_shot_examples = data.get("few_shot_examples", [])
            template.metadata = data.get("metadata", {})
            # Note: analysis_results would need to be reconstructed from dict
            return template
        else:
            # For markdown, we'd need a parser - simplified for now
            raise NotImplementedError("Markdown template loading not yet implemented")
