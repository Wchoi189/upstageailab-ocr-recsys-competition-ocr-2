"""Report Integration.

Auto-populates incident reports and assessments with VLM-generated content.
"""

import re
from pathlib import Path

from AgentQMS.vlm.core.contracts import AnalysisResult
from AgentQMS.vlm.core.interfaces import IntegrationError


class ReportIntegrator:
    """Integrates VLM analysis results into reports."""

    def populate_report(
        self,
        report_path: Path,
        analysis_results: list[AnalysisResult],
        via_annotations: Path | None = None,
    ) -> None:
        """Populate a report with analysis results.

        Args:
            report_path: Path to report file
            analysis_results: List of analysis results to include
            via_annotations: Optional path to VIA annotations

        Raises:
            IntegrationError: If report population fails
        """
        if not report_path.exists():
            raise IntegrationError(f"Report file does not exist: {report_path}")

        if not self.supports_report_type(report_path):
            raise IntegrationError(f"Unsupported report type: {report_path}")

        try:
            content = report_path.read_text()

            # Find the appropriate section to populate
            # For incident reports, populate "Visual Artifacts" section
            # For assessments, add to analysis section

            if "incident_report" in str(report_path) or "incident" in str(report_path).lower():
                content = self._populate_incident_report(content, analysis_results, via_annotations)
            else:
                content = self._populate_assessment(content, analysis_results, via_annotations)

            report_path.write_text(content)

        except Exception as e:
            raise IntegrationError(f"Failed to populate report: {e}") from e

    def _populate_incident_report(
        self,
        content: str,
        results: list[AnalysisResult],
        via_annotations: Path | None,
    ) -> str:
        """Populate incident report with analysis results.

        Args:
            content: Report content
            results: Analysis results
            via_annotations: Optional VIA annotations path

        Returns:
            Updated content
        """
        # Find "Visual Artifacts" section
        pattern = r"(### 1\. Visual Artifacts.*?\n)(.*?)(\n### 2\.)"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            section_header = match.group(1)
            existing_content = match.group(2)
            next_section = match.group(3)

            # Build new content
            new_content = section_header

            # Add VLM analysis
            if results:
                result = results[0]  # Use first result
                new_content += f"\n* **VLM Analysis:** {result.analysis_text}\n"
                new_content += f"* **Analysis Mode:** {result.mode.value}\n"
                new_content += f"* **Backend:** {result.backend_used}\n"

            # Add VIA annotations reference
            if via_annotations:
                new_content += f"* **VIA Annotations:** {via_annotations}\n"

            # Preserve existing content if any
            if existing_content.strip():
                new_content += existing_content

            new_content += next_section

            return content[: match.start()] + new_content + content[match.end() :]

        # If section not found, append at end
        new_section = "\n\n## VLM Analysis\n\n"
        for result in results:
            new_section += f"**Mode:** {result.mode.value}\n"
            new_section += f"**Backend:** {result.backend_used}\n"
            new_section += f"**Analysis:**\n{result.analysis_text}\n\n"

        if via_annotations:
            new_section += f"**VIA Annotations:** {via_annotations}\n"

        return content + new_section

    def _populate_assessment(
        self,
        content: str,
        results: list[AnalysisResult],
        via_annotations: Path | None,
    ) -> str:
        """Populate assessment with analysis results.

        Args:
            content: Assessment content
            results: Analysis results
            via_annotations: Optional VIA annotations path

        Returns:
            Updated content
        """
        # Find a good place to insert (before "## Next Steps" or at end)
        pattern = r"(\n## Next Steps)"
        match = re.search(pattern, content)

        new_section = "\n\n## VLM Image Analysis\n\n"
        for result in results:
            new_section += f"### Analysis ({result.mode.value})\n\n"
            new_section += f"**Backend:** {result.backend_used}\n"
            new_section += f"**Processing Time:** {result.processing_time_seconds:.2f}s\n\n"
            new_section += f"{result.analysis_text}\n\n"

        if via_annotations:
            new_section += f"**VIA Annotations:** {via_annotations}\n\n"

        if match:
            return content[: match.start()] + new_section + content[match.start() :]
        else:
            return content + new_section

    def supports_report_type(self, report_path: Path) -> bool:
        """Check if report type is supported.

        Args:
            report_path: Path to report file

        Returns:
            True if report type is supported
        """
        # Support markdown files
        if report_path.suffix.lower() in (".md", ".markdown"):
            return True

        return False
