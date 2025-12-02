"""Experiment Tracker Integration.

Bridges VLM tools with the experiment-tracker module.
"""

from pathlib import Path
from typing import List, Optional

from AgentQMS.vlm.core.contracts import AnalysisRequest, AnalysisResult
from AgentQMS.vlm.core.client import VLMClient
from AgentQMS.vlm.integrations.reports import ReportIntegrator
from AgentQMS.vlm.utils.paths import get_path_resolver


class ExperimentTrackerIntegration:
    """Integration with experiment-tracker module."""

    def __init__(self):
        """Initialize experiment tracker integration."""
        self.resolver = get_path_resolver()
        self.client = VLMClient()
        self.report_integrator = ReportIntegrator()

    def analyze_experiment_artifacts(
        self,
        experiment_id: str,
        artifact_paths: List[Path],
        mode: str = "defect",
        auto_populate: bool = True,
    ) -> List[AnalysisResult]:
        """Analyze artifacts from an experiment.

        Args:
            experiment_id: Experiment ID
            artifact_paths: List of artifact image paths
            mode: Analysis mode
            auto_populate: Whether to auto-populate reports

        Returns:
            List of analysis results
        """
        from AgentQMS.vlm.core.contracts import AnalysisMode

        # Create analysis requests
        requests = []
        for artifact_path in artifact_paths:
            request = AnalysisRequest(
                mode=AnalysisMode(mode),
                image_paths=[artifact_path],
                experiment_id=experiment_id,
                auto_populate=auto_populate,
            )
            requests.append(request)

        # Perform analyses
        results = self.client.analyze_batch(requests)

        # Auto-populate reports if requested
        if auto_populate:
            self._populate_experiment_reports(experiment_id, results)

        return results

    def _populate_experiment_reports(
        self,
        experiment_id: str,
        results: List[AnalysisResult],
    ) -> None:
        """Populate experiment reports with analysis results.

        Args:
            experiment_id: Experiment ID
            results: Analysis results
        """
        try:
            experiment_dir = self.resolver.get_experiment_tracker_path("experiments", experiment_id)

            # Find incident reports
            incident_reports_dir = experiment_dir / "incident_reports"
            if incident_reports_dir.exists():
                for report_file in incident_reports_dir.glob("*.md"):
                    try:
                        self.report_integrator.populate_report(report_file, results)
                    except Exception:
                        # Continue with other reports
                        continue

            # Find assessments
            assessments_dir = experiment_dir / "assessments"
            if assessments_dir.exists():
                for assessment_file in assessments_dir.glob("*.md"):
                    try:
                        self.report_integrator.populate_report(assessment_file, results)
                    except Exception:
                        # Continue with other assessments
                        continue

        except Exception:
            # Integration is optional, don't fail if experiment-tracker not available
            pass
