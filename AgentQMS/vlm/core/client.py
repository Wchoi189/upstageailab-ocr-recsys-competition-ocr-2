"""VLM Client Facade.

Main coordinator that handles backend selection, image preprocessing, prompt management,
few-shot learning, and error handling.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from AgentQMS.vlm.backends import create_backend
from AgentQMS.vlm.core.config import get_config, resolve_env_value
from AgentQMS.vlm.core.contracts import (
    AnalysisRequest,
    AnalysisResult,
)
from AgentQMS.vlm.core.interfaces import BackendError, VLMBackend
from AgentQMS.vlm.core.preprocessor import VLMImagePreprocessor
from AgentQMS.vlm.integrations.via import VIAIntegration
from AgentQMS.vlm.prompts.manager import PromptManager
from AgentQMS.vlm.utils.paths import get_path_resolver


class VLMClient:
    """Main VLM client that coordinates all components."""

    def __init__(
        self,
        backend_preference: str | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize VLM client.

        Args:
            backend_preference: Preferred backend ('openrouter', 'solar_pro2', 'cli')
            cache_dir: Directory for caching processed images
        """
        self._config = get_config()
        env_backend = resolve_env_value("VLM_BACKEND")
        self.backend_preference = backend_preference or env_backend or self._config.backends.default
        self._backend_priority = self._build_backend_priority()
        resolver = get_path_resolver()
        self.cache_dir = cache_dir or resolver.get_cache_path()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = VLMImagePreprocessor(cache_dir=self.cache_dir)
        self.prompt_manager = PromptManager()
        self.via_integration = VIAIntegration()

        self._backend: VLMBackend | None = None

    def _get_backend(self) -> VLMBackend:
        """Get or create backend instance.

        Returns:
            Available backend instance

        Raises:
            BackendError: If no backend is available
        """
        if self._backend is not None:
            return self._backend

        # Try backends in priority order
        for backend_type in self._backend_priority:
            if backend_type is None:
                continue

            try:
                config_dict = self._build_backend_config(backend_type)
                backend = create_backend(backend_type, config_dict)

                if backend.is_available():
                    self._backend = backend
                    return backend
            except Exception:
                # Try next backend
                continue

        raise BackendError("No VLM backend is available. Check API keys and CLI installation.")

    def analyze(
        self,
        request: AnalysisRequest,
    ) -> AnalysisResult:
        """Analyze image(s) according to the request.

        Args:
            request: Analysis request with parameters

        Returns:
            Analysis result

        Raises:
            BackendError: If analysis fails
        """
        start_time = time.time()

        # Get backend
        backend = self._get_backend()
        max_resolution = min(request.max_resolution, backend.get_max_resolution())

        # Preprocess images
        processed_images = self.preprocessor.preprocess_batch(
            request.image_paths,
            max_resolution=max_resolution,
        )

        if not processed_images:
            raise BackendError("Failed to preprocess any images")

        # Load VIA annotations if provided
        via_context = {}
        if request.via_annotations:
            try:
                annotation = self.via_integration.load_annotations(request.via_annotations)
                via_export = self.via_integration.export_annotations_for_vlm(annotation)
                via_context["via_annotations"] = via_export.get("annotations", [])
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load VIA annotations: {e}")

        # Build prompt context
        context: dict[str, Any] = {
            "initial_description": request.initial_description,
            **via_context,
        }

        # Load few-shot examples if provided
        if request.few_shot_examples:
            try:
                examples_data = json.loads(request.few_shot_examples.read_text())
                context["few_shot_examples"] = examples_data.get("examples", [])
            except Exception:
                # Fallback to default examples
                context = self.prompt_manager.inject_few_shot_examples(context)
        else:
            context = self.prompt_manager.inject_few_shot_examples(context)

        # Render prompt
        prompt = self.prompt_manager.render_template(
            request.mode,
            context=context,
            use_jinja2=True,
        )

        # Analyze with backend (use first processed image for now)
        # TODO: Support multiple images for comparison mode
        processed_image = processed_images[0]

        try:
            analysis_text = backend.analyze_image(
                processed_image,
                prompt,
                request.mode,
            )
        except BackendError as e:
            raise BackendError(f"Backend analysis failed: {e}") from e

        processing_time = time.time() - start_time

        # Build rich metadata for downstream consumers (CLI, experiment agents, etc.)
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        image_info: list[dict[str, Any]] = []
        for img in processed_images:
            image_info.append(
                {
                    "original_path": str(img.original_path.resolve()),
                    "processed_path": str(img.processed_path.resolve()) if img.processed_path else None,
                    "width": img.width,
                    "height": img.height,
                    "original_width": img.original_width,
                    "original_height": img.original_height,
                    "resize_ratio": img.resize_ratio,
                }
            )

        metadata: dict[str, Any] = {
            "max_resolution": max_resolution,
            "images_processed": len(processed_images),
            "backend_type": backend.config.backend_type,
            "model_name": getattr(backend, "model_name", backend.config.model or backend.config.backend_type),
            "timestamp": timestamp,
            "image_info": image_info,
        }

        if request.experiment_id:
            metadata["experiment_id"] = request.experiment_id

        return AnalysisResult(
            mode=request.mode,
            image_paths=request.image_paths,
            analysis_text=analysis_text,
            structured_data=None,  # Could parse analysis_text here
            backend_used=backend.config.backend_type,
            processing_time_seconds=processing_time,
            metadata=metadata,
        )

    def analyze_batch(
        self,
        requests: list[AnalysisRequest],
    ) -> list[AnalysisResult]:
        """Analyze multiple images in batch.

        Args:
            requests: List of analysis requests

        Returns:
            List of analysis results
        """
        results = []
        for request in requests:
            try:
                result = self.analyze(request)
                results.append(result)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Failed to analyze {request.image_paths}: {e}")
                # Continue with next request
                continue

        return results

    def _build_backend_priority(self) -> list[str]:
        priority: list[str] = []
        ordered = [self.backend_preference] if self.backend_preference else []
        ordered.extend(self._config.backends.priority)
        for backend in ordered:
            if backend and backend not in priority:
                priority.append(backend)
        return priority

    def _build_backend_config(self, backend_type: str) -> dict[str, Any]:
        defaults = self._config.backend_defaults
        config: dict[str, Any] = {
            "timeout_seconds": defaults.timeout_seconds,
            "max_retries": defaults.max_retries,
            "max_resolution": defaults.max_resolution,
        }

        if backend_type == "openrouter":
            settings = self._config.backends.openrouter
            api_key = resolve_env_value(settings.api_key_env)
            if api_key:
                config["api_key"] = api_key
            config["model"] = settings.default_model
            config["endpoint"] = settings.base_url
        elif backend_type == "solar_pro2":
            settings = self._config.backends.solar_pro2
            api_key = resolve_env_value(settings.api_key_env)
            if api_key:
                config["api_key"] = api_key
            config["model"] = settings.default_model
            config["endpoint"] = settings.endpoint
        elif backend_type == "dashscope":
            settings = self._config.backends.dashscope
            api_key = resolve_env_value(settings.api_key_env)
            if api_key:
                config["api_key"] = api_key
            config["model"] = settings.default_model
            config["endpoint"] = settings.endpoint

        return config
