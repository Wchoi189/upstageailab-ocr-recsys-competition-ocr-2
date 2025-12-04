"""Integration tests for FastAPI inference preview contract.  BUG-001.

These tests ensure that the FastAPI /inference/preview endpoint preserves
the preview image contract introduced for BUG-001:

- When the underlying inference engine returns a ``preview_image_base64``
  field alongside polygons/texts/confidences, the API response model
  ``InferencePreviewResponse`` must expose this field unchanged so that
  frontends can render overlays in the correct coordinate system.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from apps.backend.services.playground_api.routers import inference as inference_module


def test_run_inference_preview_passes_through_preview_image_base64(monkeypatch) -> None:
    """BUG-001: preview_image_base64 should flow from engine result to API model."""

    class DummyEngine:
        def load_model(self, checkpoint_path: str, config_path: str | None = None) -> bool:  # noqa: ARG002
            return True

        def predict_array(self, **_: Any) -> dict[str, Any]:
            return {
                "polygons": "0 0 10 0 10 10 0 10",
                "texts": ["foo"],
                "confidences": [0.9],
                "preview_image_base64": "Zm9v",  # 'foo' in base64
            }

    def fake_load_image(image_base64: str | None, image_path: str | None) -> np.ndarray:  # noqa: ARG001
        # Return a tiny dummy image; shape is irrelevant for this contract test.
        return np.zeros((8, 8, 3), dtype=np.uint8)

    # BUG-001: ensure router uses dummy engine and load_image, and that the
    # inference engine is considered available.
    monkeypatch.setattr(inference_module, "InferenceEngine", DummyEngine)
    monkeypatch.setattr(inference_module, "_load_image", fake_load_image)
    monkeypatch.setattr(inference_module, "INFERENCE_AVAILABLE", True)

    request = inference_module.InferencePreviewRequest(
        checkpoint_path="dummy.ckpt",
        image_base64="...",
        confidence_threshold=0.5,
        nms_threshold=0.4,
    )

    response = inference_module.run_inference_preview(request)

    assert response.preview_image_base64 == "Zm9v"
    assert len(response.regions) == 1
    region = response.regions[0]
    assert region.text == "foo"
    assert region.confidence == 0.9
    assert region.polygon == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

