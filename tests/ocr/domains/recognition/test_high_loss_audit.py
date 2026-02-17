import types

import pytest
import torch

from ocr.domains.recognition.module import HighLossSample, RecognitionPLModule


@pytest.fixture
def bare_module():
    module = RecognitionPLModule.__new__(RecognitionPLModule)
    module._trainer = types.SimpleNamespace(global_step=0, current_epoch=0, loggers=[])
    module._high_loss_epoch_buffer = []
    module._high_loss_counter = 0
    module._get_wandb_cfg = lambda: {
        "high_loss_audit": {
            "enabled": True,
            "top_k": 2,
            "log_every_n_epochs": 1,
            "min_global_step": 0,
            "max_image_side": 640,
            "include_table": True,
            "include_correct_but_high_loss": False,
        }
    }
    return module


def test_collect_high_loss_samples_applies_top_k_and_filters_non_finite(bare_module):
    bare_module._compute_per_sample_validation_loss = types.MethodType(
        lambda self, pred, inference_out, batch: torch.tensor([0.4, float("inf"), 3.7, float("nan"), 2.5]),
        bare_module,
    )

    batch = {
        "images": torch.randn(5, 3, 8, 8),
        "text_tokens": torch.tensor(
            [
                [1, 10, 11, 2, 0],
                [1, 12, 13, 2, 0],
                [1, 14, 15, 2, 0],
                [1, 16, 17, 2, 0],
                [1, 18, 19, 2, 0],
            ],
            dtype=torch.long,
        ),
        "image_filename": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
    }
    pred_texts = ["a", "b", "c", "d", "e"]
    gt_texts = ["x", "y", "z", "w", "v"]

    bare_module._collect_high_loss_samples(
        batch=batch,
        pred={},
        inference_out={},
        pred_texts=pred_texts,
        gt_texts=gt_texts,
        batch_idx=0,
    )

    retained_losses = sorted([entry[2].loss for entry in bare_module._high_loss_epoch_buffer], reverse=True)
    assert retained_losses == pytest.approx([3.7, 2.5])


def test_log_validation_images_requires_wandb_run_when_enabled(bare_module):
    bare_module._get_wandb_cfg = lambda: {"log_recognition_images": True}
    bare_module._get_wandb_experiment = lambda: None

    batch = {"images": torch.randn(2, 3, 8, 8)}

    with pytest.raises(RuntimeError, match="no active WandB experiment"):
        bare_module._log_validation_images(
            batch=batch,
            pred_texts=["pred1", "pred2"],
            gt_texts=["gt1", "gt2"],
            batch_idx=0,
        )


def test_epoch_buffer_never_exceeds_top_k(bare_module):
    top_k = 2
    for index, loss_value in enumerate([0.1, 1.2, 0.3, 4.5, 2.2]):
        sample = HighLossSample(
            epoch=0,
            global_step=0,
            batch_idx=0,
            sample_idx=index,
            loss=loss_value,
            gt_text="gt",
            pred_text="pred",
            filename=f"file_{index}.jpg",
            image_ref=torch.randn(3, 8, 8),
        )
        bare_module._push_high_loss_sample(sample, top_k)

    assert len(bare_module._high_loss_epoch_buffer) == top_k
    retained_losses = sorted([entry[2].loss for entry in bare_module._high_loss_epoch_buffer], reverse=True)
    assert retained_losses == [4.5, 2.2]


def test_log_high_loss_audit_logs_no_more_than_top_k_images(bare_module, monkeypatch):
    class DummyRun:
        def __init__(self):
            self.logged = []

        def log(self, payload):
            self.logged.append(payload)

    captured = {"count": 0}

    def fake_log_recognition_images(*, images, pred_texts, gt_texts, epoch, limit, seed, filenames, caption_prefix, max_image_side, patch_native_view, wandb_run):
        captured["count"] = len(images)
        assert caption_prefix == "audit/high_loss_samples"
        assert patch_native_view is True

    monkeypatch.setattr(
        "ocr.domains.recognition.callbacks.wandb_logging.log_recognition_images",
        fake_log_recognition_images,
    )

    bare_module._get_wandb_experiment = lambda: DummyRun()
    top_k = 2
    for index, loss_value in enumerate([0.1, 1.2, 0.3, 4.5, 2.2]):
        sample = HighLossSample(
            epoch=0,
            global_step=0,
            batch_idx=0,
            sample_idx=index,
            loss=loss_value,
            gt_text="gt",
            pred_text="pred",
            filename=f"file_{index}.jpg",
            image_ref=torch.randn(3, 8, 8),
        )
        bare_module._push_high_loss_sample(sample, top_k)

    bare_module._log_high_loss_audit()

    assert captured["count"] == top_k
    assert len(bare_module._high_loss_epoch_buffer) == 0
