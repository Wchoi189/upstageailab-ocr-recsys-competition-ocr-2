import torch

from ocr.domains.recognition.data.collate import recognition_collate_fn


def _sample(text_tokens):
    return {
        "image": torch.zeros(3, 32, 128),
        "text_tokens": text_tokens,
        "label": "dummy",
    }


def test_recognition_collate_default_parity_no_trim():
    batch = [
        _sample(torch.tensor([1, 10, 2, 0, 0, 0], dtype=torch.long)),
        _sample(torch.tensor([1, 11, 12, 2, 0, 0], dtype=torch.long)),
    ]

    out = recognition_collate_fn(batch)

    assert out["images"].shape == (2, 3, 32, 128)
    assert out["text_tokens"].shape == (2, 6)
    assert torch.equal(out["text_tokens"][0], torch.tensor([1, 10, 2, 0, 0, 0]))
    assert torch.equal(out["text_tokens"][1], torch.tensor([1, 11, 12, 2, 0, 0]))


def test_recognition_collate_trim_to_batch_max_non_pad():
    batch = [
        _sample(torch.tensor([1, 10, 2, 0, 0, 0], dtype=torch.long)),
        _sample(torch.tensor([1, 11, 12, 2, 0, 0], dtype=torch.long)),
    ]

    out = recognition_collate_fn(
        batch,
        trim_pad_to_batch_max=True,
        min_keep_tokens=2,
        pad_token_id=0,
        log_length_stats=False,
    )

    assert out["text_tokens"].shape == (2, 4)
    assert torch.equal(out["text_tokens"][0], torch.tensor([1, 10, 2, 0]))
    assert torch.equal(out["text_tokens"][1], torch.tensor([1, 11, 12, 2]))


def test_recognition_collate_trim_respects_min_keep_tokens():
    batch = [
        _sample(torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)),
        _sample(torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)),
    ]

    out = recognition_collate_fn(
        batch,
        trim_pad_to_batch_max=True,
        min_keep_tokens=3,
        pad_token_id=0,
        log_length_stats=False,
    )

    assert out["text_tokens"].shape == (2, 3)


def test_recognition_collate_supports_list_token_input():
    batch = [
        _sample([1, 20, 2, 0, 0]),
        _sample([1, 21, 22, 2, 0]),
    ]

    out = recognition_collate_fn(
        batch,
        trim_pad_to_batch_max=True,
        min_keep_tokens=2,
        pad_token_id=0,
        log_length_stats=False,
    )

    assert out["text_tokens"].shape == (2, 4)
    assert out["text_tokens"].dtype == torch.long
