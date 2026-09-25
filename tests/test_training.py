from dataclasses import replace

import pytest
import torch

from bobcat.training import TrainConfig, mask_tokens, train


def test_masking_preserves_special_tokens():
    ids = torch.tensor([[2, 10, 20, 3, 0], [2, 7, 8, 9, 3]])
    corrupted, selected = mask_tokens(ids, vocab_size=100, seed=3)
    assert selected.any()
    assert not selected[ids < 6].any()
    assert torch.equal(corrupted[ids < 6], ids[ids < 6])
    again, repeated = mask_tokens(ids, vocab_size=100, seed=3)
    assert torch.equal(corrupted, again) and torch.equal(selected, repeated)


@pytest.mark.parametrize("initialize_from_mlm", [False, True])
def test_checkpoint_resume_is_exact_with_dropout(
    corpus, small_config, tmp_path, initialize_from_mlm
):
    small_config.dropout = 0.1
    settings = TrainConfig(
        steps=4,
        groups_per_batch=2,
        device="cpu",
        seed=19,
        checkpoint_every=2,
        log_every=4,
        warmup_steps=1,
    )
    root = corpus[0]
    inputs = dict(train_path=root / "data" / "train.jsonl", tokenizer_path=root / "tokenizer.json")
    full = tmp_path / "full"
    partial = tmp_path / "partial"
    parent = None
    if initialize_from_mlm:
        pretraining = tmp_path / "pretraining"
        train(
            small_config, replace(settings, steps=2, objective="mlm"), **inputs, output=pretraining
        )
        parent = pretraining / "last.pt"
    train(small_config, settings, **inputs, output=full, init_from=parent)
    train(small_config, settings, **inputs, output=partial, init_from=parent, stop_after_steps=2)
    train(small_config, settings, **inputs, output=partial, resume=partial / "last.pt")
    first = torch.load(full / "last.pt", weights_only=True)
    second = torch.load(partial / "last.pt", weights_only=True)
    assert first["step"] == second["step"] == 4
    assert first["provenance"] == second["provenance"]
    for key, tensor in first["model"].items():
        torch.testing.assert_close(tensor, second["model"][key], rtol=0, atol=0)


def test_scratch_mlm_checkpoint_can_initialize_decisions(corpus, small_config, tmp_path):
    root = corpus[0]
    settings = TrainConfig(
        steps=2,
        groups_per_batch=2,
        device="cpu",
        checkpoint_every=1,
        log_every=2,
        objective="mlm",
    )
    inputs = dict(train_path=root / "data" / "train.jsonl", tokenizer_path=root / "tokenizer.json")
    pretraining = tmp_path / "mlm"
    train(small_config, settings, **inputs, output=pretraining)
    report = train(
        small_config,
        replace(settings, objective="decision"),
        **inputs,
        output=tmp_path / "decision",
        init_from=pretraining / "last.pt",
    )
    assert report["completed_steps"] == 2
    assert report["provenance"]["weight_origin"] == "random_initialization"
    assert report["provenance"]["initialization_checkpoint_sha256"]
