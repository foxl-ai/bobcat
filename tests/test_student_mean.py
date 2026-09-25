import copy

import pytest
import torch

from bobcat.batching import EncodedDataset
from bobcat.metrics import scored_row
from bobcat.model import ModelConfig
from bobcat.public_decisions import student_example
from bobcat.schema import Choice, Example


def test_mean_batch_keeps_order_precision_and_annotations_outside_model_inputs(corpus):
    tokenizer = corpus[2]
    config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2,
                         encoder_layers=2, max_context_tokens=256, max_schema_tokens=128)
    mean = Example(
        id="PRIVATE", group_id="PRIVATE", family="fixture", split="train",
        context="A customer asked about delivery.", instruction="Rate the urgency.",
        choices=[Choice(str(i), text) for i, text in enumerate(("low", "medium", "high"))],
        kind="ordinal", target=None, supervision="score_mean", score_target=1 / 3,
    )
    mean.validate()
    changed = copy.deepcopy(mean)
    changed.score_target = 5 / 3
    left = EncodedDataset([mean], tokenizer, config, include_sentinels=False).collate(
        [0], shuffle_seed=78,
    )
    right = EncodedDataset([changed], tokenizer, config, include_sentinels=False).collate(
        [0], shuffle_seed=78,
    )
    assert left.candidate_ids == [["0", "1", "2"]]
    assert left.tensors["targets"].tolist() == [-100]
    assert left.tensors["score_targets"].dtype == torch.float64
    assert left.tensors["score_targets"].item() == 1 / 3
    for key, value in left.model_inputs().items():
        assert torch.equal(value, right.model_inputs()[key])
    assert "score_targets" not in left.to(torch.device("cpu")).model_inputs()
    with pytest.raises(ValueError, match="sentinel"):
        EncodedDataset([mean], tokenizer, config)
    with pytest.raises(ValueError, match="ordinal mean"):
        scored_row({"supervision": "score_mean", "target": None})


def test_typed_student_adapter_preserves_observed_scalar_without_dummy_label():
    row = {
        "id": "private-row", "group_id": "private-group", "observation_id": "private-observation",
        "family": "similarity", "task": "fixture", "language": "ko", "split": "train",
        "source_split": "train", "source": {"private": True}, "kind": "ordinal",
        "supervision": "score_mean", "target": None, "score_target": 0.7,
        "candidate_ids": ["0", "1", "2"],
        "request": {
            "model": "bobcat", "state": {"문장1": "오늘은 맑음", "문장2": "화창한 날씨"},
            "questions": {"decision": {"type": "score", "instructions": "의미 유사도",
                                       "criteria": ["다름", "비슷함", "같음"]}},
        },
    }
    example = student_example(row)
    assert example.target is None and example.score_target == 0.7
    assert example.choices == [Choice("0", "다름"), Choice("1", "비슷함"), Choice("2", "같음")]
    assert "private" not in example.context + example.instruction
    restored = Example.from_dict(example.to_dict())
    assert restored.score_target == example.score_target and restored.target is None
    assert restored.input_fingerprint() == example.input_fingerprint()
    changed = copy.deepcopy(example)
    changed.choices.reverse()
    assert changed.input_fingerprint() != example.input_fingerprint()
