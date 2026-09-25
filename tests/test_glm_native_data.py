import copy
import sys
from dataclasses import dataclass, field, replace
from types import ModuleType

import pytest
import test_glm_readout

from bobcat.glm_feature_data import MIXED_PLAN_SCHEMA
from bobcat.glm_native_data import compile_plan, packed_rank_batch, single_rank_batch, validate
from bobcat.schema import json_hash
from bobcat.supervision import MEAN_TARGET

compiler = test_glm_readout.compiler


def seal(value):
    value["content_sha256"] = json_hash(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    return value


def plan():
    groups = []
    for index, split in enumerate(("train", "train", "dev_train")):
        mean = index == 1
        question = (
            {"type": "score", "instructions": "두 문장의 의미 유사도",
             "criteria": ["아주 다름", "다름", "약간 다름", "약간 비슷", "비슷", "같음"]}
            if mean else {"type": "noul", "instructions": "본문이 환불을 요청하는가?"}
        )
        row = {
            "id": f"PRIVATE-SOURCE-{index}", "group_id": f"PRIVATE-GROUP-{index}",
            "context_id": f"context-{index}", "family": "fixture", "split": split,
            "kind": "ordinal" if mean else "boolean", "language": "ko",
            "context_weight": 1.0, "target": None if mean else "yes",
            "candidate_ids": [str(i) for i in range(6)] if mean else ["no", "yes"],
        }
        if mean:
            row.update(supervision="score_mean", score_target=13 / 6)
        groups.append({
            "split": split, "task": "sts" if mean else "noul", "group_id": row["group_id"],
            "observation_id": row["id"], "rows": [row], "request": {
                "model": "bobcat-latest", "state": f"환불을 신청합니다. 문서 {index}",
                "questions": {"PRIVATE-QUESTION-NAME": question},
            },
        })
    return seal({
        "schema": MIXED_PLAN_SCHEMA, "group_count": 3, "question_count": 3, "groups": groups,
        "dataset_manifest_sha256": "fixture-data", "seed": 17,
    })


def test_native_inputs_exclude_gold_source_and_question_identifiers(compiler):
    model, _, _ = compiler
    original = plan()
    data = compile_plan(original, model)
    validate(data)
    for row in data["records"]:
        assert set(row["inputs"]) == {"input_ids", "option_token_ids"}
        text = model.host_tokenizer.decode(row["inputs"]["input_ids"])
        assert "PRIVATE" not in text
    changed = copy.deepcopy(original)
    changed["groups"][0]["rows"][0]["target"] = "no"
    changed = compile_plan(seal(changed), model)
    assert changed["records"][0]["inputs"] == data["records"][0]["inputs"]
    assert changed["records"][0]["supervision"] != data["records"][0]["supervision"]


def test_ordinal_annotator_mean_does_not_become_a_class_label(compiler):
    data = compile_plan(plan(), compiler[0])
    row = data["records"][1]
    assert row["supervision"]["target_index"] == MEAN_TARGET
    assert row["supervision"]["score_mean"] == 13 / 6
    assert row["row"]["target"] is None
    assert data["ordinal_means_are_not_vote_distributions"]


def test_ep_padding_preserves_the_real_scored_position_and_document_boundary(compiler):
    row = compile_plan(plan(), compiler[0])["records"][0]
    length = len(row["inputs"]["input_ids"])
    batch = single_rank_batch(row, length + 7)
    assert batch["input_ids"].shape == (1, length + 7)
    assert batch["input_ids"][0, :length].tolist() == row["inputs"]["input_ids"]
    assert batch["_packed_seq_ids"][0, :length].eq(1).all()
    assert batch["_packed_seq_ids"][0, length:].eq(0).all()
    assert batch["logits_to_keep"].tolist() == [length - 1]
    with pytest.raises(ValueError, match="without truncation"):
        single_rank_batch(row, length - 1)


def test_modified_compiled_tokens_are_rejected_even_if_outer_hash_is_recomputed(compiler):
    data = compile_plan(plan(), compiler[0])
    data["records"][0]["inputs"]["input_ids"][0] += 1
    with pytest.raises(ValueError, match="misaligned"):
        validate(seal(data))


def test_source_components_cannot_cross_the_native_training_development_boundary(compiler):
    data = compile_plan(plan(), compiler[0])
    data["records"][2]["row"]["group_id"] = data["records"][0]["row"]["group_id"]
    with pytest.raises(ValueError, match="split boundary"):
        validate(seal(data))


def test_packing_preserves_complete_questions_boundaries_and_readout_alignment():
    rows = [{"inputs": {"input_ids": ids}} for ids in ([10, 11], [20, 21, 22], [30])]
    batch = packed_rank_batch(rows, 8, pad_token_id=7)
    assert batch["input_ids"].tolist() == [[10, 11, 20, 21, 22, 30, 7, 7]]
    assert batch["_packed_seq_ids"].tolist() == [[1, 1, 2, 2, 2, 3, 0, 0]]
    assert batch["logits_to_keep"].tolist() == [1, 4, 5]
    selected = batch["input_ids"][0, batch["logits_to_keep"]].tolist()
    assert selected == [11, 22, 30]
    changed = copy.deepcopy(rows)
    changed[1]["supervision"] = {"target_index": 9, "private_source_id": "SECRET"}
    for key, value in packed_rank_batch(changed, 8, pad_token_id=7).items():
        assert value.equal(batch[key])
    with pytest.raises(ValueError, match="without truncation"):
        packed_rank_batch(rows, 5)


@pytest.mark.parametrize("ids", [[], [-1], [True], [1.5]])
def test_empty_or_invalid_question_cannot_be_hidden_by_packing(ids):
    with pytest.raises(ValueError, match="nonempty"):
        packed_rank_batch([{"inputs": {"input_ids": [1]}}, {"inputs": {"input_ids": ids}}], 8)


@pytest.mark.parametrize("lengths,padding", [([1], 0), ([1], 127), ([2, 3, 1], 0),
                                           ([2, 3, 1], 2), ([127, 129], 128)])
def test_optional_cached_boundaries_match_every_document_and_padding_run(
    monkeypatch, lengths, padding,
):
    """Host input contract only; the real native GPU path needs its own admission."""
    import torch

    @dataclass
    class ContextContract:
        doc_ids: object
        original_seq_len: int | None = None
        _cu_seqlens: dict = field(default_factory=dict)

    module_name = "nemo_automodel.components.models.glm5_next.cp"
    stub = ModuleType(module_name)
    stub.Glm5NextPackedContext = ContextContract
    monkeypatch.setitem(sys.modules, module_name, stub)
    rows = [{"inputs": {"input_ids": list(range(1, n + 1))}} for n in lengths]
    expected = packed_rank_batch(rows, sum(lengths) + padding)
    cached = packed_rank_batch(rows, sum(lengths) + padding, cache_packed_boundaries=True)
    for key in expected:
        assert torch.equal(cached[key], expected[key])
    context = cached["glm5_next_packed_context"]
    assert context.doc_ids is cached["_packed_seq_ids"]
    assert context.original_seq_len == sum(lengths) + padding
    device, cpu = replace(context)._cu_seqlens[0]
    ids = expected["_packed_seq_ids"][0]
    # Compare against transitions in the emitted model document IDs, rather
    # than reconstructing the implementation's cumulative-length arithmetic.
    transitions = (torch.nonzero(ids[1:] != ids[:-1]).flatten() + 1).tolist()
    assert cpu.tolist() == [0, *transitions, len(ids)]
    assert cpu.dtype == torch.long and device.dtype == torch.long
    assert torch.equal(device.cpu(), cpu)
    assert cpu.device.type == "cpu"


def test_boundary_cache_option_does_not_silently_accept_an_ambiguous_value():
    with pytest.raises(ValueError, match="explicit boolean"):
        packed_rank_batch([{"inputs": {"input_ids": [1]}}], 8, cache_packed_boundaries="yes")
