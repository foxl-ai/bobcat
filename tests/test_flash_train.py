"""Flash trainer pieces: the loss terms and the token-budget batching."""

import math
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bobcat.flash_train import micro_batches, row_loss  # noqa: E402

ARGS = types.SimpleNamespace(kd_weight=1.0, gold_weight=0.5, teacher_temperature=1.0)


def test_kd_is_zero_when_student_equals_teacher():
    z = torch.tensor([1.0, -0.5, 2.0])
    row = {"supervision": "teacher_only", "target": None}
    loss, count = row_loss(torch, z, row, np.array([1.0, -0.5, 2.0], dtype=np.float32), ARGS)
    assert count == 1 and abs(float(loss)) < 1e-6


def test_gold_and_kd_add():
    z = torch.tensor([0.0, 0.0])
    row = {"supervision": "hard_label", "target": 1}
    loss, count = row_loss(torch, z, row, np.array([0.0, 0.0], dtype=np.float32), ARGS)
    assert count == 2 and abs(float(loss) - 0.5 * math.log(2)) < 1e-6


def test_teacher_temperature_softens_targets():
    z = torch.tensor([2.0, 0.0])
    row = {"supervision": "teacher_only", "target": None}
    sharp = row_loss(torch, z, row, np.array([4.0, 0.0], np.float32), ARGS)[0]
    soft_args = types.SimpleNamespace(**{**vars(ARGS), "teacher_temperature": 2.0})
    soft = row_loss(torch, z, row, np.array([4.0, 0.0], np.float32), soft_args)[0]
    assert float(soft) < 1e-6 < float(sharp)


def test_score_mean_uses_expected_level():
    z = torch.tensor([0.0, 0.0, 0.0])
    row = {"supervision": "score_mean", "score_target": 1.0, "target": None}
    loss, count = row_loss(torch, z, row, None, ARGS)
    assert count == 1 and abs(float(loss)) < 1e-6


def test_mismatched_teacher_is_ignored():
    z = torch.tensor([0.0, 1.0])
    row = {"supervision": "teacher_only", "target": None}
    loss, count = row_loss(torch, z, row, np.array([0.0, 1.0, 2.0], np.float32), ARGS)
    assert count == 0 and float(loss) == 0.0


def test_micro_batches_respect_budget_and_cover_rows():
    rng = np.random.default_rng(0)
    rows = [{"input_ids": np.zeros(int(n), np.int32)} for n in rng.integers(10, 900, 500)]
    batches = micro_batches(rows, 4096, 64, 1, 0)
    assert sorted(i for b in batches for i in b) == list(range(500))
    for batch in batches:
        longest = max(len(rows[i]["input_ids"]) for i in batch)
        assert len(batch) == 1 or longest * len(batch) <= 4096
        assert len(batch) <= 64
