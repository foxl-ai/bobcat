import copy
import math

import pytest
import torch

from bobcat.gradient_statistics import (
    admit_gradient_statistics,
    compare_gradient_statistics,
    gradient_norm_squared,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_batched_checks_preserve_exact_reduction_and_gradient_values(dtype):
    generator = torch.Generator().manual_seed(2718)
    gradients = [torch.randn(3, n, generator=generator, dtype=dtype)
                 for n in (1, 3, 17, 39, 73)]
    originals = [value.clone() for value in gradients]
    first = gradient_norm_squared(gradients, batched=False)
    second = gradient_norm_squared(gradients, batched=True)
    assert first == second
    assert all(torch.equal(a, b) for a, b in zip(gradients, originals, strict=True))
    report = compare_gradient_statistics(gradients, device=torch.device("cpu"))
    assert report["local_value_exact"] and report["finite"]
    assert not report["optimizer_updated"] and not report["additional_model_forward_backward"]


def test_explicit_add_order_does_not_use_new_python_compensated_sum():
    gradients = [torch.tensor([value], dtype=torch.float64) for value in (1e8, 1., 1., 1.)]
    assert gradient_norm_squared(gradients, batched=False) == 1e16
    assert gradient_norm_squared(gradients, batched=True) == 1e16
    assert math.fsum([1e16, 1., 1., 1.]) != 1e16


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gradient_never_disappears_in_batched_statistics(invalid):
    gradients = [torch.tensor([1.]), torch.tensor([invalid])]
    for batched in (False, True):
        with pytest.raises(ValueError, match="Non-finite"):
            gradient_norm_squared(gradients, batched=batched)
    with pytest.raises(ValueError, match="every intended"):
        gradient_norm_squared([None], batched=True)
    with pytest.raises(ValueError, match="every intended"):
        gradient_norm_squared([], batched=True)


def rank_reports():
    return [{
        "rank": rank, "finite": True, "local_value_exact": True,
        "gradient_modified": False, "optimizer_updated": False,
        "samples": [
            {"batched": mode, "warmup": index < 2, "seconds": 1. if mode else 2.}
            for index, mode in enumerate((False, True, False, True, True, False))
        ],
    } for rank in range(8)]


def test_admission_requires_exact_values_on_all_ranks_and_slowest_rank_speedup():
    reports = rank_reports()
    result = admit_gradient_statistics(reports)
    assert result["activated"] and result["measured_speedup"] == 2.
    bad = copy.deepcopy(reports)
    bad[5]["local_value_exact"] = False
    assert not admit_gradient_statistics(bad)["activated"]
    slow = copy.deepcopy(reports)
    for row in slow[7]["samples"]:
        if row["batched"]:
            row["seconds"] = 2.1
    assert not admit_gradient_statistics(slow)["activated"]
    with pytest.raises(ValueError, match="eight unique"):
        admit_gradient_statistics(reports[:-1])
