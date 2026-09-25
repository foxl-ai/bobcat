from types import SimpleNamespace

import pytest
import torch

from bobcat.expert_saliency import (
    ExpertObservations,
    instrument_native_loop,
    merge_reports,
    select_experts,
)
from bobcat.schema import file_hash, json_hash


def observed_report(counts=(8, 8, 8), ko=(1.0, 6.0, 3.0), en=(7.0, 1.0, 2.0)):
    obs = ExpertObservations(
        config_sha256="config",
        dataset_sha256="train-data",
        source_checkpoint_sha256="checkpoint",
        experts=3,
        layers=[3],
    )
    with torch.no_grad():
        for stratum, norms in (("ko", ko), ("en", en)):
            obs.begin_stratum(stratum)
            for expert, (count, norm) in enumerate(zip(counts, norms, strict=True)):
                obs.add(
                    3,
                    expert,
                    torch.full((count, 2), norm / 2**0.5),
                    torch.ones(count, 1),
                    weighted_before_down=True,
                )
            obs.end_stratum()
    report = obs.report()
    return {**report, "rank": 0, "world_size": 1}


def test_language_balanced_conditional_score_not_raw_frequency():
    report = merge_reports([observed_report()])
    selected = select_experts(report, keep=1, stratum_weights={"ko": 0.75, "en": 0.25})
    assert selected["experts_by_layer"] == {"3": [1]}
    # Oversampling English must not change the per-language conditional means.
    second = observed_report(counts=(80, 80, 80))
    more = select_experts(merge_reports([second]), keep=1, stratum_weights={"ko": 0.75, "en": 0.25})
    assert more["experts_by_layer"] == selected["experts_by_layer"]


def test_underobserved_experts_are_protected_and_insufficient_coverage_fails():
    report = merge_reports([observed_report(counts=(1, 8, 8))])
    selected = select_experts(report, keep=1)
    assert selected["experts_by_layer"] == {"3": [0]}
    assert selected["audit"]["3"]["protected_underobserved"] == [0]
    incomplete = merge_reports([observed_report(counts=(1, 1, 8))])
    with pytest.raises(ValueError, match="underobserved"):
        select_experts(incomplete, keep=1)


def test_complete_unique_rank_reduction_and_input_identity():
    report = observed_report()
    with pytest.raises(ValueError, match="every rank"):
        merge_reports([report, report])
    two = [{**report, "rank": i, "world_size": 2} for i in range(2)]
    result = merge_reports(two)
    assert result["layers"]["3"]["ko"]["activation_count"] == [16, 16, 16]
    assert select_experts(result, keep=2)["measured"]
    two[1]["dataset_sha256"] = "other-inputs"
    with pytest.raises(ValueError, match="different models, data"):
        merge_reports(two)


def test_no_training_gradient_nonfinite_or_heldout_data_allowed():
    with pytest.raises(ValueError):
        ExpertObservations(
            config_sha256="c",
            dataset_sha256="d",
            experts=3,
            source_checkpoint_sha256="checkpoint",
            layers=[3],
            partition="dev_train",
        )
    obs = ExpertObservations(
        config_sha256="c",
        dataset_sha256="d",
        experts=3,
        layers=[3],
        source_checkpoint_sha256="checkpoint",
        strata=("ko",),
    )
    obs.begin_stratum("ko")
    with pytest.raises(ValueError, match="no-grad"):
        obs.add(3, 0, torch.ones(1, 2), torch.ones(1, 1), weighted_before_down=True)
    with torch.no_grad():
        obs.add(3, 0, torch.full((1, 2), float("nan")), torch.ones(1, 1), weighted_before_down=True)
    obs.end_stratum()
    with pytest.raises(ValueError, match="Nonfinite"):
        obs.report()


class SmallNativeLoop:
    """Only an instrumentation fixture; not a GLM accuracy/performance test."""

    use_torch_mm = False
    use_mxfp8 = False

    def __init__(self, apply_after):
        self.config = SimpleNamespace(apply_router_weight_after_down=apply_after)

    def _forward_loop(
        self,
        x,
        weights,
        indices,
        token_mask,
        gate_and_up_projs,
        down_projs,
        gate_up_proj_bias,
        down_proj_bias,
        n_local_experts,
        experts_start_idx,
        experts_end_idx,
    ):
        y = torch.zeros_like(x)
        for i in range(experts_start_idx, experts_end_idx):
            idx, top = torch.where((indices == i) & token_mask.unsqueeze(-1))
            if idx.numel() == 0:
                continue
            w = weights[idx, top, None]
            expert_out = x[idx] @ gate_and_up_projs[i] @ down_projs[i]
            if not self.config.apply_router_weight_after_down:
                expert_out = expert_out * w
            if self.config.apply_router_weight_after_down:
                expert_out = expert_out * w
            y.index_add_(0, idx, expert_out)
        return y


@pytest.mark.parametrize("apply_after", [False, True])
def test_observer_keeps_forward_exact_and_excludes_padding(apply_after):
    loop = SmallNativeLoop(apply_after)
    obs = ExpertObservations(
        config_sha256="c",
        dataset_sha256="d",
        experts=3,
        layers=[3],
        source_checkpoint_sha256="checkpoint",
        strata=("ko",),
    )
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]])
    weights = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.5, 0.5]])
    indices = torch.tensor([[0, 1], [1, 2], [0, 1]])
    mask = torch.tensor([True, True, False])
    projections = torch.eye(2).repeat(3, 1, 1)
    args = (x, weights, indices, mask, projections, projections, None, None, 3, 0, 3)
    with torch.no_grad():
        expected = loop._forward_loop(*args)
        restore = instrument_native_loop(
            loop, obs.observer(3), expected_file_sha256=file_hash(__file__)
        )
        obs.begin_stratum("ko")
        actual = loop._forward_loop(*args)
        obs.end_stratum()
        restore()
        restored = loop._forward_loop(*args)
    assert torch.equal(expected, actual) and torch.equal(expected, restored)
    result = obs.report()["layers"]["3"]["ko"]
    assert result["activation_count"] == [1, 2, 1]
    assert result["router_weight_sum"] == pytest.approx([0.2, 1.2, 0.6])
    assert result["weighted_output_norm_sum"] == pytest.approx(
        [
            5**0.5 * 0.2,
            5**0.5 * 0.8 + 5 * 0.4,
            5 * 0.6,
        ]
    )


def test_tampered_statistics_cannot_become_a_selection():
    result = merge_reports([observed_report()])
    result["layers"]["3"]["ko"]["activation_count"][0] += 1
    assert (
        json_hash({k: v for k, v in result.items() if k != "content_sha256"})
        != result["content_sha256"]
    )
    with pytest.raises(ValueError, match="checksummed"):
        select_experts(result, keep=2)
