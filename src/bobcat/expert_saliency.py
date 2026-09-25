"""Train-only, language-balanced observations for GLM expert pruning.

The score is the conditional mean norm of a routed expert's weighted output,
as in the REAP family of metrics. Counts and router mass are also retained.
This is an importance heuristic, not a prediction of quality after pruning.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
import types
from pathlib import Path

import torch

from bobcat.schema import file_hash, json_hash


class ExpertObservations:
    def __init__(
        self,
        *,
        config_sha256: str,
        dataset_sha256: str,
        source_checkpoint_sha256: str,
        experts: int,
        layers: list[int],
        strata=("ko", "en"),
        partition="train",
        adapter_sha256: str | None = None,
    ):
        if (
            partition != "train"
            or not config_sha256
            or not dataset_sha256
            or not source_checkpoint_sha256
            or type(experts) is not int
            or experts < 1
            or not layers
            or layers != sorted(set(layers))
            or not strata
            or len(set(strata)) != len(strata)
        ):
            raise ValueError("Use attributable training inputs and distinct layers/strata.")
        self.config_sha256, self.dataset_sha256 = config_sha256, dataset_sha256
        self.source_checkpoint_sha256, self.adapter_sha256 = (
            source_checkpoint_sha256,
            adapter_sha256,
        )
        self.experts, self.layers, self.strata = experts, layers, tuple(strata)
        self.buffers = {}
        self.stratum = None

    def begin_stratum(self, stratum: str):
        if self.stratum is not None or stratum not in self.strata:
            raise ValueError("Profile one explicit homogeneous language stratum at a time.")
        self.stratum = stratum

    def end_stratum(self):
        if self.stratum is None:
            raise ValueError("No observation stratum is active.")
        self.stratum = None

    def observer(self, layer: int):
        if layer not in self.layers:
            raise ValueError("Unregistered source layer.")
        return LayerObserver(self, layer)

    def add(
        self,
        layer: int,
        expert: int,
        output: torch.Tensor,
        weights: torch.Tensor,
        *,
        weighted_before_down: bool,
    ):
        if (
            torch.is_grad_enabled()
            or self.stratum is None
            or not 0 <= expert < self.experts
            or layer not in self.layers
            or output.ndim != 2
            or weights.shape != (output.shape[0], 1)
        ):
            raise ValueError(
                "Only explicit no-grad, valid dispatched token observations are allowed."
            )
        key = (layer, self.stratum)
        if key not in self.buffers:
            self.buffers[key] = torch.zeros(
                self.experts, 4, dtype=torch.float64, device=output.device
            )
        buffer = self.buffers[key]
        if buffer.device != output.device:
            raise ValueError("Do not move an active profiler between devices or ranks.")
        norm = torch.linalg.vector_norm(output.detach().float(), dim=-1)
        w = weights.detach().float().squeeze(-1)
        if not weighted_before_down:
            norm = norm * w.abs()
        # Do not .item() or copy per expert/token: synchronize only on report().
        invalid = (~torch.isfinite(norm)).sum() + (~torch.isfinite(w)).sum() + (w < 0).sum()
        buffer[expert, 0] += output.shape[0]
        buffer[expert, 1] += norm.double().sum()
        buffer[expert, 2] += w.double().sum()
        buffer[expert, 3] += invalid

    def report(self) -> dict:
        if self.stratum is not None:
            raise ValueError("End the forward observation window before materializing a report.")
        expected = {(layer, stratum) for layer in self.layers for stratum in self.strata}
        if set(self.buffers) != expected:
            raise ValueError("Some layers or language strata were never observed.")
        records = {}
        for (layer, stratum), buffer in sorted(self.buffers.items()):
            values = buffer.detach().cpu()
            if not torch.isfinite(values).all() or values[:, 3].sum() != 0:
                raise ValueError("Nonfinite or negative routing measurements invalidate selection.")
            records.setdefault(str(layer), {})[stratum] = {
                "activation_count": values[:, 0].long().tolist(),
                "weighted_output_norm_sum": values[:, 1].tolist(),
                "router_weight_sum": values[:, 2].tolist(),
            }
        result = {
            "schema": "bobcat-expert-observations-v1",
            "source_config_sha256": self.config_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "adapter_sha256": self.adapter_sha256,
            "dataset_sha256": self.dataset_sha256,
            "fitting_partition": "train",
            "source_experts_per_layer": self.experts,
            "strata": list(self.strata),
            "layers": records,
            "metric": "conditional_mean_router_weighted_output_l2",
            "model_quality_after_pruning_measured": False,
            "padding_policy": "Only token_mask-selected dispatches enter the observer.",
            "distributed_reduction_performed": False,
        }
        result["content_sha256"] = json_hash(result)
        return result


class LayerObserver:
    def __init__(self, observations, layer):
        self.observations, self.layer = observations, layer

    def add_expert(self, expert, output, weights, *, weighted_before_down):
        self.observations.add(
            self.layer, expert, output, weights, weighted_before_down=weighted_before_down
        )


def instrument_native_loop(module, observer: LayerObserver, *, expected_file_sha256: str):
    """Insert a read-only observer in a hash-pinned native per-expert loop.

    No expert is recomputed. Its output is observed immediately before the
    existing scatter/reduction. This supports the already validated `torch`
    expert backend; grouped GEMM/DeepEP are deliberately rejected.
    """
    if (
        getattr(module, "use_torch_mm", None) is not False
        or getattr(module, "use_mxfp8", None) is not False
        or hasattr(module, "_bobcat_observer")
    ):
        raise ValueError("Only the uninstrumented native Torch loop is supported.")
    original = module._forward_loop
    source_file = inspect.getsourcefile(original)
    if source_file is None or file_hash(Path(source_file)) != expected_file_sha256:
        raise ValueError("The installed native expert source is not the pinned implementation.")
    parsed = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = parsed.body[0]
    expected_args = [
        "self",
        "x",
        "weights",
        "indices",
        "token_mask",
        "gate_and_up_projs",
        "down_projs",
        "gate_up_proj_bias",
        "down_proj_bias",
        "n_local_experts",
        "experts_start_idx",
        "experts_end_idx",
    ]
    if (
        not isinstance(function, ast.FunctionDef)
        or [arg.arg for arg in function.args.args] != expected_args
    ):
        raise ValueError("The native loop signature changed.")
    inserted = 0
    for statement in function.body:
        if not isinstance(statement, ast.For):
            continue
        for index, child in enumerate(statement.body):
            if (
                isinstance(child, ast.If)
                and ast.unparse(child.test) == "self.config.apply_router_weight_after_down"
            ):
                callback = ast.parse(
                    "self._bobcat_observer.add_expert("
                    "i, expert_out, w, "
                    "weighted_before_down=not self.config.apply_router_weight_after_down)"
                ).body[0]
                statement.body.insert(index, callback)
                inserted += 1
                break
    if inserted != 1:
        raise ValueError("The expected expert-output observation point changed.")
    ast.fix_missing_locations(parsed)
    namespace = dict(original.__func__.__globals__)
    exec(compile(parsed, source_file + "::bobcat_observer", "exec"), namespace)
    module._bobcat_observer = observer
    module._forward_loop = types.MethodType(namespace[function.name], module)

    def restore():
        if getattr(module, "_bobcat_observer", None) is not observer:
            raise ValueError("Observer ownership changed before cleanup.")
        module._forward_loop = original
        del module._bobcat_observer

    return restore


def merge_reports(reports: list[dict]) -> dict:
    """Sum disjoint expert-parallel/DP dispatch observations, never average ranks.

    The caller must supply one report per unique rank. In native EP each routed
    token/expert pair is evaluated only by its owner; language windows must be
    homogeneous across all ranks. Rank identity and completeness are validated.
    """
    if not reports:
        raise ValueError("No rank observations.")
    world = reports[0].get("world_size")
    if (
        type(world) is not int
        or world != len(reports)
        or {r.get("rank") for r in reports} != set(range(world))
    ):
        raise ValueError("Need exactly one observation report for every rank.")
    identity = (
        "source_config_sha256",
        "source_checkpoint_sha256",
        "adapter_sha256",
        "dataset_sha256",
        "fitting_partition",
        "source_experts_per_layer",
        "strata",
        "metric",
        "world_size",
    )
    for report in reports:
        for key in identity:
            if report.get(key) != reports[0].get(key):
                raise ValueError("Ranks profiled different models, data or strata.")
        raw = {
            key: value
            for key, value in report.items()
            if key not in ("content_sha256", "rank", "world_size")
        }
        if json_hash(raw) != report["content_sha256"]:
            raise ValueError("A rank observation was modified.")
        if set(report["layers"]) != set(reports[0]["layers"]):
            raise ValueError("Ranks observed different sparse layers.")
    result = {
        key: value for key, value in reports[0].items() if key not in ("rank", "content_sha256")
    }
    result["layers"] = {}
    for layer in reports[0]["layers"]:
        result["layers"][layer] = {}
        for stratum in result["strata"]:
            arrays = {}
            for field in ("activation_count", "weighted_output_norm_sum", "router_weight_sum"):
                values = [r["layers"][layer][stratum][field] for r in reports]
                if any(len(v) != result["source_experts_per_layer"] for v in values):
                    raise ValueError("Wrong expert vocabulary in rank statistics.")
                arrays[field] = [sum(v[i] for v in values) for i in range(len(values[0]))]
            result["layers"][layer][stratum] = arrays
    result["distributed_reduction_performed"] = True
    result["rank_reports_sha256"] = [
        r["content_sha256"] for r in sorted(reports, key=lambda r: r["rank"])
    ]
    result["content_sha256"] = json_hash(result)
    return result


def select_experts(report: dict, *, keep: int, min_activations=8, stratum_weights=None) -> dict:
    """Protect underobserved experts instead of assuming they are unimportant."""
    raw = {key: value for key, value in report.items() if key != "content_sha256"}
    if (
        json_hash(raw) != report.get("content_sha256")
        or report.get("schema") != "bobcat-expert-observations-v1"
        or report.get("fitting_partition") != "train"
        or report.get("distributed_reduction_performed") is not True
    ):
        raise ValueError("Select from a complete, checksummed train-only rank reduction.")
    total = report["source_experts_per_layer"]
    if type(keep) is not int or not 1 <= keep <= total or min_activations < 1:
        raise ValueError("Invalid keep count or coverage threshold.")
    weights = stratum_weights or {
        stratum: 1 / len(report["strata"]) for stratum in report["strata"]
    }
    if (
        set(weights) != set(report["strata"])
        or any(w <= 0 for w in weights.values())
        or abs(sum(weights.values()) - 1) > 1e-9
    ):
        raise ValueError("Use explicit positive, normalized language weights.")
    selected, audit = {}, {}
    for layer, strata in report["layers"].items():
        scores, protected = [0.0] * total, set()
        for stratum, weight in weights.items():
            entry = strata[stratum]
            counts, norms = entry["activation_count"], entry["weighted_output_norm_sum"]
            if len(counts) != total or len(norms) != total:
                raise ValueError("Incomplete expert observations.")
            for i, (count, norm) in enumerate(zip(counts, norms, strict=True)):
                if (
                    type(count) is not int
                    or count < 0
                    or not math.isfinite(norm)
                    or norm < 0
                    or (count == 0 and norm != 0)
                ):
                    raise ValueError("Invalid count or activation norm.")
                if count < min_activations:
                    protected.add(i)
                else:
                    scores[i] += weight * norm / count
        if len(protected) > keep:
            raise ValueError(
                f"Layer {layer}: {len(protected)} underobserved experts exceed keep={keep}."
            )
        ordered = sorted(set(range(total)) - protected, key=lambda i: (-scores[i], i))
        chosen = sorted(protected | set(ordered[: keep - len(protected)]))
        selected[layer] = chosen
        audit[layer] = {
            "protected_underobserved": sorted(protected),
            "conditional_scores": scores,
            "kept_experts": chosen,
        }
    return {
        "schema": "bobcat-expert-selection-v1",
        "measured": True,
        "source_config_sha256": report["source_config_sha256"],
        "source_checkpoint_sha256": report["source_checkpoint_sha256"],
        "observed_adapter_sha256": report["adapter_sha256"],
        "observations_sha256": report["content_sha256"],
        "dataset_sha256": report["dataset_sha256"],
        "fitting_partition": "train",
        "metric": report["metric"],
        "stratum_weights": weights,
        "minimum_activations_per_expert_per_stratum": min_activations,
        "experts_by_layer": selected,
        "audit": audit,
        "quality_after_pruning_measured": False,
    }
