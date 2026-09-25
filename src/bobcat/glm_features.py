"""Verify frozen GLM features and provide an identity-initialized residual head.

The live identity gate requires --return-hidden-states-mode last and
--enable-fp32-lm-head on the digest-pinned SGLang server. It compares candidate
logit differences, because native vocabulary log probabilities have a common
normalizer. These checks are not a model-quality or calibration evaluation.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from bobcat.checkpoints import verify_checkpoint
from bobcat.glm_readout import GLMCompiler, SGLangScorer
from bobcat.schema import file_hash

FEATURE_PROFILE = "glm53_last_postnorm_fp32_lm_head_v1"


def _resolve_singletons(scorer, compiler, state, questions):
    """A singleton's conditional distribution is fixed; it needs no feature.

    Validate the entire request before removing any singleton. Otherwise a
    large singleton suffix could evade the combined request/context limits.
    Non-singleton questions still use the supplied scorer, including its trained
    residual head, and every feature used by that scorer remains verified.
    """
    compiled = compiler.compile(state, questions)
    active = [i for i, question in enumerate(questions) if len(question.labels) > 1]
    if active:
        # The recursive call contains no singletons and takes the ordinary path.
        values, tokens = scorer.score(state, [questions[i] for i in active])
        if len(values) != len(active):
            raise ValueError("The backend omitted a non-singleton question.")
        measurement = dict(scorer.last_measurement or {})
    else:
        values, tokens, measurement = [], 0, {
            "feature_identity": [], "questions": 0, "logical_input_tokens": 0,
            "native_prompt_tokens": 0, "native_completion_tokens": 0,
            "native_http_seconds": 0.0, "timing_scope": "No native request was made.",
        }
    scores = [[0.0] for _ in questions]
    for index, row in zip(active, values, strict=True):
        scores[index] = row
    scorer.last_measurement = {
        **measurement,
        "singleton_resolution": "host_constant_after_full_request_validation",
        "host_resolved_singleton_indices": [
            i for i, question in enumerate(questions) if len(question.labels) == 1
        ],
        "native_question_indices": active,
        "model_scored_questions": len(active),
        "total_typed_questions": len(questions),
        "validated_logical_request_tokens": compiled.logical_input_tokens,
        "model_scored_logical_input_tokens": tokens,
    }
    return scores, tokens


class FrozenOptionHead:
    def __init__(self, path: Path, expected_sha256: str, compiler: GLMCompiler):
        if file_hash(path) != expected_sha256:
            raise ValueError("Option-head artifact checksum mismatch.")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        source = compiler.source
        shard = next(
            (f for f in source["files"] if f["path"] == payload.get("verified_shard")), None,
        )
        weights, ids = payload.get("weights"), payload.get("token_ids")
        if (payload.get("format") != "bobcat-glm-option-head-v1"
                or payload.get("base_repo") != source["repo"]
                or payload.get("base_revision") != source["revision"]
                or payload.get("head_tensor") != "lm_head.weight"
                or payload.get("head_shape") != [154880, 4096]
                or shard is None or shard["sha256"] != payload.get("verified_shard_sha256")
                or ids != compiler.identifier_ids
                or not isinstance(weights, torch.Tensor)
                or tuple(weights.shape) != (len(ids), 4096)
                or weights.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                or not torch.isfinite(weights).all()):
            raise ValueError("Original option rows, tokenizer or GLM provenance do not match.")
        self.weights = weights.double()
        self.sha256 = expected_sha256
        self.hidden_size = weights.shape[1]
        self.option_count = weights.shape[0]
        self.provenance = {k: payload[k] for k in (
            "base_repo", "base_revision", "head_tensor", "head_shape",
            "verified_shard", "verified_shard_sha256", "token_ids",
        )}

    def verify(self, native_logits: list[list[float]], rows: list[dict],
               *, max_absolute_error: float = 0.005):
        if (not math.isfinite(max_absolute_error) or not 0 < max_absolute_error <= 0.01
                or len(native_logits) != len(rows) or not rows):
            raise ValueError("Use a strict fixed feature-identity gate.")
        features, checks = [], []
        for scores, row in zip(native_logits, rows, strict=True):
            vector = row["meta_info"].get("hidden_states")
            if (not isinstance(vector, list) or len(vector) != self.hidden_size
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
                raise ValueError("Expected one finite last-position hidden vector per question.")
            count = len(scores)
            if not 2 <= count <= self.option_count:
                raise ValueError("Feature identity needs at least two offered candidates.")
            if any(not math.isfinite(x) for x in scores):
                raise ValueError("Native candidate scores are non-finite.")
            hidden = torch.tensor(vector, dtype=torch.float64)
            projected = self.weights[:count] @ hidden
            observed = torch.tensor(scores, dtype=torch.float64)
            # A wrong vector, pre-normalization state or token position must fail.
            difference = (projected - projected[0]) - (observed - observed[0])
            error = float(difference.abs().max())
            if error > max_absolute_error:
                raise ValueError(
                    f"GLM hidden/readout mismatch: {error:.6g} > {max_absolute_error}. "
                    "Do not train on these features."
                )
            features.append(hidden.float())
            checks.append({
                "max_candidate_relative_logit_error": error,
                "allowed_absolute_error": max_absolute_error,
                "candidates": count,
            })
        return torch.stack(features), checks


class VerifiedFeatureScorer:
    """Base GLM probabilities plus a last-position feature integrity check."""

    def __init__(self, native: SGLangScorer, basis: FrozenOptionHead):
        self.native, self.basis = native, basis
        self.model_name = native.model_name + "-fp32-feature-probe"
        self.release_gate_passed = False
        self.temperatures = {}
        self.limits = dict(native.limits)
        self.limits["min_choices_for_feature_identity"] = 2
        self.limits["input_token_accounting"] = (
            "Unique prefix plus non-singleton suffixes, not physical GPU work; "
            "the complete request is still validated against all input limits."
        )
        self.provenance = {
            **native.provenance, "feature_profile": FEATURE_PROFILE,
            "option_head_sha256": basis.sha256,
            "feature_verifier_sha256": file_hash(Path(__file__)),
            "projection_precision": "BF16 model hidden/weights, FP32 native LM-head output",
            "feature_identity_checked_on_every_request": True,
            "feature_identity_scope": "Every native feature request; host singletons "
            "have no feature or native forward.",
            "singleton_resolution": "host_constant_after_full_request_validation",
            "base_weights_updated": False,
            "measurement_includes_internal_hidden_vector_transfer": True,
        }
        self.last_measurement = None

    def extract(self, state, questions):
        if any(len(question.labels) < 2 for question in questions):
            raise ValueError("Feature extraction needs at least two candidates per question.")
        logits, tokens, rows = self.native.native_readout(
            state, questions, capture_last_hidden=True,
        )
        return self._verified(logits, tokens, rows)

    def extract_many(self, requests):
        if any(len(question.labels) < 2 for _, questions in requests for question in questions):
            raise ValueError("Feature extraction needs at least two candidates per question.")
        logits, tokens, rows = self.native.native_many(requests, capture_last_hidden=True)
        return self._verified(logits, tokens, rows)

    def _verified(self, logits, tokens, rows):
        hidden, checks = self.basis.verify(logits, rows)
        self.last_measurement = {
            **self.native.last_measurement,
            "feature_profile": FEATURE_PROFILE, "feature_identity": checks,
            "hidden_values_returned": hidden.numel(),
        }
        return logits, hidden, tokens

    def score(self, state, questions):
        if any(len(question.labels) == 1 for question in questions):
            return _resolve_singletons(self, self.native.compiler, state, questions)
        logits, _, tokens = self.extract(state, questions)
        return logits, tokens


class ResidualDecisionHead(nn.Module):
    """A small learned correction; initialization preserves every base logit.

    Labels remain in the optimizer's loss inputs. This module sees only frozen
    GLM features, base candidate logits and the mask for the offered candidates.
    Training it does not update the GLM backbone or implement its hidden RLCD.
    """

    def __init__(self, hidden_size: int = 4096, max_choices: int = 255, rank: int = 32):
        super().__init__()
        if not 0 < rank <= hidden_size or not 2 <= max_choices <= 255:
            raise ValueError("Invalid residual decision-head dimensions.")
        self.hidden_size, self.max_choices, self.rank = hidden_size, max_choices, rank
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, max_choices, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden, base_logits, candidate_keep):
        if (hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size
                or base_logits.shape != candidate_keep.shape
                or base_logits.shape[0] != hidden.shape[0]
                or base_logits.shape[1] > self.max_choices
                or candidate_keep.dtype != torch.bool
                or not candidate_keep.any(-1).all()):
            raise ValueError("Residual readout requires aligned nonempty candidate sets.")
        correction = self.up(F.silu(self.down(hidden.float())))[:, :base_logits.shape[1]]
        return (base_logits.float() + correction).masked_fill(~candidate_keep, float("-inf"))


class ResidualFeatureScorer:
    """Evaluate the learned head with the exact frozen GLM feature profile."""

    def __init__(self, features: VerifiedFeatureScorer, checkpoint: Path):
        record = verify_checkpoint(checkpoint, expected_format="bobcat-glm-residual-v1")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source = saved["provenance"]["scorer"]
        for key in (
            "base_repo", "base_revision", "model_source_manifest_content_sha256",
            "compiler_sha256", "profile", "feature_profile", "option_head_sha256",
        ):
            if key not in source or source[key] != features.provenance.get(key):
                raise ValueError("Live GLM feature profile differs from the trained head.")
        config = saved["config"]
        if (saved.get("format") != "bobcat-glm-residual-v1" or saved["step"] < 1
                or config["hidden_size"] != 4096 or config["max_choices"] != 255
                or saved["provenance"]["base_weights_updated"] is not False):
            raise ValueError("Use a trained residual head for the original frozen GLM.")
        self.model = ResidualDecisionHead(**config).eval().requires_grad_(False)
        self.model.load_state_dict(saved["model"], strict=True)
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Residual head contains non-finite parameters.")
        self.features = features
        self.model_name = "bobcat-glm53-residual-" + record["sha256"][:12]
        self.release_gate_passed = False
        self.temperatures = {}
        self.limits = dict(features.limits)
        self.provenance = {
            **features.provenance, "head_checkpoint_sha256": record["sha256"],
            "head_training": saved["provenance"], "calibration_fitted": False,
            "base_weights_updated": False,
        }
        self.last_measurement = None

    @torch.inference_mode()
    def score(self, state, questions):
        if any(len(question.labels) == 1 for question in questions):
            return _resolve_singletons(self, self.features.native.compiler, state, questions)
        scores, hidden, tokens = self.features.extract(state, questions)
        started = time.perf_counter()
        width = max(map(len, scores))
        base = torch.full((len(scores), width), float("-inf"))
        keep = torch.zeros_like(base, dtype=torch.bool)
        for i, values in enumerate(scores):
            base[i, :len(values)] = torch.tensor(values)
            keep[i, :len(values)] = True
        adjusted = self.model(hidden, base, keep)
        self.last_measurement = {
            **self.features.last_measurement,
            "residual_head_cpu_seconds": time.perf_counter() - started,
            "head_checkpoint_sha256": self.provenance["head_checkpoint_sha256"],
            "candidate_probability_semantics": (
                "Learned residual candidate logits; T=1, calibration not fitted."
            ),
        }
        # The modified head is not a full-vocabulary language-model output.
        self.last_measurement["base_candidate_log_probability_mass"] = (
            self.last_measurement.pop("candidate_log_probability_mass", None)
        )
        return [adjusted[i, :len(values)].tolist() for i, values in enumerate(scores)], tokens
