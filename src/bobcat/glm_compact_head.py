"""A physically compact native GLM readout initialized from original head rows.

The input vocabulary and backbone are unchanged. Only the output projection is
allocated at the bank width. Row identities must be bound by the checkpoint/job
manifest; this module never samples or generates answer tokens.
"""

from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from bobcat.schema import file_hash, json_hash


def load_verified_native_bank(folder, *, source, source_manifest_sha256, token_ids):
    """Read the previously exported original GLM rows with their exact identity."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    manifest_file = folder / "manifest.json"
    if manifest_file.is_symlink():
        raise ValueError("Use an immutable local bank manifest.")
    manifest = json.loads(manifest_file.read_text())
    projection, artifact = manifest["projection"], manifest["artifact"]
    path = folder / artifact["path"]
    if (manifest.get("schema") != "bobcat-original-decision-bank-export-v1"
            or manifest.get("status") != "passed"
            or manifest["base_repo"] != source["repo"]
            or manifest["base_revision"] != source["revision"]
            or manifest["source_manifest_sha256"] != source_manifest_sha256
            or projection["source_revision"] != source["revision"]
            or projection["token_ids"] != list(token_ids)
            or projection["original_weight_parameters"] != 154880 * 4096
            or projection["selected_weight_parameters"] != 255 * 4096
            or manifest["verified_original_shard_sha256"] not in {
                item["sha256"] for item in source["files"]
                if item["path"].endswith(".safetensors")
            }
            or path.name != artifact["path"] or path.is_symlink()
            or path.stat().st_size != artifact["bytes"]
            or file_hash(path) != artifact["sha256"]):
        raise ValueError("The bank is not bound to the original model, row order, and bytes.")
    with safe_open(path, framework="pt", device="cpu") as stream:
        metadata = stream.metadata()
    if metadata != {
        "format": "bobcat-original-decision-bank-v1",
        "base_repo": source["repo"], "base_revision": source["revision"],
        "source_head_sha256": manifest["source_head_sha256"], "complete_model": "false",
    }:
        raise ValueError("Bank safetensors metadata differs from its source manifest.")
    values = load_file(path, device="cpu")
    if (set(values) != {"weight", "token_ids"}
            or values["weight"].shape != (255, 4096)
            or values["weight"].dtype != torch.bfloat16
            or values["token_ids"].dtype != torch.long
            or values["token_ids"].tolist() != list(token_ids)
            or not bool(torch.isfinite(values["weight"]).all())):
        raise ValueError("The native bank tensors or identifier mapping changed.")
    raw = values["weight"].contiguous().view(torch.uint8).numpy()
    if hashlib.sha256(raw).hexdigest() != projection["row_sha256"]:
        raise ValueError("Original row checksum mismatch.")
    return values["weight"], {
        "manifest_sha256": file_hash(manifest_file),
        "bank_sha256": artifact["sha256"], "bank_bytes": artifact["bytes"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_head_sha256": manifest["source_head_sha256"],
        "row_sha256": projection["row_sha256"],
        "identifier_order_sha256": json_hash(list(token_ids)),
        "original_output_parameters": projection["original_weight_parameters"],
        "compact_output_parameters": projection["selected_weight_parameters"],
        "backbone_pruned": False, "new_weight_updates": 0,
    }


class NativeDecisionProjection(nn.Linear):
    """An initialization-compatible linear bank with host-side identifier mapping."""

    def __init__(self, hidden_size, token_ids, *, original_vocabulary, dtype, device):
        if (type(hidden_size) is not int or hidden_size < 1
                or type(original_vocabulary) is not int or original_vocabulary < 1
                or not 1 <= len(token_ids) <= 255
                or len(set(token_ids)) != len(token_ids)
                or any(type(i) is not int or not 0 <= i < original_vocabulary
                       for i in token_ids)):
            raise ValueError("Use the original hidden width and distinct in-vocabulary IDs.")
        super().__init__(hidden_size, len(token_ids), bias=False, dtype=dtype, device=device)
        self.requires_grad_(False)
        self._identifiers = tuple(token_ids)
        self._index_by_token = {token: index for index, token in enumerate(token_ids)}
        self.original_vocabulary = original_vocabulary
        self.decision_only = True
        self.external_audit_bank = None
        self.audit_option_ids = None
        self.last_audit = None

    def forward(self, hidden):
        result = super().forward(hidden)
        if self.audit_option_ids is None:
            return result
        external = self.external_audit_bank
        if (not isinstance(external, torch.Tensor) or external.requires_grad
                or external.shape != self.weight.shape or external.dtype != self.weight.dtype
                or external.device != self.weight.device
                or not self.audit_option_ids
                or len(set(self.audit_option_ids)) != len(self.audit_option_ids)
                or any(token not in self._index_by_token for token in self.audit_option_ids)):
            raise ValueError("Audit the materialized compact head against its verified row bank.")
        selected = torch.tensor(
            [self._index_by_token[token] for token in self.audit_option_ids],
            dtype=torch.long, device=result.device,
        )
        actual = result.index_select(-1, selected).float()
        expected = nn.functional.linear(hidden, external).index_select(-1, selected).float()
        raw = self.weight.detach().cpu().contiguous().view(torch.uint8).numpy()
        self.last_audit = {
            "same_hidden": True, "loaded_rows_bitwise_equal": bool(torch.equal(
                self.weight, external,
            )),
            "loaded_row_sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
            "loaded_shape": list(self.weight.shape), "loaded_dtype": str(self.weight.dtype),
            "maximum_probability_tv": float(
                (actual.softmax(-1) - expected.softmax(-1)).abs().sum(-1).max() / 2,
            ),
            "argmax_equal": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
            "maximum_logit_difference": float((actual - expected).abs().max()),
            "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
            "production_latency_measured": False,
        }
        return result

    def select_batch(self, bank_logits, option_token_ids):
        if (bank_logits.ndim != 2 or bank_logits.shape != (
                len(option_token_ids), len(self._identifiers)) or not option_token_ids):
            raise ValueError("Retain one candidate-bank vector per question.")
        result = []
        for values, requested in zip(bank_logits, option_token_ids, strict=True):
            if (not requested or len(set(requested)) != len(requested)
                    or any(type(i) is not int or i not in self._index_by_token
                           for i in requested)):
                raise ValueError("Never fill an unrepresented or duplicate candidate row.")
            indices = torch.tensor([self._index_by_token[i] for i in requested],
                                   dtype=torch.long, device=bank_logits.device)
            result.append(values.index_select(-1, indices).float())
        return result


def compare_frozen_body_hashes(observed, reference):
    """Compare actual frozen shards, excluding only the intentionally changed head."""
    head = "lm_head.weight"
    if (head not in observed or head not in reference or set(observed) != set(reference)
            or len(observed) < 2
            or any(not isinstance(digest, str) or len(digest) != 64
                   or any(char not in "0123456789abcdef" for char in digest)
                   for hashes in (observed, reference) for digest in hashes.values())):
        raise ValueError("Use matching native frozen-state keys and SHA256 values.")
    names = sorted(name for name in observed if name != head)
    differences = [name for name in names if observed[name] != reference[name]]
    return {
        "compared_body_tensors": len(names), "body_values_exact": not differences,
        "different_body_tensors": differences,
        "excluded_keys": [head], "original_head_sha256": reference[head],
        "compact_head_sha256": observed[head],
        "head_values_equal": observed[head] == reference[head],
        "different_process_comparison": True,
    }


def install_uninitialized_native_decision_projection(model, token_ids):
    """Construct the small output before FSDP; a verified loader must fill it.

    The new output is deliberately uninitialized with respect to the source
    model. Callers must not evaluate it until the whole checkpoint restore,
    including the separately verified bank, completes.
    """
    if any(hasattr(p, "placements") for p in model.parameters()):
        raise ValueError("Replace the output before distributed wrapping.")
    head, embedding = model.get_output_embeddings(), model.get_input_embeddings()
    if (type(head) is not nn.Linear or head.bias is not None
            or head.weight.requires_grad):
        raise ValueError("Use the unchanged frozen, bias-free native vocabulary output.")
    if head.weight is getattr(embedding, "weight", None):
        raise ValueError("Tied embeddings need a separate preservation design.")
    projection = NativeDecisionProjection(
        head.in_features, token_ids, original_vocabulary=head.out_features,
        dtype=head.weight.dtype, device=head.weight.device,
    )
    model.set_output_embeddings(projection)
    if model.get_input_embeddings() is not embedding:
        raise ValueError("Output construction changed the original input embedding.")
    return projection


def compact_glm_checkpoint_adapter(original, row_bank, *, on_part_loaded=None, **kwargs):
    """Load the original backbone and a verified bank without a full output tensor.

    The vendor layer converter remains responsible for every backbone tensor.
    The first nonempty shared load part also installs the output bank into its
    existing local shard. The original full-vocabulary output is not a DCP
    destination. This function does not itself validate the external bank's
    source signature; the caller must verify its artifact and row-ID manifest.
    """
    from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadPart

    from bobcat.glm_checkpoint_parts import _check_copy_layout, _local, bounded_glm_adapter

    if (not isinstance(row_bank, torch.Tensor) or row_bank.is_meta
            or row_bank.ndim != 2 or not 1 <= row_bank.shape[0] <= 255
            or row_bank.device.type != "cpu" or row_bank.requires_grad
            or not bool(torch.isfinite(row_bank).all())
            or row_bank.numel() * row_bank.element_size() > 16 * 1024**2):
        raise ValueError("Use a finite, materialized, frozen CPU decision bank under 16 MiB.")
    bank = row_bank.detach().contiguous().clone()
    bounded = bounded_glm_adapter(original, on_part_loaded=on_part_loaded, **kwargs)
    head_key = "lm_head.weight"

    class CompactOutputAdapter(type(bounded)):
        def iter_checkpoint_load_parts(self, model_state_dict, device_mesh=None):
            target = model_state_dict.get(head_key)
            if (not isinstance(target, torch.Tensor) or target.is_meta
                    or tuple(target.shape) != tuple(bank.shape) or target.dtype != bank.dtype):
                raise ValueError("Native compact output storage differs from the verified bank.")
            if any(name.startswith("lm_head.") and name != head_key
                   for name in model_state_dict):
                raise ValueError("Unexpected output bias, mapping buffer, or auxiliary state.")
            remaining = {name: value for name, value in model_state_dict.items()
                         if name != head_key}
            parts = super().iter_checkpoint_load_parts(remaining, device_mesh)
            return self._with_output(parts, target)

        def _with_output(self, parts, target):
            first = next(parts)
            if (not first.checkpoint_tensors or not first.model_keys
                    or any(name.startswith("model.language_model.layers.")
                           for name in first.model_keys)
                    or head_key in first.checkpoint_tensors):
                raise ValueError("Initialize the bank with the original nonempty shared part.")
            finished = {"attempted": False, "complete": False}

            def finish():
                if finished["attempted"]:
                    raise ValueError("A compact output load part may be installed only once.")
                finished["attempted"] = True
                first.finish()
                with torch.no_grad():
                    if hasattr(target, "placements"):
                        from torch.distributed.tensor import distribute_tensor
                        loaded = distribute_tensor(
                            bank.to(target.device_mesh.device_type),
                            device_mesh=target.device_mesh, placements=target.placements,
                        )
                    else:
                        loaded = bank.to(target.device)
                    _check_copy_layout(target, loaded, allow_cuda_to_cpu=True)
                    _local(target).copy_(_local(loaded))
                if on_part_loaded is not None:
                    on_part_loaded((head_key,),
                                   _local(target).numel() * target.element_size())
                finished["complete"] = True

            yield CheckpointLoadPart(
                checkpoint_tensors=first.checkpoint_tensors,
                model_keys=first.model_keys | frozenset({head_key}),
                temporary_checkpoint_keys=first.temporary_checkpoint_keys,
                finish=finish,
            )
            if not finished["complete"]:
                raise ValueError("Finish the compact shared part before loading a layer.")
            yield from parts

    return CompactOutputAdapter(
        original.config, original.moe_config, original.backend, original.dtype,
    )
