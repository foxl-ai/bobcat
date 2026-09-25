"""Extract the original decision rows without changing a language backbone.

This replaces the last vocabulary projection, not the Transformer or its
language embeddings. Reduced projection FLOPs are not an end-to-end speed claim.
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn


class DecisionProjection(nn.Module):
    def __init__(self, rows, token_ids, bias=None):
        super().__init__()
        if (rows.ndim != 2 or rows.is_meta or not 1 <= len(token_ids) <= 255
                or len(token_ids) != rows.shape[0] or len(set(token_ids)) != len(token_ids)
                or any(type(i) is not int or i < 0 for i in token_ids)):
            raise ValueError("Each decision identifier needs one materialized original row.")
        if bias is not None and bias.shape != (len(token_ids),):
            raise ValueError("Selected vocabulary bias rows are misaligned.")
        self._identifiers = tuple(token_ids)
        self._index_by_token = {token: index for index, token in enumerate(token_ids)}
        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long))
        self.weight = nn.Parameter(rows.detach().clone(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(),
                                                          requires_grad=False)

    @classmethod
    def from_linear(cls, original, token_ids):
        if not isinstance(original, nn.Linear) or hasattr(original.weight, "placements"):
            raise ValueError("Provide verified, materialized original vocabulary rows.")
        if not token_ids or max(token_ids) >= original.out_features:
            raise ValueError("A decision identifier is outside the original vocabulary.")
        indices = torch.tensor(token_ids, device=original.weight.device, dtype=torch.long)
        bias = None if original.bias is None else original.bias.index_select(0, indices)
        return cls(original.weight.index_select(0, indices), token_ids, bias)

    def _indices(self, option_token_ids, *, device):
        if (not option_token_ids or len(set(option_token_ids)) != len(option_token_ids)
                or any(type(i) is not int or i < 0 for i in option_token_ids)):
            raise ValueError("Provide distinct, valid requested decision identifiers.")
        if any(token not in self._index_by_token for token in option_token_ids):
            raise ValueError("An offered option has no original decision row; never fill it.")
        return torch.tensor([self._index_by_token[token] for token in option_token_ids],
                            device=device, dtype=torch.long)

    def forward(self, hidden, option_token_ids=None):
        """Project into the decision bank, or just one request's candidate rows.

        The no-argument form fits a native ``lm_head(hidden)`` call. It computes
        at most 255 decision slots, with no vocabulary logits or token sampling.
        Candidate meanings still come from the unchanged input descriptions.
        """
        if hidden.shape[-1] != self.weight.shape[1]:
            raise ValueError("The original backbone hidden width must be retained.")
        if option_token_ids is None:
            return nn.functional.linear(hidden, self.weight, self.bias)
        selected = self._indices(option_token_ids, device=self.weight.device)
        bias = None if self.bias is None else self.bias.index_select(0, selected)
        return nn.functional.linear(hidden, self.weight.index_select(0, selected), bias)

    def select_batch(self, bank_logits, option_token_ids):
        """Recover each question's ordered, possibly different-size candidate set."""
        if (bank_logits.ndim != 2 or bank_logits.shape != (
                len(option_token_ids), len(self._identifiers)) or not option_token_ids):
            raise ValueError("Retain exactly one decision position per question.")
        # Host IDs are deliberately not copied back from a CUDA buffer per query.
        indices = [self._indices(ids, device=bank_logits.device) for ids in option_token_ids]
        return [value.index_select(-1, index).float()
                for value, index in zip(bank_logits, indices, strict=True)]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A saved row bank must never silently acquire another row-to-ID mapping.
        saved = state_dict.get(prefix + "token_ids")
        expected = torch.tensor(self._identifiers, dtype=torch.long)
        if saved is not None and (saved.dtype != torch.long or not torch.equal(
                saved.detach().cpu(), expected)):
            error_msgs.append(f"{prefix}decision identifier order differs from the checkpoint")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def provenance(self, *, original_vocabulary, source_revision):
        if (type(original_vocabulary) is not int
                or original_vocabulary <= max(self._identifiers)):
            raise ValueError("Record the original vocabulary size, not the reduced bank size.")
        raw = self.weight.detach().cpu().contiguous().view(torch.uint8).numpy()
        original = original_vocabulary * self.weight.shape[1]
        selected = self.weight.numel()
        return {
            "source_revision": source_revision, "row_sha256": hashlib.sha256(raw).hexdigest(),
            "token_ids": list(self._identifiers), "original_weight_parameters": original,
            "selected_weight_parameters": selected,
            "vocabulary_projection_parameter_fraction": selected / original,
            "backbone_pruned": False, "end_to_end_speed_measured": False,
            "free_text_generation": False, "full_vocabulary_probability_mass_available": False,
        }


def install_materialized_decision_projection(model, token_ids):
    """Install a native-compatible head before distributed wrapping/optimization.

    Full FSDP checkpoint conversion is a separate integration gate. Replacing a
    live FSDP module would leave its ownership/hooks inconsistent, so reject it.
    This does not change the tokenizer, input embeddings, or language backbone.
    """
    if any(hasattr(parameter, "placements") for parameter in model.parameters()):
        raise ValueError("Install the decision head before distributed parameter wrapping.")
    head = model.get_output_embeddings()
    embedding = model.get_input_embeddings()
    if (not isinstance(head, nn.Linear) or head.weight.is_meta
            or head.weight.requires_grad or getattr(head, "bias", None) is not None
            and head.bias.requires_grad):
        raise ValueError("Use the materialized frozen original output projection.")
    if head.weight is getattr(embedding, "weight", None):
        raise ValueError("A tied vocabulary head needs an explicit embedding-preservation design.")
    projection = DecisionProjection.from_linear(head, token_ids)
    model.set_output_embeddings(projection)
    if model.get_input_embeddings() is not embedding:
        raise ValueError("Installing the decision output changed the input embedding.")
    return projection


class RetainedVocabularyDecisionProjection(nn.Linear):
    """Select rows before the GEMM while retaining the original checkpoint layout.

    This is the distributed integration bridge: the existing FSDP loader still
    owns the complete frozen vocabulary parameter. Only the requested decision
    bank participates in the output multiplication. It saves projection work,
    not checkpoint bytes or the full-parameter all-gather.

    The ID mapping is part of the immutable job manifest rather than an extra
    state-dict entry. Install this module before FSDP construction.
    """

    def __init__(self, original, token_ids):
        if (not isinstance(original, nn.Linear)
                or hasattr(original.weight, "placements")
                or original.weight.requires_grad
                or original.bias is not None and original.bias.requires_grad
                or not 1 <= len(token_ids) <= 255
                or len(set(token_ids)) != len(token_ids)
                or any(type(i) is not int or not 0 <= i < original.out_features
                       for i in token_ids)):
            raise ValueError("Use a frozen, undistributed vocabulary head and valid unique IDs.")
        # Avoid allocating a second full vocabulary tensor, including on CPU.
        super().__init__(original.in_features, original.out_features,
                         bias=original.bias is not None, device="meta",
                         dtype=original.weight.dtype)
        self.weight, self.bias = original.weight, original.bias
        self._identifiers = tuple(token_ids)
        self._index_by_token = {token: index for index, token in enumerate(token_ids)}
        self.decision_only = False
        self.audit_option_ids = None
        self.last_audit = None
        self.external_audit_bank = None

    def forward(self, hidden):
        if self.audit_option_ids is not None:
            # A diagnostic call compares both projections from exactly the same
            # hidden tensor. It is not a production latency measurement.
            indices = torch.tensor(self._identifiers, device=self.weight.device,
                                   dtype=torch.long)
            full = nn.functional.linear(hidden, self.weight, self.bias)
            bias = None if self.bias is None else self.bias.index_select(0, indices)
            bank = nn.functional.linear(hidden, self.weight.index_select(0, indices), bias)
            offered = self.audit_option_ids
            selected = torch.tensor([self._index_by_token[i] for i in offered],
                                    device=bank.device, dtype=torch.long)
            original = torch.tensor(offered, device=full.device, dtype=torch.long)
            left = full.index_select(-1, original).float()
            right = bank.index_select(-1, selected).float()
            finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            self.last_audit = {
                "same_hidden": True, "finite": finite,
                "maximum_probability_tv": float(
                    (left.softmax(-1) - right.softmax(-1)).abs().sum(-1).max() / 2
                ),
                "argmax_equal": bool(torch.equal(left.argmax(-1), right.argmax(-1))),
                "maximum_centered_logit_difference": float((
                    (left - left[..., :1]) - (right - right[..., :1])
                ).abs().max()),
                "production_latency_measured": False,
            }
            external = self.external_audit_bank
            if external is not None:
                if (not isinstance(external, torch.Tensor) or external.requires_grad
                        or external.shape != (len(self._identifiers), self.in_features)
                        or external.dtype != self.weight.dtype
                        or external.device != self.weight.device or self.bias is not None):
                    raise ValueError("Use the frozen original bank on this head's device.")
                external_logits = nn.functional.linear(hidden, external).index_select(
                    -1, selected,
                ).float()
                self.last_audit.update(
                    external_original_rows_equal=bool(torch.equal(
                        external, self.weight.index_select(0, indices),
                    )),
                    external_finite=bool(torch.isfinite(external_logits).all()),
                    external_maximum_probability_tv=float((
                        left.softmax(-1) - external_logits.softmax(-1)
                    ).abs().sum(-1).max() / 2),
                    external_argmax_equal=bool(torch.equal(
                        left.argmax(-1), external_logits.argmax(-1),
                    )),
                )
            return full
        if not self.decision_only:
            return nn.functional.linear(hidden, self.weight, self.bias)
        indices = torch.tensor(self._identifiers, device=self.weight.device, dtype=torch.long)
        bias = None if self.bias is None else self.bias.index_select(0, indices)
        return nn.functional.linear(hidden, self.weight.index_select(0, indices), bias)

    def select_batch(self, logits, option_token_ids):
        width = len(self._identifiers) if self.decision_only else self.out_features
        if logits.ndim != 2 or logits.shape != (len(option_token_ids), width):
            raise ValueError("The native decision positions and projection mode disagree.")
        values = []
        for value, identifiers in zip(logits, option_token_ids, strict=True):
            if (not identifiers or len(set(identifiers)) != len(identifiers)
                    or any(type(i) is not int or i not in self._index_by_token
                           for i in identifiers)):
                raise ValueError("An offered option has no original decision row.")
            selected = [self._index_by_token[i] for i in identifiers] \
                if self.decision_only else identifiers
            values.append(value.index_select(
                -1, torch.tensor(selected, device=logits.device, dtype=torch.long),
            ).float())
        return values


def install_retained_vocabulary_projection(model, token_ids):
    """Keep every original state key/parameter while changing the final operation."""
    if any(hasattr(parameter, "placements") for parameter in model.parameters()):
        raise ValueError("Install the decision head before distributed parameter wrapping.")
    head, embedding = model.get_output_embeddings(), model.get_input_embeddings()
    if head.weight is getattr(embedding, "weight", None):
        raise ValueError("A tied vocabulary head requires explicit embedding preservation.")
    projection = RetainedVocabularyDecisionProjection(head, token_ids)
    model.set_output_embeddings(projection)
    if model.get_input_embeddings() is not embedding:
        raise ValueError("The original language input embedding changed.")
    return projection
