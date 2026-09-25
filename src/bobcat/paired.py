"""A supervised contrast on oracle-labelled counterfactual pairs."""

from collections import defaultdict

import torch
from torch import Tensor
from torch.nn import functional as F

from bobcat.batching import Batch


def counterfactual_loss(logits: Tensor, batch: Batch, margin: float = 1.0) -> tuple[Tensor, int]:
    groups = defaultdict(dict)
    for index, example in enumerate(batch.examples):
        if example.pair_id:
            groups[example.pair_id][example.variant] = index
    coordinates = []
    for pair in groups.values():
        if "base" not in pair or "counterfactual" not in pair:
            continue
        a, b = pair["base"], pair["counterfactual"]
        first, second = batch.examples[a], batch.examples[b]
        if first.target == second.target:
            continue
        if {c.id: c.text for c in first.choices} != {c.id: c.text for c in second.choices}:
            # Reused IDs with different meanings are not a valid aligned contrast.
            continue
        a_ids, b_ids = batch.candidate_ids[a], batch.candidate_ids[b]
        if not all(target in a_ids and target in b_ids for target in [first.target, second.target]):
            continue
        coordinates.append(
            [
                a,
                b,
                a_ids.index(first.target),
                a_ids.index(second.target),
                b_ids.index(first.target),
                b_ids.index(second.target),
            ]
        )
    if not coordinates:
        return logits.new_zeros(()), 0
    index = torch.tensor(coordinates, device=logits.device)
    a, b, aa, ab, ba, bb = index.unbind(1)
    gap = (logits[a, aa] - logits[a, ab]) - (logits[b, ba] - logits[b, bb])
    return F.softplus(margin - gap.float()).mean(), len(coordinates)
