from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict

import numpy as np
import torch

from bobcat.batching import EncodedDataset
from bobcat.metrics import evaluate_rows, scored_row
from bobcat.model import DecisionModel
from bobcat.schema import SENTINELS, Choice, Example
from bobcat.tokenization import ScratchTokenizer
from bobcat.training import autocast_context, predict


def prior_baseline(training: list[Example], evaluation: list[Example]) -> dict:
    counts = defaultdict(Counter)
    for example in training:
        if example.split != "train":
            raise ValueError("The prior baseline may only fit training labels.")
        if example.target in SENTINELS:
            label = example.target
        elif example.kind == "choice":
            label = "any_candidate"
        else:
            label = next(c.text for c in example.choices if c.id == example.target)
        counts[example.kind][label] += 1
    rows = []
    for example in evaluation:
        current = counts[example.kind]
        weights = []
        for choice in example.choices:
            weight = (
                (current["any_candidate"] + 1) / len(example.choices)
                if example.kind == "choice"
                else current[choice.text] + 1
            )
            weights.append(weight)
        weights.extend(current[label] + 1 for label in SENTINELS)
        rows.append(
            {
                "id": example.id,
                "group_id": example.group_id,
                "family": example.family,
                "kind": example.kind,
                "split": example.split,
                "target": example.target,
                "candidate_ids": [c.id for c in example.choices] + list(SENTINELS),
                "logits": [math.log(weight) for weight in weights],
                "pair_id": example.pair_id,
                "variant": example.variant,
            }
        )
    return {
        "method": (
            "Training-label priors per output type, uniform over unseen dynamic actions. "
            "No context, instructions, or evaluation labels used to choose the prediction."
        ),
        "metrics": evaluate_rows(rows),
    }


@torch.inference_mode()
def perturbations(
    model: DecisionModel,
    examples: list[Example],
    tokenizer: ScratchTokenizer,
    device: torch.device,
    precision: str = "fp32",
    limit: int = 120,
) -> dict:
    if limit < 1 or not examples:
        raise ValueError("Perturbation probes need examples and a positive limit.")
    # Fixed strides can alias the repeating choice/Boolean/counterfactual row layout.
    # Sample independently of labels and report the resulting type composition.
    rng = np.random.default_rng(891)
    indices = sorted(rng.choice(len(examples), min(limit, len(examples)), replace=False).tolist())
    selected = [examples[index] for index in indices]
    data = EncodedDataset(selected, tokenizer, model.config)
    raw = predict(model, data, device, precision)
    raw_by_id = {row["id"]: row for row in raw}
    intact = data.collate(list(range(len(selected)))).to(device)
    reordered = data.collate(list(range(len(selected))), shuffle_seed=891).to(device)
    with autocast_context(device, precision):
        intact_logits = model(intact.model_inputs()).float().cpu()
        values = model(reordered.model_inputs()).float().cpu()
    original = {
        example.id: scored_row(
            {
                **raw_by_id[example.id],
                "candidate_ids": ids,
                "logits": logits[: len(ids)].tolist(),
            }
        )
        for example, ids, logits in zip(
            intact.examples, intact.candidate_ids, intact_logits, strict=True
        )
    }
    permutation_errors = []
    same_decision = []
    for example, ids, logits in zip(
        reordered.examples, reordered.candidate_ids, values, strict=True
    ):
        changed = scored_row(
            {
                **original[example.id],
                "candidate_ids": ids,
                "logits": logits[: len(ids)].tolist(),
            }
        )
        reference = original[example.id]
        for index, cid in enumerate(ids):
            permutation_errors.append(
                abs(
                    changed["probabilities"][index]
                    - reference["probabilities"][reference["candidate_ids"].index(cid)]
                )
            )
        same_decision.append(changed["prediction"] == reference["prediction"])
    without_context = copy.deepcopy(selected)
    for example in without_context:
        example.context = "No context was supplied."
    removed_rows = predict(
        model, EncodedDataset(without_context, tokenizer, model.config), device, precision
    )
    removed = [scored_row(row) for row in removed_rows]
    distractors = copy.deepcopy(selected)
    for example in distractors:
        example.choices.append(Choice("added-distractor", "quartz-extra-impossible-action"))
    distractor_rows = predict(
        model, EncodedDataset(distractors, tokenizer, model.config), device, precision
    )
    duplicated = []
    for example in selected:
        if example.target not in SENTINELS:
            changed = copy.deepcopy(example)
            answer = next(c.text for c in example.choices if c.id == example.target)
            changed.choices.append(Choice("equivalent-duplicate", answer))
            duplicated.append(changed)
    duplicate_rows = (
        predict(model, EncodedDataset(duplicated, tokenizer, model.config), device, precision)
        if duplicated
        else []
    )
    correct_semantic = []
    mass_changes = []
    for row in duplicate_rows:
        scored = scored_row(row)
        correct_semantic.append(scored["prediction"] in {row["target"], "equivalent-duplicate"})
        mass = sum(
            scored["probabilities"][row["candidate_ids"].index(cid)]
            for cid in [row["target"], "equivalent-duplicate"]
        )
        before = original[row["id"]]
        old_mass = before["probabilities"][before["candidate_ids"].index(row["target"])]
        mass_changes.append(mass - old_mass)
    return {
        "sample_count": len(selected),
        "sample_selection": {
            "method": "uniform row sample without replacement, independent of labels",
            "seed": 891,
            "kind_counts": dict(Counter(example.kind for example in selected)),
            "family_counts": dict(Counter(example.family for example in selected)),
            "note": "Exploratory probes; sample scores are not full-partition accuracy estimates.",
        },
        "intact_accuracy": float(np.mean([r["correct"] for r in original.values()])),
        "candidate_permutation": {
            "maximum_probability_change": max(permutation_errors, default=0),
            "same_decision_fraction": float(np.mean(same_decision)),
            "note": "Floating-point kernels need not be bitwise permutation invariant.",
        },
        "context_removed": {
            "original_target_accuracy": float(np.mean([row["correct"] for row in removed])),
            "same_prediction_fraction": float(
                np.mean([row["prediction"] == original[row["id"]]["prediction"] for row in removed])
            ),
            "note": (
                "Ablation against ORIGINAL labels, not valid accuracy for the modified problem. "
                "Measures dependence on evidence; does not evaluate an updated oracle."
            ),
        },
        "irrelevant_candidate_added": {
            "accuracy": evaluate_rows(distractor_rows)["accuracy"],
            "note": "The added candidate cannot be produced by these policy generators.",
        },
        "correct_candidate_duplicated": {
            "count": len(duplicate_rows),
            "semantic_accuracy": float(np.mean(correct_semantic)) if correct_semantic else None,
            "mean_correct_semantic_probability_mass_change": (
                float(np.mean(mass_changes)) if mass_changes else None
            ),
            "note": "Both equivalent IDs are accepted; duplicate stability is not guaranteed.",
        },
    }
