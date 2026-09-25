import copy

import torch

from bobcat.batching import EncodedDataset
from bobcat.paired import counterfactual_loss
from bobcat.schema import Choice


def paired_batch(corpus, small_config):
    data = EncodedDataset(corpus[1], corpus[2], small_config)
    indices = data.training_indices(3, 4, 17, strategy="world")
    return data.collate(indices, shuffle_seed=77)


def test_world_sampler_keeps_counterfactuals_together(corpus, small_config):
    batch = paired_batch(corpus, small_config)
    worlds = {example.group_id for example in batch.examples}
    assert len(worlds) == 4
    assert len(batch.examples) == 24
    assert batch.unique_contexts == 8
    for world in worlds:
        assert {e.variant for e in batch.examples if e.group_id == world} == {
            "base",
            "counterfactual",
        }


def test_pair_objective_uses_ids_after_candidate_shuffling(corpus, small_config):
    batch = paired_batch(corpus, small_config)
    logits = torch.zeros(batch.tensors["candidate_keep"].shape)
    neutral, count = counterfactual_loss(logits, batch)
    assert count > 0
    for index, example in enumerate(batch.examples):
        logits[index, batch.candidate_ids[index].index(example.target)] = 5
    correct, correct_count = counterfactual_loss(logits, batch)
    assert correct_count == count
    assert correct < neutral / 100


def test_pair_loss_has_finite_nonzero_gradients(corpus, small_config):
    batch = paired_batch(corpus, small_config)
    logits = torch.zeros(batch.tensors["candidate_keep"].shape, requires_grad=True)
    loss, count = counterfactual_loss(logits, batch)
    assert count > 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_reused_ids_with_changed_meanings_are_not_contrasted(corpus, small_config):
    batch = paired_batch(corpus, small_config)
    altered = copy.deepcopy(batch)
    for example in altered.examples:
        if example.variant == "counterfactual":
            example.choices = [Choice(c.id, "changed meaning: " + c.text) for c in example.choices]
    loss, count = counterfactual_loss(torch.zeros(batch.tensors["candidate_keep"].shape), altered)
    assert count == 0
    assert loss == 0
