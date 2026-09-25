import copy

import pytest
import torch

from bobcat.batching import EncodedDataset
from bobcat.model import DecisionModel
from bobcat.schema import Choice


def make_batch(corpus, config):
    dataset = EncodedDataset(corpus[1][:12], corpus[2], config)
    return dataset, dataset.collate(list(range(12)))


@pytest.mark.parametrize("pooling", ["marker", "candidate_mean"])
def test_candidate_permutation_equivariance(corpus, small_config, pooling):
    small_config.reader_passes = 2
    small_config.candidate_pooling = pooling
    model = DecisionModel(small_config).eval()
    dataset, batch = make_batch(corpus, small_config)
    reordered = dataset.collate(list(range(12)), shuffle_seed=991)
    with torch.no_grad():
        original = model(batch.tensors).softmax(-1)
        other = model(reordered.tensors).softmax(-1)
    for index, ids in enumerate(batch.candidate_ids):
        for col, cid in enumerate(ids):
            new_col = reordered.candidate_ids[index].index(cid)
            torch.testing.assert_close(
                original[index, col], other[index, new_col], atol=1e-6, rtol=1e-5
            )


@pytest.mark.parametrize("pooling", ["marker", "candidate_mean"])
def test_cached_and_fresh_state_are_equivalent(corpus, small_config, pooling):
    small_config.candidate_pooling = pooling
    model = DecisionModel(small_config).eval()
    dataset, batch = make_batch(corpus, small_config)
    fresh = dataset.collate(list(range(12)), reuse_context=False)
    assert batch.unique_contexts < fresh.unique_contexts
    with torch.no_grad():
        together = model(batch.tensors)
        uncached = model(fresh.tensors)
        memory = model.encode_state(batch.tensors["context_ids"])
        cached = model.decide(
            memory,
            batch.tensors["context_index"],
            batch.tensors["schema_ids"],
            batch.tensors["candidate_keep"],
            batch.tensors["schema_candidate_mask"],
        )
    torch.testing.assert_close(together, uncached, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(together, cached, atol=1e-6, rtol=1e-5)


def test_context_reuse_preserves_training_gradients(corpus, small_config):
    first = DecisionModel(small_config)
    second = copy.deepcopy(first)
    dataset, batch = make_batch(corpus, small_config)
    uncached = dataset.collate(list(range(12)), reuse_context=False)
    for model, value in [(first, batch), (second, uncached)]:
        loss = torch.nn.functional.cross_entropy(model(value.tensors), value.tensors["targets"])
        loss.backward()
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=2e-6)


def test_candidate_ids_never_enter_the_model(corpus, small_config):
    originals = corpus[1][:6]
    renamed = copy.deepcopy(originals)
    for example in renamed:
        mapping = {c.id: f"unrelated-api-key-{i}" for i, c in enumerate(example.choices)}
        example.choices = [Choice(mapping[c.id], c.text) for c in example.choices]
        example.target = mapping.get(example.target, example.target)
    first = EncodedDataset(originals, corpus[2], small_config).collate(list(range(6)))
    second = EncodedDataset(renamed, corpus[2], small_config).collate(list(range(6)))
    for key in first.tensors:
        assert torch.equal(first.tensors[key], second.tensors[key])


@pytest.mark.parametrize("architecture", ["shared", "joint"])
@pytest.mark.parametrize("pooling", ["marker", "candidate_mean"])
def test_variable_candidate_counts_and_backprop(corpus, small_config, architecture, pooling):
    small_config.architecture = architecture
    small_config.candidate_pooling = pooling
    model = DecisionModel(small_config)
    _, batch = make_batch(corpus, small_config)
    logits = model(batch.tensors)
    assert torch.isneginf(logits[~batch.tensors["candidate_keep"]]).all()
    assert torch.isfinite(logits[batch.tensors["candidate_keep"]]).all()
    loss = torch.nn.functional.cross_entropy(logits, batch.tensors["targets"])
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_long_policies_fail_instead_of_silent_truncation(corpus, small_config):
    small_config.max_context_tokens = 8
    with pytest.raises(ValueError, match="refusing to truncate"):
        EncodedDataset(corpus[1][:6], corpus[2], small_config)
