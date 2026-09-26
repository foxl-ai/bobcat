"""Packed (block-masked) scoring equals scoring every question alone, and questions cannot
see each other (question isolation). Tiny random Qwen3 model on CPU, FP32."""

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from bobcat import flash_packed  # noqa: E402
from bobcat.flash_packed import PackedScorer, block_mask, pack  # noqa: E402


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    config = transformers.Qwen3Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config)
    folder = tmp_path_factory.mktemp("tiny-qwen3")
    model.save_pretrained(folder)
    return PackedScorer(folder, device="cpu", dtype=torch.float32)


def request(rng, prefix_len, count, suffix_range=(3, 12)):
    prefix = [rng.randrange(5, 500) for _ in range(prefix_len)]
    sequences = [prefix + [rng.randrange(5, 500) for _ in range(rng.randint(*suffix_range))]
                 for _ in range(count)]
    options = [rng.sample(range(5, 500), rng.randint(2, 6)) for _ in range(count)]
    return sequences, options


def close(a, b, tol=1e-4):
    return all(abs(x - y) <= tol for u, v in zip(a, b, strict=True)
               for x, y in zip(u, v, strict=True))


def test_mask_shape_and_isolation():
    p = pack([[1, 2, 3, 4], [1, 2, 3, 5, 6]])
    assert p.prefix == 3 and p.ids == [1, 2, 3, 4, 5, 6]
    assert p.positions == [0, 1, 2, 3, 3, 4]
    mask = block_mask(torch, p.segments, p.positions)
    assert mask[3, :4].tolist() == [True, True, True, True]
    assert mask[4, 3].item() is False            # question 2 cannot see question 1
    assert mask[5, :].tolist() == [True, True, True, False, True, True]


def test_packed_equals_unpacked(scorer):
    rng = random.Random(1)
    for prefix_len, count in ((40, 1), (40, 5), (7, 12), (120, 3)):
        sequences, options = request(rng, prefix_len, count)
        packed = scorer.score_requests([(sequences, options)])[0]
        alone = scorer.score_unpacked(sequences, options)
        assert close(packed, alone), (prefix_len, count)


def test_staged_path_equals_unpacked(scorer, monkeypatch):
    monkeypatch.setattr(flash_packed, "SINGLE_MAX_TOKENS", 16)
    monkeypatch.setattr(flash_packed, "GROUP_TOKENS", 20)
    rng = random.Random(2)
    sequences, options = request(rng, 60, 9)
    packed = scorer.score_requests([(sequences, options)])[0]
    assert close(packed, scorer.score_unpacked(sequences, options))


def test_batched_requests_with_padding(scorer):
    rng = random.Random(3)
    requests = [request(rng, rng.randint(5, 80), rng.randint(1, 6)) for _ in range(5)]
    together = scorer.score_requests(requests)
    for (sequences, options), got in zip(requests, together, strict=True):
        assert close(got, scorer.score_unpacked(sequences, options))


def test_no_leakage_between_questions(scorer):
    rng = random.Random(4)
    sequences, options = request(rng, 30, 4)
    base = scorer.score_requests([(sequences, options)])[0]
    changed = [list(s) for s in sequences]
    changed[2] = changed[2][:30] + [7] * 20      # rewrite question 3 only
    after = scorer.score_requests([(changed, options)])[0]
    for q in (0, 1, 3):
        assert close([base[q]], [after[q]], tol=1e-6)
    assert not close([base[2]], [after[2]], tol=1e-6)


def test_identical_questions_share_one_segment(scorer):
    rng = random.Random(5)
    sequences, options = request(rng, 25, 2)
    sequences = [sequences[0], sequences[1], list(sequences[0])]
    options = [options[0], options[1], options[0]]
    p = pack(sequences)
    assert p.unique == [0, 1, 0]
    got = scorer.score_requests([(sequences, options)])[0]
    assert got[0] == got[2]
    assert close(got, scorer.score_unpacked(sequences, options))


def test_refuses_hybrid_or_sliding(tmp_path):
    config = transformers.Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["sliding_attention", "full_attention"], sliding_window=8,
        use_sliding_window=True)
    transformers.Qwen3ForCausalLM(config).save_pretrained(tmp_path)
    with pytest.raises(NotImplementedError):
        PackedScorer(tmp_path, device="cpu", dtype=torch.float32)
