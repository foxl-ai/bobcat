import pytest
import torch

from bobcat import student_serve as ss


def test_common_prefix():
    assert ss.common_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    assert ss.common_prefix([[5, 6], [5, 6]]) == 2
    assert ss.common_prefix([[1], [2]]) == 0


def tiny_hybrid_student():
    """A random 4-layer Qwen3.5 text model (3 Gated DeltaNet layers + 1 full attention)."""
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        # 64 * 0.25 = 16 rotary dims = 8 frequencies, split over the three M-RoPE axes.
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2],
                         "mrope_interleaved": True},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server = ss.ServingStudent.__new__(ss.ServingStudent)
    server.torch, server.device, server.shared_prefix = torch, device, False
    server.backbone = Qwen3_5TextModel(config).to(device).float().eval()
    server.lm_weight = torch.randn(97, 64, device=device)
    return server


def test_shared_prefix_branches_match_full_sequences():
    server = tiny_hybrid_student()
    prefix = list(range(3, 90))
    sequences = [prefix + [5, 6, 7], prefix + [8, 9, 10, 11, 12, 13, 14, 15], prefix + [16]]
    options = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    full, full_tokens = server.logits(sequences, options, shared=False)
    shared, shared_tokens = server.logits(sequences, options, shared=True)
    assert shared_tokens == len(prefix) + 3 + 8 + 1 < full_tokens
    for a, b in zip(full, shared, strict=True):
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) < 2e-3
    alone = [server.logits([s], [o], shared=False)[0][0] for s, o in zip(sequences, options,
                                                                          strict=True)]
    for a, b in zip(alone, full, strict=True):  # right padding never reaches a read position
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) < 2e-3
