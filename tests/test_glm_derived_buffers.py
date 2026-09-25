from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bobcat.glm_derived_buffers import restore_vision_rotary_buffer


class Glm5NextVisionRotaryEmbedding(nn.Module):
    """Constructor spy: the loader must call it, not hard-code a frequency."""

    def __init__(self, dim):
        super().__init__()
        assert dim == 4
        self.register_buffer("inv_freq", torch.tensor([.25, .125]), persistent=False)


Glm5NextVisionRotaryEmbedding.__module__ = "nemo_automodel.components.models.glm5_next.vision"


def make_model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.visual = nn.Module()
    visual = model.model.visual
    visual.config = SimpleNamespace(hidden_size=32, num_heads=4)
    visual.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(4)
    visual.weight = nn.Parameter(torch.tensor([3.]))
    visual.register_buffer("router_state", torch.tensor([7.]))
    return model, visual.rotary_pos_emb


@pytest.mark.parametrize("initial", ["meta", "nan", "finite_garbage"])
def test_rebuilds_only_constructor_derived_buffer(initial):
    model, rotary = make_model()
    learned = {name: value.clone() for name, value in model.state_dict().items()}
    rotary.inv_freq = (
        torch.empty(2, device="meta", dtype=torch.bfloat16) if initial == "meta"
        else torch.full((2,), float("nan") if initial == "nan" else 42.,
                        dtype=torch.bfloat16)
    )
    result = restore_vision_rotary_buffer(model, torch.device("cpu"))
    assert torch.equal(rotary.inv_freq, torch.tensor([.25, .125], dtype=torch.bfloat16))
    assert result["constructor_reference_exact"]
    assert "model.visual.rotary_pos_emb.inv_freq" not in model.state_dict()
    assert all(torch.equal(value, learned[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("violation", ["persistent", "parameter", "layout", "class"])
def test_refuses_to_reinitialize_checkpoint_or_unknown_state(violation):
    model, rotary = make_model()
    if violation == "persistent":
        rotary._non_persistent_buffers_set.clear()
    elif violation == "parameter":
        rotary.weight = nn.Parameter(torch.ones(1))
    elif violation == "layout":
        rotary.inv_freq = torch.zeros(3)
    else:
        model.model.visual.rotary_pos_emb = nn.Identity()
    with pytest.raises(ValueError):
        restore_vision_rotary_buffer(model, torch.device("cpu"))
