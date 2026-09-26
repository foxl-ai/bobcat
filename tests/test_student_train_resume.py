"""`student_train --save-every/--resume` continues exactly where an interrupted run stopped.
GPU only (the trainer binds a CUDA device); the GPU host's preflight runs it."""

import argparse
import json
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
if not torch.cuda.is_available():
    pytest.skip("the trainer needs a CUDA device", allow_module_level=True)

from bobcat import student_train  # noqa: E402


def tiny_model(folder):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2],
                         "mrope_interleaved": True},
    )
    Qwen3_5ForCausalLM(config).save_pretrained(folder)


def arguments(model, train, out, **extra):
    base = dict(model_dir=model, train=train, out=out, eval=[], readout="vocab",
                objective="sft", init=None, epochs=1.0, steps=6, accumulation=2, lr=1e-3,
                head_lr=1e-3, lora_rank=4, seed=5, log_every=1, max_seconds=None,
                eval_only=False, head_only=False, pointer_scale_init=0.0, keep_order=False,
                save_every=3, resume=None)
    base.update(extra)
    out.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(**base)


def lora(folder):
    from safetensors.torch import load_file

    return load_file(str(folder / "adapter" / "lora" / "adapter_model.safetensors"))


def test_resume_matches_an_uninterrupted_run(tmp_path):
    model = tmp_path / "model"
    tiny_model(model)
    rng = random.Random(1)
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps({
        "id": f"r{i}", "input_ids": [rng.randrange(7, 97) for _ in range(rng.randrange(8, 30))],
        "option_ids": [3, 4, 5], "candidate_ends": [0, 1, 2], "supervision": "hard_label",
        "target": i % 3, "score_target": None, "input_sha256": str(i)}) + "\n"
        for i in range(40)))
    whole = tmp_path / "whole"
    student_train.train(arguments(model, train, whole))
    assert (whole / "checkpoint" / "state.pt").exists()
    assert json.loads((whole / "checkpoint" / "step.json").read_text())["step"] == 2
    resumed = tmp_path / "resumed"
    student_train.train(arguments(model, train, resumed,
                                  init=whole / "checkpoint" / "adapter",
                                  resume=whole / "checkpoint" / "state.pt"))
    summary = json.loads((resumed / "train-summary.json").read_text())
    assert summary["resumed_at_step"] == 3 and summary["steps_run"] == 6
    a, b = lora(whole), lora(resumed)
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key].float(), b[key].float(), atol=1e-3, rtol=1e-2), key
    # A schedule that differs from the saved one is refused.
    with pytest.raises(ValueError):
        student_train.train(arguments(model, train, tmp_path / "bad", steps=8,
                                      init=whole / "checkpoint" / "adapter",
                                      resume=whole / "checkpoint" / "state.pt"))
