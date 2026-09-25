import pytest

from bobcat.glm_runtime_stability import PAD_ALLOCATION, zero_padding_patch


def test_padding_patch_changes_only_the_declared_allocator():
    text = (
        "vendor_license_and_model_logic\n"
        f"    {PAD_ALLOCATION}\n"
        "    input_tensor = buffer_init(shape)\n"
        "    input_tensor_scale = buffer_init(scale_shape)\n"
        "    m_indices = buffer_init(count)\n"
        "unchanged_kernel_arguments_and_weights\n"
    )
    changed = zero_padding_patch(text)
    assert changed.replace("buffer_init = torch.zeros", PAD_ALLOCATION, 1) == text
    assert "torch.empty" not in changed
    for invalid in (text + text, text.replace("m_indices = buffer_init(", "other("), changed):
        with pytest.raises(ValueError, match="pinned DeepGEMM"):
            zero_padding_patch(invalid)


@pytest.mark.parametrize("batch_drift", [False, True])
def test_runtime_controls_align_reverse_order_and_detect_same_argmax_batch_drift(
    tmp_path, monkeypatch, batch_drift,
):
    import bobcat.glm_runtime_stability as module

    rows = [{
        "id": f"q-{i}", "group_id": f"g-{i}", "input_sha256": f"{i:064x}",
        "kind": "categorical", "task": "fixture", "language": "ko",
        "language_origin": "native", "supervision": "hard_label",
        "target_index": 1, "score_mean": None, "input_tokens": 20,
        "option_token_ids": [32, 33],
    } for i in range(64)]
    monkeypatch.setattr(module, "read_suite", lambda _: ({}, rows))
    (tmp_path / "manifest.json").write_text("{}")
    calls = []

    class Scorer:
        def __init__(self, *_args, **_kwargs):
            pass

        def read(self, batch, *, label):
            calls.append((label, [row["id"] for row in batch]))
            logits = [0., 3.] if batch_drift and len(batch) > 1 else [0., 2.]
            return [{**row, "logits": logits} for row in batch], {
                "questions": len(batch), "native_http_seconds": .2,
                "input_tokens": 20 * len(batch),
            }

    monkeypatch.setattr(module, "CompiledReadout", Scorer)
    value = module.matched_controls(
        tmp_path, tmp_path / "out", client=None, model_path="/model", run_id="fixture",
    )
    assert value["serial_repeat"]["passed"]
    assert value["serial_vs_batch"]["passed"] is not batch_drift
    assert value["serial_vs_batch"]["argmax_changes"] == 0
    assert calls[128] == ("reverse-0", ["q-63"])
    assert len(calls) == 64 * 3 + 8
    assert value["latency"]["serial"]["all"]["requests"] == 64
