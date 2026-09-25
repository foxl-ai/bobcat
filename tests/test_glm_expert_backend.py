from types import SimpleNamespace

import pytest

from bobcat.glm_expert_backend import expert_backend_name, inspect_grouped_experts


class Model:
    def __init__(self, count=42, *, grouped=False, mxfp8=False):
        self.modules = [
            (f"layers.{i}.experts",
             SimpleNamespace(use_torch_mm=grouped, use_mxfp8=mxfp8))
            for i in range(count)
        ]

    def named_modules(self):
        return iter(self.modules)


def test_actual_expert_inventory_and_precision_must_match():
    assert inspect_grouped_experts(Model(), "torch")["expert_module_count"] == 42
    assert inspect_grouped_experts(Model(grouped=True), "torch_mm")["mxfp8_enabled"] is False
    with pytest.raises(ValueError, match="kernel or precision"):
        inspect_grouped_experts(Model(grouped=True), "torch")
    with pytest.raises(ValueError, match="kernel or precision"):
        inspect_grouped_experts(Model(grouped=True, mxfp8=True), "torch_mm")
    with pytest.raises(ValueError, match="number"):
        inspect_grouped_experts(Model(count=41), "torch")


def test_unknown_backend_cannot_silently_change_precision():
    for value in ("torch_mm_mxfp8", "te", "triton", None, True):
        with pytest.raises(ValueError):
            expert_backend_name(value)
