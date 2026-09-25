"""Record the actual GLM expert implementation without changing its precision."""
from __future__ import annotations


def expert_backend_name(value="torch"):
    if value not in ("torch", "torch_mm"):
        raise ValueError("Select the original loop or BF16 grouped GEMM explicitly.")
    return value


def inspect_grouped_experts(model, expected, *, expected_count=42):
    expected = expert_backend_name(expected)
    found = []
    for name, module in model.named_modules():
        if not hasattr(module, "use_torch_mm"):
            continue
        if (type(module.use_torch_mm) is not bool
                or module.use_torch_mm != (expected == "torch_mm")
                or getattr(module, "use_mxfp8", None) is not False):
            raise ValueError("The constructed expert kernel or precision differs from its profile.")
        found.append(name)
    if len(found) != expected_count:
        raise ValueError("The number of constructed GLM expert modules changed.")
    return {
        "expert_backend": expected, "expert_modules": found,
        "expert_module_count": len(found), "mxfp8_enabled": False,
        "precision_changed": False, "performance_verified": False,
    }
