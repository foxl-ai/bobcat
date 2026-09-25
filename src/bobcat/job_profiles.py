"""Explicit bounded hardware profiles for the initial language experiments."""

LANGUAGE_PROFILES = {
    "l40s": {
        "world_size": 1, "supported_gpu_names": ["L40S"], "minimum_gpu_memory_mib": 44000,
        "batch_per_gpu": 16, "accumulation": 4,
        "max_tokens": 50_000_000, "max_seconds": 10800, "boot_hours": 4,
    },
    "a10g": {
        "world_size": 1, "supported_gpu_names": ["A10G"], "minimum_gpu_memory_mib": 22000,
        "batch_per_gpu": 4, "accumulation": 16,
        "max_tokens": 50_000_000, "max_seconds": 10800, "boot_hours": 4,
    },
    "h100": {
        "world_size": 8, "supported_gpu_names": ["H100"], "minimum_gpu_memory_mib": 78000,
        "batch_per_gpu": 16, "accumulation": 4,
        "max_tokens": 1_000_000_000, "max_seconds": 25200, "boot_hours": 8,
    },
}
