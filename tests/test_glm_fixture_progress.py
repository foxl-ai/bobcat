import json

from bobcat.glm_fsdp_adapter_probe import PhaseProgress


def test_external_timeout_retains_last_entered_phase_without_marking_success(tmp_path):
    path = tmp_path / "rank-0.json"
    record = {"rank": 0, "status": "running", "release_gate_passed": False}
    ticks = iter((10.0, 14.5, 17.0))
    progress = PhaseProgress(path, record, clock=lambda: next(ticks))
    progress.mark("checkpoint_load")
    progress.mark("baseline_forward_including_any_cold_compilation")
    # A killed process will never reach a finally block. This durable file
    # must still distinguish loading from the uncompleted forward operation.
    recovered = json.loads(path.read_text())
    assert recovered["status"] == "running"
    assert recovered["release_gate_passed"] is False
    assert recovered["phase_progress"]["current_phase"].startswith("baseline_forward")
    assert recovered["phase_progress"]["completed_phases"] == [
        {"phase": "checkpoint_load", "host_elapsed_seconds": 4.5}
    ]
    assert recovered["phase_progress"]["clock"] == (
        "host_monotonic_not_synchronized_gpu_timing"
    )


def test_phase_updates_preserve_source_and_failed_result(tmp_path):
    path = tmp_path / "rank-1.json"
    record = {"status": "running", "source": {"sha256": "source-pin"}}
    ticks = iter((0.0, 3.0, 9.0))
    progress = PhaseProgress(path, record, clock=lambda: next(ticks))
    progress.mark("first_forward")
    progress.mark("first_backward")
    record.update(status="failed", error="numerical mismatch")
    progress.mark("failure_export")
    recovered = json.loads(path.read_text())
    assert recovered["source"]["sha256"] == "source-pin"
    assert recovered["status"] == "failed"
    assert recovered["error"] == "numerical mismatch"
    assert recovered["phase_progress"]["completed_phases"] == [
        {"phase": "first_forward", "host_elapsed_seconds": 3.0},
        {"phase": "first_backward", "host_elapsed_seconds": 6.0},
    ]
