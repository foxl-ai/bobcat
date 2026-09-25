import hashlib
import json
import os

import pytest

from bobcat.glm_serving_assets import (
    portable_receipt,
    validate_native_source,
    validate_portable_assets,
    validate_verified_assets,
    verify,
)


def test_preverification_reads_shared_weights_once_and_rejects_later_changes(tmp_path):
    a, b = tmp_path / "original", tmp_path / "derived"
    a.mkdir()
    b.mkdir()
    raw = b"unaltered-base-weight"
    (a / "weight.bin").write_bytes(raw)
    os.link(a / "weight.bin", b / "weight.bin")
    file = {"path": "weight.bin", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    plan = tmp_path / "models.json"
    plan.write_text(json.dumps({
        "schema": "bobcat-glm-serving-model-set-v1", "models": [
            {"name": "original", "model_dir": str(a), "files": [file]},
            {"name": "derived", "model_dir": str(b), "files": [file]},
        ],
    }))
    receipt = tmp_path / "verified.json"
    result = verify(plan, receipt)
    assert result["verified_unique_files"] == 1
    assert result["model_file_references"] == 2
    assert validate_verified_assets(plan, receipt)["status"] == "completed"
    (a / "weight.bin").write_bytes(b"X" * len(raw))
    with pytest.raises(ValueError, match="changed after"):
        validate_verified_assets(plan, receipt)


def test_changed_bytes_fail_even_with_expected_file_size(tmp_path):
    root = tmp_path / "weights"
    root.mkdir()
    (root / "weight.bin").write_bytes(b"bad")
    plan = tmp_path / "models.json"
    plan.write_text(json.dumps({
        "schema": "bobcat-glm-serving-model-set-v1", "models": [{
            "name": "test", "model_dir": str(root), "files": [{
                "path": "weight.bin", "bytes": 3,
                "sha256": hashlib.sha256(b"good"[:3]).hexdigest(),
            }],
        }],
    }))
    receipt = tmp_path / "verified.json"
    with pytest.raises(ValueError, match="SHA256"):
        verify(plan, receipt)
    assert json.loads(receipt.read_text())["status"] == "failed"


def test_cross_host_receipt_requires_same_readonly_filesystem_and_original_source(
    tmp_path, monkeypatch,
):
    from bobcat import glm_serving_assets as assets

    root = tmp_path / "original"
    root.mkdir()
    data = b"original pinned bytes"
    weight = root / "weight.bin"
    weight.write_bytes(data)
    files = [{"path": weight.name, "bytes": len(data),
              "sha256": hashlib.sha256(data).hexdigest()}]
    plan = tmp_path / "models.json"
    plan.write_text(json.dumps({
        "schema": "bobcat-glm-serving-model-set-v1", "models": [{
            "name": "original", "role": "untuned_original_fp8",
            "model_dir": str(root), "files": files,
        }],
    }))
    cpu, portable = tmp_path / "cpu.json", tmp_path / "portable.json"
    verify(plan, cpu)
    mount = {"uuid": "same-volume", "fstype": "ext4", "read_only": False}
    monkeypatch.setattr(assets, "filesystem_identity", lambda _path: dict(mount))
    portable_receipt(plan, cpu, portable)
    with pytest.raises(ValueError, match="read-only"):
        validate_portable_assets(plan, portable)
    mount["read_only"] = True
    original_identity = assets.identity
    monkeypatch.setattr(
        assets, "identity", lambda path: {**original_identity(path), "device": 999999},
    )
    result = validate_native_source(plan, portable, root, {"revision": "pinned", "files": files})
    assert result["read_only_inode_size_mtime_verified"]
    assert result["full_sha256_repeated_on_gpu_host"] is False
    source_files = [{**row, "git_blob_id": "acquisition-metadata"} for row in files]
    assert validate_native_source(
        plan, portable, root, {"revision": "pinned", "files": source_files},
    )["files"] == result["files"]
    for key, value in (
        ("bytes", len(data) + 1), ("sha256", "0" * 64), ("path", "another-file.bin"),
    ):
        with pytest.raises(ValueError, match="exact original"):
            validate_native_source(
                plan, portable, root,
                {"revision": "pinned", "files": [{**source_files[0], key: value}]},
            )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_native_source(
            plan, portable, root, {"revision": "pinned", "files": source_files * 2},
        )
    mount["uuid"] = "different-volume"
    with pytest.raises(ValueError, match="unchanged"):
        validate_portable_assets(plan, portable)
    mount["uuid"] = "same-volume"
    with pytest.raises(ValueError, match="exact original"):
        validate_native_source(plan, portable, root, {"revision": "pinned", "files": []})
    weight.write_bytes(b"X" * len(data))
    with pytest.raises(ValueError, match="inode/size/mtime"):
        validate_portable_assets(plan, portable)
