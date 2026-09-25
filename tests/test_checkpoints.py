import hashlib
import io
import json
from pathlib import Path

import pytest
import torch

from bobcat.checkpoints import (
    S3CheckpointStore,
    prune_uploaded_generations,
    verify_checkpoint,
    write_checkpoint,
)
from bobcat.corpus import atomic_json


def payload(step):
    return {
        "format": "bobcat-real-mlm-v1", "step": step,
        "model": {"weight": torch.arange(4.0) + step},
        "counters": {"tokens": step * 64}, "provenance": {"test_fixture": True},
    }


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class MemoryS3:
    """Protocol double; live S3 recovery is validated separately on the AWS node."""
    def __init__(self):
        self.objects = {}
        self.fail_upload = False

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise MissingObject()
        data, _ = self.objects[Bucket, Key]
        return {"Body": io.BytesIO(data), "ETag": hashlib.sha256(data).hexdigest()}

    def upload_file(self, path, bucket, key, ExtraArgs):
        if self.fail_upload:
            raise OSError("interrupted transfer")
        self.objects[bucket, key] = (Path(path).read_bytes(), ExtraArgs["Metadata"])

    def head_object(self, Bucket, Key, ChecksumMode):
        data, metadata = self.objects[Bucket, Key]
        return {"ContentLength": len(data), "Metadata": metadata, "VersionId": "test-version"}

    def put_object(self, Bucket, Key, Body, **kwargs):
        existing = self.objects.get((Bucket, Key))
        if kwargs.get("IfNoneMatch") and existing is not None:
            raise ValueError("conditional pointer conflict")
        if "IfMatch" in kwargs and (
            existing is None or hashlib.sha256(existing[0]).hexdigest() != kwargs["IfMatch"]
        ):
            raise ValueError("conditional pointer conflict")
        self.objects[Bucket, Key] = (Body, {})

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[bucket, key][0])


def test_generation_aliases_and_failed_save_preserve_completed_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / "last.pt"
    first = write_checkpoint(payload(1), target)
    second = write_checkpoint(payload(2), target)
    assert verify_checkpoint(target) == second
    assert verify_checkpoint(tmp_path / "previous.pt") == first

    def interrupted(_payload, path):
        Path(path).write_bytes(b"unfinished")
        raise OSError("simulated interruption before publication")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interruption"):
        write_checkpoint(payload(3), target)
    assert verify_checkpoint(target) == second
    assert verify_checkpoint(tmp_path / "previous.pt") == first
    assert not list(tmp_path.glob(".checkpoint-*.tmp"))


def test_checkpoint_integrity_is_checked_before_loading(tmp_path):
    target = tmp_path / "last.pt"
    write_checkpoint(payload(1), target)
    with target.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(target)


def test_remote_restore_checks_full_contents_and_failed_upload_does_not_advance(tmp_path):
    remote = MemoryS3()
    store = S3CheckpointStore("s3://unit-test/runs/fixture", client=remote)
    target = tmp_path / "source" / "last.pt"
    first = write_checkpoint(payload(1), target)
    store.publish(target, first)
    restored = store.restore(tmp_path / "recovered")
    assert verify_checkpoint(restored)["sha256"] == first["sha256"]
    torch.testing.assert_close(
        torch.load(restored, weights_only=True)["model"]["weight"], payload(1)["model"]["weight"],
    )
    second = write_checkpoint(payload(2), target)
    remote.fail_upload = True
    with pytest.raises(OSError, match="interrupted transfer"):
        store.publish(target, second)
    assert store._latest()[0]["metadata"]["step"] == 1
    receipt = store._latest()[0]["checkpoint"]
    data, metadata = remote.objects[store.bucket, receipt["key"]]
    remote.objects[store.bucket, receipt["key"]] = (bytes([data[0] ^ 1]) + data[1:], metadata)
    with pytest.raises(ValueError, match="checksum"):
        store.restore(tmp_path / "corrupted-restore")
    assert not (tmp_path / "corrupted-restore" / "last.pt").exists()


def test_remote_prefix_cannot_regress_or_change_provenance(tmp_path):
    store = S3CheckpointStore("s3://unit-test/runs/fixture", client=MemoryS3())
    target = tmp_path / "last.pt"
    second = write_checkpoint(payload(2), target)
    store.publish(target, second)
    with pytest.raises(ValueError, match="backwards"):
        store.publish(target, {**second, "step": 1})
    with pytest.raises(ValueError, match="provenance"):
        store.publish(target, {**second, "provenance": {"different_run": True}})


def test_retention_keeps_two_local_generations_and_unuploaded_files(tmp_path):
    store = S3CheckpointStore("s3://unit-test/runs/fixture", client=MemoryS3())
    target = tmp_path / "last.pt"
    for step in range(1, 4):
        record = write_checkpoint(payload(step), target)
        receipt = store.publish(target, record)
        atomic_json(target.resolve().with_suffix(".json"), {**record, "remote": receipt})
    unuploaded = tmp_path / "step-000000000-unuploaded.pt"
    unuploaded.write_bytes(b"diagnostic data")
    unuploaded.with_suffix(".json").write_text(json.dumps({"format": "bobcat-real-mlm-v1"}))
    prune_uploaded_generations(target)
    assert verify_checkpoint(target)["step"] == 3
    assert verify_checkpoint(tmp_path / "previous.pt")["step"] == 2
    assert unuploaded.exists()
    assert len(list(tmp_path.glob("step-*.pt"))) == 3
