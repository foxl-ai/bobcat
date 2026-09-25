import hashlib
import json
from pathlib import Path

import pytest
from test_checkpoints import MemoryS3

from bobcat.checkpoints import S3CheckpointStore
from bobcat.decision_bundle import DATA_FILES, restore
from bobcat.schema import file_hash


@pytest.fixture
def publication():
    source = json.loads(
        (Path(__file__).parents[1] / "configs/korean-decisions-klue-v1.json").read_text()
    )
    bodies = {name: b"{}\n" for name in DATA_FILES}
    bodies["adapter.py"] = b'raise AssertionError("The adapter must never be executed")\n'
    bodies["LICENSE.txt"] = b"software fixture licence\n"
    source["license_sha256"] = hashlib.sha256(bodies["LICENSE.txt"]).hexdigest()
    bodies["source.json"] = json.dumps(source).encode()
    sha = lambda name: hashlib.sha256(bodies[name]).hexdigest()  # noqa: E731
    manifest = {
        "schema": "bobcat-korean-decisions-v1", "dataset_license": "CC-BY-SA-4.0",
        "source": source, "source_config_sha256": sha("source.json"),
        "adapter_sha256": sha("adapter.py"),
        "files": {name: {"sha256": sha(name), "bytes": len(bodies[name])} for name in DATA_FILES},
    }
    bodies["dataset-manifest.json"] = json.dumps(manifest).encode()
    manifest_hash = sha("dataset-manifest.json")
    store = S3CheckpointStore("s3://fixture/runs/decisions", client=MemoryS3())
    entries = []
    for name, body in bodies.items():
        key = f"{store.prefix}/objects/{sha(name)}/{name}"
        store.client.put_object(Bucket=store.bucket, Key=key, Body=body)
        entries.append({"path": name, "key": key, "bytes": len(body), "sha256": sha(name)})
    bundle = {
        "schema": "bobcat-korean-data-s3-v1", "complete": True, "errors": [],
        "bucket": store.bucket, "prefix": store.prefix, "license": "CC-BY-SA-4.0",
        "dataset_manifest_sha256": manifest_hash, "files": entries,
    }
    key = f"{store.prefix}/bundles/{manifest_hash}.json"
    store.client.put_object(Bucket=store.bucket, Key=key, Body=json.dumps(bundle).encode())
    return store, manifest_hash, entries, bundle


def test_full_restore_preserves_pinned_manifest_without_running_adapter(publication, tmp_path):
    store, digest, entries, _ = publication
    out = tmp_path / "restored"
    result = restore(store, digest, out)
    assert file_hash(out / "manifest.json") == digest
    assert result["bundled_adapter_executed"] is False
    for item in entries:
        assert file_hash(out / item["path"]) == item["sha256"]
    with pytest.raises(ValueError, match="new directory"):
        restore(store, digest, out)


def test_corrupt_training_data_never_gets_completion_receipt(publication, tmp_path):
    store, digest, entries, _ = publication
    item = next(f for f in entries if f["path"] == "train.jsonl")
    store.client.objects[store.bucket, item["key"]] = (b"xx\n", {})
    with pytest.raises(ValueError, match="checksum"):
        restore(store, digest, tmp_path / "bad")
    assert not (tmp_path / "bad/bundle-restored.json").exists()
    assert not (tmp_path / "bad/manifest.json").exists()


def test_publication_cannot_redirect_to_another_object(publication, tmp_path):
    store, digest, _, bundle = publication
    bundle["files"][0]["key"] = "other-project/data"
    store.client.put_object(
        Bucket=store.bucket, Key=f"{store.prefix}/bundles/{digest}.json",
        Body=json.dumps(bundle).encode(),
    )
    with pytest.raises(ValueError, match="object key"):
        restore(store, digest, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
