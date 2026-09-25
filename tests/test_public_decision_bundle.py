import hashlib
import json

import pytest
from test_checkpoints import MemoryS3

from bobcat.checkpoints import S3CheckpointStore
from bobcat.public_decision_bundle import collect, publish, restore, validate
from bobcat.public_decisions import SCHEMA, SPLITS
from bobcat.schema import file_hash, json_hash


@pytest.fixture
def sources(tmp_path):
    data, prior, support = (tmp_path / name for name in ("data", "prior", "support"))
    for path in (data, prior, support):
        path.mkdir()
    code = b'raise RuntimeError("Bundled source must never execute")\n'
    (data / "adapter.py").write_bytes(code)
    (prior / "adapter.py").write_bytes(code)
    (prior / "LICENSE.txt").write_text("Original Korean fixture attribution.\n")
    (support / "banking-license.txt").write_text("Banking fixture attribution.\n")
    prior_source = {"license_sha256": file_hash(prior / "LICENSE.txt")}
    (prior / "source.json").write_text(json.dumps(prior_source))
    prior_manifest = {
        "schema": "bobcat-korean-decisions-v1", "source": prior_source,
        "source_config_sha256": file_hash(prior / "source.json"),
        "adapter_sha256": file_hash(prior / "adapter.py"),
        "files": {f"{s}.jsonl": {"sha256": hashlib.sha256(s.encode()).hexdigest()}
                  for s in SPLITS},
    }
    (prior / "manifest.json").write_text(json.dumps(prior_manifest))
    source = {
        "schema": "bobcat-public-decision-sources-v1",
        "support": [{"path": "banking-license.txt",
                     "sha256": file_hash(support / "banking-license.txt")}],
    }
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source))
    files = {}
    for split in SPLITS:
        path = data / f"{split}.jsonl"
        path.write_text(json.dumps({"split": split, "score_target": 1.2727272727}) + "\n")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    (data / "audit.json").write_text('{"fixture_only":true}')
    manifest = {
        "schema": SCHEMA, "source": source, "source_config_sha256": file_hash(source_file),
        "files": files, "adapter_sha256": file_hash(data / "adapter.py"),
        "audit_sha256": file_hash(data / "audit.json"),
        "prior": {
            "source": prior_source, "manifest_sha256": file_hash(prior / "manifest.json"),
            "original_partitions_preserved": True,
            "partition_sha256": {s: prior_manifest["files"][f"{s}.jsonl"]["sha256"]
                                 for s in SPLITS},
        },
        "teacher_or_jev_labels_used": False, "mean_score_histograms_invented": False,
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data, source_file, prior, support


def test_roundtrip_preserves_means_sources_and_does_not_execute_code(sources, tmp_path):
    store = S3CheckpointStore("s3://fixture/runs/public-data", client=MemoryS3())
    bundle, paths = collect(*sources)
    published = publish(*sources, store)
    assert published["bundle"] == bundle
    recovered = tmp_path / "restored"
    result = restore(store, bundle["content_sha256"], recovered)
    assert result["bundled_code_executed"] is False
    for name, path in paths.items():
        assert (recovered / name).read_bytes() == path.read_bytes()
    assert file_hash(recovered / "manifest.json") == file_hash(sources[0] / "manifest.json")
    with pytest.raises(ValueError, match="new recovery"):
        restore(store, bundle["content_sha256"], recovered)


def test_corrupt_object_has_no_training_manifest_or_completion_marker(sources, tmp_path):
    store = S3CheckpointStore("s3://fixture/runs/public-data", client=MemoryS3())
    bundle = publish(*sources, store)["bundle"]
    entry = next(item for item in bundle["files"] if item["path"] == "train.jsonl")
    key = store.prefix + "/blobs/" + entry["sha256"]
    body, metadata = store.client.objects[store.bucket, key]
    store.client.objects[store.bucket, key] = (body.replace(b"1.2727272727", b"4.2727272727"),
                                              metadata)
    out = tmp_path / "corrupt"
    with pytest.raises(ValueError, match="checksum"):
        restore(store, bundle["content_sha256"], out)
    assert not (out / "manifest.json").exists()
    assert not (out / "bundle-restored.json").exists()


def test_bundle_identity_paths_and_prior_partition_chain_are_checked(sources):
    bundle, _ = collect(*sources)
    with pytest.raises(ValueError, match="checksum"):
        validate(bundle, "0" * 64)
    bundle["files"][0]["path"] = "../outside"
    bundle["content_sha256"] = json_hash({
        key: value for key, value in bundle.items() if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="Unsafe"):
        validate(bundle, bundle["content_sha256"])
    path = sources[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["prior"]["partition_sha256"]["train"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="partition"):
        collect(*sources)
