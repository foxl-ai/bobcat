import json
import shutil

import pytest
from test_checkpoints import MemoryS3
from test_corpus import real_format_corpus as real_format_corpus

from bobcat.checkpoints import S3CheckpointStore
from bobcat.corpus import pack
from bobcat.corpus_bundle import collect, publish, restore, validate
from bobcat.schema import file_hash, json_hash


@pytest.fixture
def bundle_fixture(real_format_corpus, tmp_path):
    original, _, _, _ = real_format_corpus
    root = tmp_path / "source"
    root.mkdir()
    source = root / "fixture-sources.json"
    source.write_text('{"unit_test_only":true}\n')
    for name in ("tokenizer.json", "tokenizer.manifest.json", "downloads.json"):
        shutil.copyfile(original / name, root / name)
        if name != "tokenizer.json":
            value = json.loads((root / name).read_text())
            value["source_manifest_sha256"] = file_hash(source)
            (root / name).write_text(json.dumps(value))
    for language in ("ko", "en"):
        pack(root / "downloads.json", root / "tokenizer.json", root / "packed-512",
             language, 1024, 100, length=16)
    store = S3CheckpointStore("s3://fixture/runs/corpus", client=MemoryS3())
    return root, source, store


def test_bundle_roundtrip_uses_frozen_identity_and_full_checksums(bundle_fixture, tmp_path):
    root, source, store = bundle_fixture
    receipt = publish(root, source, store)
    result = restore(store, receipt["bundle"]["content_sha256"], tmp_path / "restored")
    assert result == receipt["bundle"]
    assert not result["raw_source_parquet_included"]
    for item in result["files"]:
        assert file_hash(tmp_path / "restored" / item["path"]) == item["sha256"]
    with pytest.raises(ValueError, match="new directory"):
        restore(store, result["content_sha256"], tmp_path / "restored")


def test_failed_upload_never_publishes_completion_manifest(bundle_fixture):
    root, source, store = bundle_fixture
    store.client.fail_upload = True
    with pytest.raises(OSError, match="interrupted"):
        publish(root, source, store)
    assert not any("/bundles/" in key for _, key in store.client.objects)
    assert not (root / "bundle-published.json").exists()


def test_corrupt_remote_data_cannot_be_marked_restored(bundle_fixture, tmp_path):
    root, source, store = bundle_fixture
    receipt = publish(root, source, store)
    item = receipt["bundle"]["files"][0]
    key = f"{store.prefix}/blobs/{item['sha256']}"
    data, metadata = store.client.objects[store.bucket, key]
    store.client.objects[store.bucket, key] = (bytes([data[0] ^ 1]) + data[1:], metadata)
    with pytest.raises(ValueError, match="checksum"):
        restore(store, receipt["bundle"]["content_sha256"], tmp_path / "bad")
    assert not (tmp_path / "bad/bundle-restored.json").exists()


def test_pending_or_modified_corpus_is_not_published(bundle_fixture):
    root, source, store = bundle_fixture
    path = root / "packed-512/ko-train.bin"
    with path.open("r+b") as stream:
        stream.write(b"xx")
    with pytest.raises(ValueError, match="checksum"):
        publish(root, source, store)
    assert not store.client.objects
    (root / "packed-512/ko.manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        collect(root, source)


def test_path_traversal_and_wrong_source_fail_before_restore(bundle_fixture):
    root, source, _ = bundle_fixture
    bundle, _ = collect(root, source)
    bundle["files"][0]["path"] = "../outside.json"
    bundle["content_sha256"] = json_hash({k: v for k, v in bundle.items()
                                         if k != "content_sha256"})
    with pytest.raises(ValueError, match="Unsafe"):
        validate(bundle, 64 * 1024**3)
    source.write_text('{"different":true}')
    with pytest.raises(ValueError, match="provenance"):
        collect(root, source)
