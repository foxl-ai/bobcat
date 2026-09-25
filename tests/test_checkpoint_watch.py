import hashlib
import io
import json

import pytest

from bobcat.checkpoint_watch import candidate_markers, download_checkpoint, producer_terminal

PREFIX = "owned-run/train/"
REVISION = "base-revision"
CURRICULUM = "c" * 64


def make_store(*, corrupt=False):
    prefix = PREFIX + "checkpoint-000064/"
    files = {".metadata": b"dcp", "__0_0.distcp": b"adapter"}
    files.update({f"rng-rank-{rank}.pt": bytes([rank]) for rank in range(8)})
    marker = {
        "step": 64, "world_size": 8, "source_revision": REVISION,
        "curriculum_sha256": CURRICULUM,
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
    }
    store = {prefix + name: raw for name, raw in files.items()}
    store[prefix + "complete.json"] = json.dumps(marker).encode()
    if corrupt:
        store[prefix + "__0_0.distcp"] = b"bad"

    class S3:
        def get_object(self, **kwargs):
            raw = store[kwargs["Key"]]
            return {"Body": io.BytesIO(raw), "ContentLength": len(raw), "VersionId": "v1"}

    return S3(), prefix + "complete.json"


def test_partial_checkpoint_is_not_ready_and_stale_evaluations_are_coalesced():
    keys = [PREFIX + "checkpoint-000001/complete.json",
            PREFIX + "checkpoint-000064/complete.json",
            PREFIX + "checkpoint-000128/__0_0.distcp"]
    chosen, skipped = candidate_markers(keys, PREFIX)
    assert chosen == (64, keys[1]) and skipped == [1]
    assert candidate_markers(keys, PREFIX, [1, 64]) == (None, [])
    with pytest.raises(ValueError):
        candidate_markers(["some-other-run/complete.json"], PREFIX)


def test_download_requires_the_published_complete_bytes_and_lineage(tmp_path):
    client, key = make_store()
    result = download_checkpoint(client, "bucket", key, tmp_path / "good",
                                 revision=REVISION, curriculum_sha256=CURRICULUM)
    assert result["verified"] and not result["evaluation_performed"]
    assert len(result["files"]) == 10
    client, key = make_store(corrupt=True)
    with pytest.raises(ValueError, match="digest"):
        download_checkpoint(client, "bucket", key, tmp_path / "bad",
                            revision=REVISION, curriculum_sha256=CURRICULUM)
    with pytest.raises(ValueError, match="lineage"):
        download_checkpoint(client, "bucket", key, tmp_path / "wrong",
                            revision="another-model", curriculum_sha256=CURRICULUM)


def test_failed_finished_producer_does_not_leave_a_paid_evaluator_waiting():
    record = {
        "job": {"s3_prefix": "owned-run/", "source_revision": REVISION,
                "curriculum_manifest_sha256": CURRICULUM},
        "status": "failed", "finished_at": "2026-09-23T00:00:00Z", "evidence": [],
    }
    kwargs = dict(prefix=PREFIX, revision=REVISION, curriculum_sha256=CURRICULUM)
    assert producer_terminal(record, **kwargs)
    assert not producer_terminal({**record, "finished_at": None}, **kwargs)
    assert not producer_terminal({**record, "export_error": "incomplete upload"}, **kwargs)
    with pytest.raises(ValueError, match="lineage"):
        producer_terminal(record, **{**kwargs, "revision": "different-base"})
