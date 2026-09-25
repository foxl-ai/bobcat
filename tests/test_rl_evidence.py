import hashlib
import io
import json

import pytest

from bobcat.rl_evidence import recover


class VersionedEvidence:
    def __init__(self):
        self.data = {f"run/rank-{r}.json": json.dumps({"rank": r}).encode()
                     for r in range(8)}
        self.marker = {
            "schema": "bobcat-native-rl-evaluation-v1", "label": "parent556",
            "files": {key.split("/")[-1]: hashlib.sha256(raw).hexdigest()
                      for key, raw in self.data.items()},
        }
        self.corrupt_key = None
        self.reads = []
        self.save_marker()

    def save_marker(self):
        self.data["run/complete.json"] = json.dumps(self.marker).encode()

    def head_object(self, *, Bucket, Key):
        raw = self.data[Key]
        return {"VersionId": "pinned-v1", "ContentLength": len(raw),
                "Metadata": {"sha256": hashlib.sha256(raw).hexdigest()}}

    def get_object(self, *, Bucket, Key, VersionId):
        assert VersionId == "pinned-v1"  # Never a mutable latest-version download.
        self.reads.append((Key, VersionId))
        raw = self.data[Key] if Key != self.corrupt_key else b"wrong"
        return {"VersionId": VersionId, "Body": io.BytesIO(raw)}


def test_recovery_requires_fresh_pinned_versions_before_completion(tmp_path):
    client = VersionedEvidence()
    result = recover(client, bucket="bucket", marker_key="run/complete.json", out=tmp_path)
    assert result["all_members_fresh_get_verified"]
    assert result["tensor_decode_performed"] is False
    assert len(result["members"]) == 8
    assert len(client.reads) == 9
    assert (tmp_path / "complete.json").read_bytes() == client.data["run/complete.json"]
    # A matching local copy still receives the independent fresh version check.
    recover(client, bucket="bucket", marker_key="run/complete.json", out=tmp_path)
    assert len(client.reads) == 18


def test_corrupt_missing_or_escaping_members_never_publish_completion(tmp_path):
    client = VersionedEvidence()
    client.corrupt_key = "run/rank-3.json"
    with pytest.raises(ValueError, match="immutable evidence"):
        recover(client, bucket="bucket", marker_key="run/complete.json", out=tmp_path / "bad")
    assert not (tmp_path / "bad/complete.json").exists()
    for bad in ("../escape", "/absolute", "rank-0.json\\escape"):
        client = VersionedEvidence()
        client.marker["files"][bad] = client.marker["files"].pop("rank-0.json")
        client.save_marker()
        with pytest.raises(ValueError, match="membership"):
            recover(client, bucket="bucket", marker_key="run/complete.json",
                    out=tmp_path / "escape")
    client = VersionedEvidence()
    del client.marker["files"]["rank-7.json"]
    client.save_marker()
    with pytest.raises(ValueError, match="eight evaluation"):
        recover(client, bucket="bucket", marker_key="run/complete.json", out=tmp_path / "missing")
