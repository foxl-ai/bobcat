"""Recover a completed native RL artifact using pinned S3 versions and hashes.

Only downloads evidence. It never loads pickle, mutates AWS resources, reads a
live optimizer or treats an incomplete directory as a recoverable checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash

MAX_MEMBER_BYTES = 128 * 1024**2
FORMATS = {"bobcat-native-rl-checkpoint-v1", "bobcat-native-rl-evaluation-v1"}


def read_version(client, bucket, key, *, expected=None, limit=MAX_MEMBER_BYTES):
    head = client.head_object(Bucket=bucket, Key=key)
    version = head.get("VersionId")
    digest = head.get("Metadata", {}).get("sha256")
    if (not version or version == "null" or not isinstance(digest, str)
            or len(digest) != 64 or head["ContentLength"] > limit
            or (expected is not None and digest != expected)):
        raise ValueError("Evidence must have a bounded, hashed, versioned S3 object.")
    response = client.get_object(Bucket=bucket, Key=key, VersionId=version)
    with response["Body"] as stream:
        raw = stream.read(limit + 1)
    if (response.get("VersionId") != version or len(raw) != head["ContentLength"]
            or hashlib.sha256(raw).hexdigest() != digest):
        raise ValueError("Downloaded S3 version does not match the immutable evidence.")
    return raw, {
        "key": key, "version_id": version, "sha256": digest, "bytes": len(raw),
        "fresh_get_verified": True,
    }


def recover(client, *, bucket: str, marker_key: str, out: Path,
            expected_recipe: str | None = None) -> dict:
    key = PurePosixPath(marker_key)
    if key.name != "complete.json" or ".." in key.parts or key.is_absolute():
        raise ValueError("Use the exact relative key of a completed artifact.")
    raw, marker_receipt = read_version(client, bucket, marker_key, limit=256 * 1024)
    marker = json.loads(raw)
    files = marker.get("files", {})
    if (marker.get("schema") not in FORMATS or not isinstance(files, dict)
            or not 1 <= len(files) <= 32
            or any(not name or PurePosixPath(name).name != name or name in (
                ".", "..", "complete.json", "recovery.json",
            ) or "\\" in name for name in files)):
        raise ValueError("Invalid or incomplete native RL artifact membership.")
    if expected_recipe is not None and marker.get("recipe_sha256") != expected_recipe:
        raise ValueError("The checkpoint belongs to another frozen training recipe.")
    expected_names = ({f"rank-{r}.json" for r in range(8)}
                      if marker["schema"] == "bobcat-native-rl-evaluation-v1" else None)
    if expected_names is not None and set(files) != expected_names:
        raise ValueError("All eight evaluation ranks must be present.")
    if expected_names is None and (
            marker.get("world_size") != 8 or not {
                ".metadata", *(f"state-rank-{r}.pt" for r in range(8)),
            }.issubset(files) or not any(name.endswith(".distcp") for name in files)):
        raise ValueError("Require a complete eight-rank native continuation checkpoint.")
    out.mkdir(parents=True, exist_ok=True)
    if out.is_symlink():
        raise ValueError("Do not recover into a symlink.")

    def member(item):
        name, expected = item
        data, receipt = read_version(
            client, bucket, str(key.parent / name), expected=expected,
        )
        target = out / name
        if target.is_symlink() or (target.exists() and file_hash(target) != expected):
            raise ValueError("Never replace different or linked local evidence.")
        if not target.exists():
            temporary = out / (name + ".recovering")
            if temporary.exists() or temporary.is_symlink():
                raise ValueError("A prior recovery needs inspection before retry.")
            temporary.write_bytes(data)
            temporary.replace(target)
        return name, receipt

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = dict(pool.map(member, files.items()))
    target = out / "complete.json"
    if target.is_symlink() or (target.exists() and target.read_bytes() != raw):
        raise ValueError("An existing completion marker differs.")
    # Publish the original marker only after every member passed fresh version GET.
    if not target.exists():
        temporary = out / "complete.json.recovering"
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("A prior completion-marker recovery needs inspection.")
        temporary.write_bytes(raw)
        temporary.replace(target)
    result = {
        "schema": "bobcat-native-rl-recovery-v1", "bucket": bucket,
        "marker": marker_receipt, "members": receipts,
        "downloaded_bytes": len(raw) + sum(r["bytes"] for r in receipts.values()),
        "all_members_fresh_get_verified": True,
        "tensor_decode_performed": False, "gpu_resume_performed": False,
    }
    atomic_json(out / "recovery.json", result)
    return result


def main():
    import boto3
    from botocore.config import Config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--marker-key", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-recipe")
    args = parser.parse_args()
    client = boto3.client("s3", region_name=args.region, config=Config(
        connect_timeout=5, read_timeout=30, retries={"total_max_attempts": 1},
    ))
    result = recover(client, bucket=args.bucket, marker_key=args.marker_key,
                     out=args.out, expected_recipe=args.expected_recipe)
    print(json.dumps({k: result[k] for k in (
        "downloaded_bytes", "all_members_fresh_get_verified",
        "tensor_decode_performed", "gpu_resume_performed",
    )}))


if __name__ == "__main__":
    main()
