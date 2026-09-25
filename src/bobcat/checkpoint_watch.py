"""Read-only checkpoint discovery and verified downloads for a separate evaluator.

The trainer owns publication. A completed marker is the sole readiness signal;
listing a shard, a changed mtime, or a decreasing training loss is insufficient.
This module neither contacts the trainer nor controls its GPU or optimizer.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash

_CHECKPOINT = re.compile(r"checkpoint-(\d{6})/complete\.json$")
_DIGEST = re.compile(r"[0-9a-f]{64}$")


def candidate_markers(keys, prefix, evaluated_steps=(), *, minimum_step=1):
    """Coalesce a backlog: evaluate the newest complete state, record skipped ones."""
    completed = set(evaluated_steps)
    choices = []
    for key in keys:
        if not key.startswith(prefix):
            raise ValueError("A listing contains an object outside the producer prefix.")
        match = _CHECKPOINT.fullmatch(key[len(prefix):])
        if match:
            step = int(match.group(1))
            if step >= minimum_step and step not in completed:
                choices.append((step, key))
    choices.sort()
    if not choices:
        return None, []
    return choices[-1], [step for step, _ in choices[:-1]]


def read_version(client, bucket, key, *, version=None, max_bytes=64 * 1024**2):
    kwargs = {"Bucket": bucket, "Key": key}
    if version is not None:
        kwargs["VersionId"] = version
    response = client.get_object(**kwargs)
    actual_version = response.get("VersionId")
    if not actual_version or actual_version == "null":
        response["Body"].close()
        raise ValueError("The evaluation source must be versioned.")
    with response["Body"] as body:
        if response["ContentLength"] > max_bytes:
            raise ValueError("An evaluation checkpoint object exceeds the frozen size bound.")
        raw = body.read(max_bytes + 1)
    if len(raw) != response["ContentLength"] or len(raw) > max_bytes:
        raise ValueError("Incomplete or oversized checkpoint object.")
    if version is not None and actual_version != version:
        raise ValueError("The requested immutable version changed.")
    return raw, {
        "key": key, "version_id": actual_version, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def validate_marker(marker, *, step, revision, curriculum_sha256):
    if (marker.get("step") != step or marker.get("world_size") != 8
            or marker.get("source_revision") != revision
            or marker.get("curriculum_sha256") != curriculum_sha256
            or not isinstance(marker.get("files"), dict)):
        raise ValueError("Checkpoint lineage differs from the frozen training run.")
    files = marker["files"]
    if not 10 <= len(files) <= 64 or ".metadata" not in files:
        raise ValueError("Expected a complete bounded eight-rank DCP checkpoint.")
    for name, digest in files.items():
        p = PurePosixPath(name)
        if (p.is_absolute() or ".." in p.parts or len(p.parts) != 1
                or name == "complete.json" or not _DIGEST.fullmatch(digest)):
            raise ValueError("Malformed checkpoint path or digest.")
    if not all(f"rng-rank-{rank}.pt" in files for rank in range(8)):
        raise ValueError("The producer has not published every rank's state.")
    if not any(name.endswith(".distcp") for name in files):
        raise ValueError("The marker contains no model shards.")
    return files


def producer_terminal(record, *, prefix, revision, curriculum_sha256):
    """Only a matching final export can end the evaluator's checkpoint wait."""
    job = record.get("job", {})
    if (job.get("s3_prefix", "") + "train/" != prefix
            or job.get("source_revision") != revision
            or job.get("curriculum_manifest_sha256") != curriculum_sha256):
        raise ValueError("The producer status has different checkpoint lineage.")
    return (record.get("status") in ("completed", "failed")
            and bool(record.get("finished_at")) and "evidence" in record
            and not record.get("export_error"))


def download_checkpoint(client, bucket, key, out: Path, *, revision, curriculum_sha256):
    """Read an immutable marker, retrieve its complete tree, verify every digest."""
    if out.exists():
        raise ValueError("Preserve an earlier evaluation download and its version receipts.")
    match = _CHECKPOINT.search(key)
    if match is None:
        raise ValueError("Use the producer's checkpoint completion marker.")
    step = int(match.group(1))
    raw, receipt = read_version(client, bucket, key, max_bytes=1024**2)
    marker = json.loads(raw)
    files = validate_marker(
        marker, step=step, revision=revision, curriculum_sha256=curriculum_sha256,
    )
    out.mkdir(parents=True)
    prefix = key.rsplit("/", 1)[0] + "/"

    def fetch(item):
        name, expected = item
        content, metadata = read_version(client, bucket, prefix + name)
        if metadata["sha256"] != expected:
            raise ValueError(f"Checkpoint file differs from the published digest: {name}")
        with (out / name).open("xb") as stream:
            stream.write(content)
        return {"path": name, **metadata}

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(fetch, files.items()))
    # Re-read the marker's exact version after its files have arrived.
    checked, _ = read_version(client, bucket, key, version=receipt["version_id"],
                              max_bytes=1024**2)
    if checked != raw:
        raise ValueError("Completion marker changed across versioned reads.")
    (out / "complete.json").write_bytes(raw)
    result = {
        "schema": "bobcat-evaluation-checkpoint-receipt-v1",
        "step": step, "source_revision": revision, "curriculum_sha256": curriculum_sha256,
        "complete_sha256": file_hash(out / "complete.json"), "marker": receipt,
        "files": receipts, "verified": True, "training_modified": False,
        "evaluation_performed": False,
    }
    atomic_json(out / "download-verified.json", result)
    return result
