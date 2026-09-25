"""Checksummed immutable checkpoints with atomic aliases and optional S3 recovery."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash

CHECKPOINT_FORMATS = {
    "bobcat-real-mlm-v1", "bobcat-real-decisions-v1", "bobcat-glm-residual-v1",
}


def _alias(link: Path, target: Path) -> None:
    temporary = link.with_name(f".{link.name}.pending-link")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.name)
    temporary.replace(link)


def write_checkpoint(payload: dict, path: Path) -> dict:
    """Publish a generation only after both its weights and metadata exist.

    last.pt/previous.pt are relative symlinks. Readers resolve the weights first
    and read that generation's metadata, avoiding a two-file alias update race.
    """
    import torch

    if payload.get("format") not in CHECKPOINT_FORMATS:
        raise ValueError("Unsupported checkpoint format.")
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.resolve(strict=True) if path.exists() else None
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        digest = file_hash(temporary)
        generation = path.parent / f"step-{payload['step']:09d}-{digest[:16]}.pt"
        temporary.replace(generation)
        record = {
            "format": payload["format"], "step": payload["step"],
            "counters": payload["counters"], "provenance": payload["provenance"],
            "bytes": generation.stat().st_size, "sha256": digest,
            "filename": generation.name,
        }
        atomic_json(generation.with_suffix(".json"), record)
        if old is not None and old != generation:
            _alias(path.with_name("previous.pt"), old)
        _alias(path, generation)
        _alias(path.with_suffix(".json"), generation.with_suffix(".json"))
        return record
    finally:
        temporary.unlink(missing_ok=True)


def verify_checkpoint(path: Path, *, expected_format: str | None = None) -> dict:
    target = path.resolve(strict=True)
    record = json.loads(target.with_suffix(".json").read_text())
    if (record.get("format") not in CHECKPOINT_FORMATS
            or (expected_format is not None and record["format"] != expected_format)
            or target.stat().st_size != record["bytes"]
            or file_hash(target) != record["sha256"]):
        raise ValueError("Checkpoint checksum, size or format mismatch.")
    return record


def prune_uploaded_generations(path: Path) -> None:
    """Keep two local generations; only remove others verified as uploaded."""
    keep = {path.resolve(strict=True)}
    previous = path.with_name("previous.pt")
    if previous.exists():
        keep.add(previous.resolve(strict=True))
    for candidate in path.parent.glob("step-*.pt"):
        metadata = candidate.with_suffix(".json")
        if candidate.resolve() in keep or not metadata.exists():
            continue
        record = json.loads(metadata.read_text())
        if (record.get("format") in CHECKPOINT_FORMATS
                and record.get("sha256")
                and record.get("remote", {}).get("sha256") == record.get("sha256")):
            candidate.unlink()
            metadata.unlink()


class S3CheckpointStore:
    def __init__(self, uri: str, *, client=None, region: str | None = None):
        parsed = urlparse(uri)
        prefix = parsed.path.strip("/")
        if (parsed.scheme != "s3" or not parsed.netloc or not prefix.startswith("runs/")
                or ".." in prefix.split("/") or parsed.query or parsed.fragment):
            raise ValueError("Use s3://bucket/runs/a-unique-run-prefix.")
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client("s3", region_name=region, config=Config(
                connect_timeout=10, read_timeout=30, retries={"max_attempts": 2},
            ))
        self.client, self.bucket, self.prefix = client, parsed.netloc, prefix

    def _latest(self):
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=f"{self.prefix}/latest.json",
            )
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None, None
            raise
        return json.loads(response["Body"].read()), response["ETag"]

    def publish(self, path: Path, metadata: dict) -> dict:
        old, etag = self._latest()
        if old is not None:
            if old["metadata"]["provenance"] != metadata["provenance"]:
                raise ValueError("S3 run prefix already belongs to different provenance.")
            if old["metadata"]["step"] > metadata["step"]:
                raise ValueError("Refusing to move the remote checkpoint backwards.")
        key = f"{self.prefix}/checkpoints/{metadata['sha256']}.pt"
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs={
            "ChecksumAlgorithm": "SHA256", "Metadata": {"sha256": metadata["sha256"]},
        })
        head = self.client.head_object(Bucket=self.bucket, Key=key, ChecksumMode="ENABLED")
        if (head["ContentLength"] != metadata["bytes"]
                or head.get("Metadata", {}).get("sha256") != metadata["sha256"]):
            raise ValueError("Remote checkpoint size or identity mismatch.")
        receipt = {
            "key": key, "sha256": metadata["sha256"], "bytes": metadata["bytes"],
            "version_id": head.get("VersionId"),
            "verification": (
                "SDK transfer checksums plus S3 size/identity; a restore verifies full SHA256"
            ),
        }
        body = json.dumps({
            "format": "bobcat-remote-checkpoint-v1", "checkpoint": receipt,
            "metadata": metadata,
        }, allow_nan=False).encode()
        # Strongly consistent, conditional pointer publication. A competing
        # writer must not silently replace a newer completed checkpoint.
        self.client.put_object(
            Bucket=self.bucket, Key=f"{self.prefix}/latest.json", Body=body,
            ContentType="application/json", ChecksumAlgorithm="SHA256",
            **({"IfMatch": etag} if etag else {"IfNoneMatch": "*"}),
        )
        return receipt

    def restore(self, out: Path, max_bytes: int = 8 * 1024**3) -> Path:
        latest, _ = self._latest()
        if latest is None or latest.get("format") != "bobcat-remote-checkpoint-v1":
            raise ValueError("No completed remote checkpoint.")
        receipt, metadata = latest["checkpoint"], latest["metadata"]
        if (not 0 < receipt["bytes"] <= max_bytes
                or not receipt["key"].startswith(f"{self.prefix}/checkpoints/")
                or receipt["sha256"] != metadata["sha256"]):
            raise ValueError("Remote checkpoint identity or size is outside the recovery limit.")
        out.mkdir(parents=True, exist_ok=True)
        if (out / "last.pt").exists() or (out / "last.pt").is_symlink():
            raise ValueError("Restore into a new directory; do not overwrite a local checkpoint.")
        generation = out / f"step-{metadata['step']:09d}-{metadata['sha256'][:16]}.pt"
        temporary = generation.with_suffix(".download")
        try:
            self.client.download_file(self.bucket, receipt["key"], str(temporary))
            if temporary.stat().st_size != metadata["bytes"]:
                raise ValueError("Downloaded checkpoint size mismatch.")
            if file_hash(temporary) != metadata["sha256"]:
                raise ValueError("Downloaded checkpoint checksum mismatch.")
            temporary.replace(generation)
            atomic_json(generation.with_suffix(".json"), {**metadata, "remote": receipt})
            _alias(out / "last.pt", generation)
            _alias(out / "last.json", generation.with_suffix(".json"))
        finally:
            temporary.unlink(missing_ok=True)
        verify_checkpoint(out / "last.pt")
        return out / "last.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--region")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = S3CheckpointStore(args.s3, region=args.region).restore(args.out)
    print(json.dumps({"restored": str(result), "sha256": verify_checkpoint(result)["sha256"]}))


if __name__ == "__main__":
    main()
