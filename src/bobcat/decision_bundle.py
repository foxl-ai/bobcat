"""Restore the pinned Korean decision bundle without executing bundled code."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from bobcat.checkpoints import S3CheckpointStore
from bobcat.corpus import atomic_json
from bobcat.klue import validate_sources
from bobcat.schema import file_hash

DATA_FILES = {
    "train.jsonl", "dev_train.jsonl", "cal_temperature.jsonl", "cal_policy.jsonl",
    "dev_public.jsonl", "evaluation-input-exclusions.jsonl", "split-audit.json",
}
FILES = DATA_FILES | {"dataset-manifest.json", "adapter.py", "source.json", "LICENSE.txt"}


def validate_bundle(bundle: dict, store: S3CheckpointStore, manifest_hash: str,
                    max_bytes: int) -> dict[str, dict]:
    if (bundle.get("schema") != "bobcat-korean-data-s3-v1"
            or bundle.get("complete") is not True or bundle.get("errors") != []
            or bundle.get("dataset_manifest_sha256") != manifest_hash
            or bundle.get("bucket") != store.bucket or bundle.get("prefix") != store.prefix
            or bundle.get("license") != "CC-BY-SA-4.0"):
        raise ValueError("Invalid or incomplete pinned Korean data publication.")
    entries = {}
    for item in bundle["files"]:
        name = item["path"]
        if (name not in FILES or name in entries
                or not re.fullmatch("[0-9a-f]{64}", item["sha256"])
                or type(item["bytes"]) is not int or not 0 < item["bytes"] <= max_bytes
                or item["key"] != f"{store.prefix}/objects/{item['sha256']}/{name}"):
            raise ValueError("Invalid decision bundle file or object key.")
        entries[name] = item
    if (set(entries) != FILES or sum(f["bytes"] for f in entries.values()) > max_bytes
            or entries["dataset-manifest.json"]["sha256"] != manifest_hash):
        raise ValueError("Incomplete decision bundle or exceeded restore size limit.")
    return entries


def restore(store: S3CheckpointStore, manifest_hash: str, out: Path,
            max_bytes: int = 512 * 1024**2) -> dict:
    if not re.fullmatch("[0-9a-f]{64}", manifest_hash):
        raise ValueError("Use the frozen dataset manifest SHA256.")
    if out.exists():
        raise ValueError("Restore to a new directory; preserve prior data.")
    key = f"{store.prefix}/bundles/{manifest_hash}.json"
    stream = store.client.get_object(Bucket=store.bucket, Key=key)["Body"]
    try:
        body = stream.read(1024**2 + 1)
    finally:
        stream.close()
    if len(body) > 1024**2:
        raise ValueError("Oversized decision publication manifest.")
    bundle = json.loads(body)
    entries = validate_bundle(bundle, store, manifest_hash, max_bytes)
    out.mkdir(parents=True)

    def download(name: str):
        item = entries[name]
        path = out / name
        temporary = path.with_suffix(path.suffix + ".download")
        request = {"Bucket": store.bucket, "Key": item["key"]}
        if item.get("version_id"):
            request["VersionId"] = item["version_id"]
        response = store.client.get_object(**request)
        stream, digest, size = response["Body"], hashlib.sha256(), 0
        try:
            with temporary.open("xb") as target:
                while chunk := stream.read(min(1024**2, item["bytes"] + 1 - size)):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if size > item["bytes"]:
                        raise ValueError(f"Oversized restored decision file: {name}")
            if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Decision file checksum mismatch: {name}")
            temporary.replace(path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)

    # This pinned manifest anchors all the remaining file identities.
    download("dataset-manifest.json")
    manifest = json.loads((out / "dataset-manifest.json").read_text())
    if (manifest.get("schema") != "bobcat-korean-decisions-v1"
            or manifest.get("dataset_license") != "CC-BY-SA-4.0"
            or set(manifest["files"]) != DATA_FILES):
        raise ValueError("Unexpected source dataset contract.")
    validate_sources(manifest["source"])
    expected = {
        **{name: f["sha256"] for name, f in manifest["files"].items()},
        "source.json": manifest["source_config_sha256"],
        "adapter.py": manifest["adapter_sha256"],
        "LICENSE.txt": manifest["source"]["license_sha256"],
    }
    if any(entries[name]["sha256"] != sha for name, sha in expected.items()):
        raise ValueError("Publication files disagree with the pinned dataset manifest.")
    for name in sorted(FILES - {"dataset-manifest.json"}):
        download(name)
    if json.loads((out / "source.json").read_text()) != manifest["source"]:
        raise ValueError("Restored source contract differs from the dataset manifest.")
    # The trainer expects this filename. Keep the exact bytes and pinned digest.
    (out / "manifest.json").write_bytes((out / "dataset-manifest.json").read_bytes())
    if file_hash(out / "manifest.json") != manifest_hash:
        raise ValueError("Dataset manifest alias changed its identity.")
    receipt = {
        "schema": "bobcat-korean-data-restored-v1", "manifest_sha256": manifest_hash,
        "bucket": store.bucket, "manifest_key": key, "files": bundle["files"],
        "total_bytes": sum(f["bytes"] for f in entries.values()),
        "verification": "Every object restored and full SHA256 checked; source chain matched.",
        "bundled_adapter_executed": False, "model_training_or_quality_verified": False,
    }
    atomic_json(out / "bundle-restored.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--region")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = restore(
        S3CheckpointStore(args.s3, region=args.region), args.manifest_sha256, args.out,
    )
    print(json.dumps({
        "manifest_sha256": receipt["manifest_sha256"], "bytes": receipt["total_bytes"],
        "files": len(receipt["files"]), "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
