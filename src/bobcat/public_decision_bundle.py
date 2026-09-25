"""Transfer frozen public judgment partitions and their complete source identity chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from bobcat.checkpoints import S3CheckpointStore
from bobcat.corpus import atomic_json
from bobcat.public_decisions import SCHEMA, SPLITS
from bobcat.schema import file_hash, json_hash

BUNDLE_SCHEMA = "bobcat-public-decision-bundle-v1"
MAX_BYTES = 768 * 1024**2
METADATA = {"dataset-manifest.json", "source.json", "prior-manifest.json"}


def expected_files(manifest: dict, source: dict, prior: dict) -> dict[str, str]:
    if (manifest.get("schema") != SCHEMA
            or source.get("schema") != "bobcat-public-decision-sources-v1"
            or prior.get("schema") != "bobcat-korean-decisions-v1"
            or manifest["source"] != source or manifest["prior"]["source"] != prior["source"]
            or manifest["prior"]["original_partitions_preserved"] is not True
            or set(manifest["files"]) != {f"{s}.jsonl" for s in SPLITS}
            or manifest["teacher_or_jev_labels_used"] is not False
            or manifest["mean_score_histograms_invented"] is not False):
        raise ValueError("The public dataset source/partition contract differs.")
    if any(manifest["prior"]["partition_sha256"][s] != prior["files"][f"{s}.jsonl"]["sha256"]
           for s in SPLITS):
        raise ValueError("The original Korean partition identities changed.")
    expected = {
        **{name: item["sha256"] for name, item in manifest["files"].items()},
        "audit.json": manifest["audit_sha256"], "adapter.py": manifest["adapter_sha256"],
        "source.json": manifest["source_config_sha256"],
        "prior-manifest.json": manifest["prior"]["manifest_sha256"],
        "prior-adapter.py": prior["adapter_sha256"],
        "prior-source.json": prior["source_config_sha256"],
        "klue-license.txt": prior["source"]["license_sha256"],
    }
    for item in source["support"]:
        name = item["path"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Source support must have a flat, safe filename.")
        name = "support-" + name
        if name in expected:
            raise ValueError("Duplicate source support identity.")
        expected[name] = item["sha256"]
    return expected


def validate(bundle: dict, expected_hash: str, max_bytes: int = MAX_BYTES) -> dict[str, dict]:
    if (not re.fullmatch("[0-9a-f]{64}", expected_hash)
            or bundle.get("schema") != BUNDLE_SCHEMA
            or bundle.get("content_sha256") != expected_hash
            or json_hash({k: v for k, v in bundle.items() if k != "content_sha256"})
            != expected_hash):
        raise ValueError("Public data bundle checksum or schema differs.")
    entries = {}
    for item in bundle["files"]:
        name = item["path"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                or name in entries or name in {"manifest.json", "bundle-restored.json"}
                or type(item["bytes"]) is not int or not 0 < item["bytes"] <= max_bytes
                or not re.fullmatch("[0-9a-f]{64}", item["sha256"])):
            raise ValueError("Unsafe public data file identity or size.")
        entries[name] = item
    total = sum(item["bytes"] for item in entries.values())
    if (not METADATA <= entries.keys() or not 0 < total <= max_bytes
            or total != bundle["total_bytes"]
            or bundle["dataset_manifest_sha256"]
            != entries["dataset-manifest.json"]["sha256"]):
        raise ValueError("Incomplete or oversized public data bundle.")
    return entries


def collect(data: Path, sources: Path, prior_root: Path, support: Path) -> tuple[dict, dict]:
    manifest = json.loads((data / "manifest.json").read_text())
    source = json.loads(sources.read_text())
    prior = json.loads((prior_root / "manifest.json").read_text())
    expected = expected_files(manifest, source, prior)
    paths = {
        **{name: data / name for name in manifest["files"]},
        "dataset-manifest.json": data / "manifest.json",
        "audit.json": data / "audit.json", "adapter.py": data / "adapter.py",
        "source.json": sources, "prior-manifest.json": prior_root / "manifest.json",
        "prior-adapter.py": prior_root / "adapter.py",
        "prior-source.json": prior_root / "source.json",
        "klue-license.txt": prior_root / "LICENSE.txt",
        **{"support-" + item["path"]: support / item["path"] for item in source["support"]},
    }
    files = []
    for name, path in sorted(paths.items()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked public data source: {name}")
        sha = file_hash(path)
        if name in expected and sha != expected[name]:
            raise ValueError(f"Public data source checksum differs: {name}")
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha})
    bundle = {
        "schema": BUNDLE_SCHEMA, "dataset_manifest_sha256": file_hash(data / "manifest.json"),
        "files": files, "total_bytes": sum(item["bytes"] for item in files),
        "licenses": "per_source; original attribution and terms preserved",
        "bundled_code_must_not_be_executed": True,
    }
    bundle["content_sha256"] = json_hash(bundle)
    validate(bundle, bundle["content_sha256"])
    return bundle, paths


def publish(data: Path, sources: Path, prior: Path, support: Path,
            store: S3CheckpointStore) -> dict:
    bundle, paths = collect(data, sources, prior, support)
    receipts = []
    for item in bundle["files"]:
        key = f"{store.prefix}/blobs/{item['sha256']}"
        store.client.upload_file(str(paths[item["path"]]), store.bucket, key, ExtraArgs={
            "ChecksumAlgorithm": "SHA256", "Metadata": {"sha256": item["sha256"]},
        })
        head = store.client.head_object(Bucket=store.bucket, Key=key, ChecksumMode="ENABLED")
        if (head["ContentLength"] != item["bytes"]
                or head.get("Metadata", {}).get("sha256") != item["sha256"]):
            raise ValueError("Public data upload identity mismatch.")
        receipts.append({"path": item["path"], "key": key, "version_id": head.get("VersionId")})
    key = f"{store.prefix}/bundles/{bundle['content_sha256']}.json"
    store.client.put_object(
        Bucket=store.bucket, Key=key, Body=json.dumps(bundle, allow_nan=False).encode(),
        ContentType="application/json", ChecksumAlgorithm="SHA256",
    )
    receipt = {
        "schema": "bobcat-public-decision-published-v1", "bundle": bundle, "manifest_key": key,
        "files": receipts, "full_s3_restore_verified": False,
    }
    atomic_json(data / "bundle-published.json", receipt)
    return receipt


def restore(store: S3CheckpointStore, bundle_hash: str, out: Path,
            max_bytes: int = MAX_BYTES) -> dict:
    if out.exists() or not re.fullmatch("[0-9a-f]{64}", bundle_hash):
        raise ValueError("Use a pinned bundle checksum and a new recovery directory.")
    key = f"{store.prefix}/bundles/{bundle_hash}.json"
    stream = store.client.get_object(Bucket=store.bucket, Key=key)["Body"]
    try:
        raw = stream.read(1024**2 + 1)
    finally:
        stream.close()
    if len(raw) > 1024**2:
        raise ValueError("Oversized public data manifest.")
    bundle = json.loads(raw)
    entries = validate(bundle, bundle_hash, max_bytes)
    out.mkdir(parents=True)

    def download(name):
        item = entries[name]
        stream = store.client.get_object(
            Bucket=store.bucket, Key=f"{store.prefix}/blobs/{item['sha256']}",
        )["Body"]
        destination = out / name
        temporary = destination.with_suffix(destination.suffix + ".download")
        sha, size = hashlib.sha256(), 0
        try:
            with temporary.open("xb") as output:
                while chunk := stream.read(min(1024**2, item["bytes"] + 1 - size)):
                    size += len(chunk)
                    output.write(chunk)
                    sha.update(chunk)
                    if size > item["bytes"]:
                        raise ValueError(f"Oversized public data object: {name}")
            if size != item["bytes"] or sha.hexdigest() != item["sha256"]:
                raise ValueError(f"Public data object checksum differs: {name}")
            temporary.replace(destination)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)

    for name in sorted(METADATA):
        download(name)
    manifest, source, prior = (
        json.loads((out / name).read_text())
        for name in ("dataset-manifest.json", "source.json", "prior-manifest.json")
    )
    expected = expected_files(manifest, source, prior)
    if (set(entries) != expected.keys() | {"dataset-manifest.json"}
            or any(entries[name]["sha256"] != sha for name, sha in expected.items())):
        raise ValueError("The public bundle and source identity chain disagree.")
    for name in sorted(entries.keys() - METADATA):
        download(name)
    if json.loads((out / "prior-source.json").read_text()) != prior["source"]:
        raise ValueError("Original Korean source contract differs.")
    (out / "manifest.json").write_bytes((out / "dataset-manifest.json").read_bytes())
    receipt = {
        "schema": "bobcat-public-decision-restored-v1", "bundle_sha256": bundle_hash,
        "dataset_manifest_sha256": file_hash(out / "manifest.json"),
        "total_bytes": bundle["total_bytes"], "files": len(entries),
        "bucket": store.bucket, "manifest_key": key,
        "verification": "Fresh download, every full SHA256, and original source/partition chain.",
        "bundled_code_executed": False, "training_or_quality_verified": False,
    }
    atomic_json(out / "bundle-restored.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "restore"))
    parser.add_argument("--s3", required=True)
    parser.add_argument("--region")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    store = S3CheckpointStore(args.s3, region=args.region)
    if args.action == "publish":
        if any(value is None for value in (args.data, args.sources, args.prior, args.support)):
            parser.error("Publication requires data, sources, prior, and support.")
        receipt = publish(args.data, args.sources, args.prior, args.support, store)
        print(json.dumps({"bundle_sha256": receipt["bundle"]["content_sha256"],
                          "bytes": receipt["bundle"]["total_bytes"],
                          "full_s3_restore_verified": False}))
    else:
        if not args.bundle_sha256 or not args.out:
            parser.error("Recovery requires bundle-sha256 and out.")
        print(json.dumps(restore(store, args.bundle_sha256, args.out)))


if __name__ == "__main__":
    main()
