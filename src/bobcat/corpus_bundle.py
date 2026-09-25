"""Publish only finalized corpus files, then restore them with full SHA256 checks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bobcat.checkpoints import S3CheckpointStore
from bobcat.corpus import MMapCorpus, atomic_json
from bobcat.schema import file_hash
from bobcat.schema import json_hash as digest
from bobcat.tokenization import ScratchTokenizer


def collect(root: Path, source: Path) -> tuple[dict, dict[str, Path]]:
    tokenizer = ScratchTokenizer(root / "tokenizer.json")
    names = ["tokenizer.json", "tokenizer.manifest.json", "downloads.json"]
    source_hash = file_hash(source)
    for name in ("downloads.json", "tokenizer.manifest.json"):
        if json.loads((root / name).read_text())["source_manifest_sha256"] != source_hash:
            raise ValueError("Tokenizer/download provenance differs from the source manifest.")
    verified_hashes = {}
    for language in ("ko", "en"):
        metadata = json.loads((root / "packed-512" / f"{language}.manifest.json").read_text())
        for split in ("train", "validation"):
            info = metadata["files"][split]
            if Path(info["path"]).name != info["path"]:
                raise ValueError("Expected a flat finalized corpus filename.")
            corpus = MMapCorpus(root / "packed-512", language, split, tokenizer.digest)
            if corpus.manifest["source_manifest_sha256"] != source_hash:
                raise ValueError("Finalized corpus belongs to a different source manifest.")
            info = corpus.manifest["files"][split]
            name = f"packed-512/{info['path']}"
            verified_hashes[name] = file_hash(root / name)
            if verified_hashes[name] != info["sha256"]:
                raise ValueError("Finalized corpus checksum mismatch.")
            names.append(name)
        names += [f"packed-512/{language}.manifest.json",
                  f"packed-512/{language}-documents.sqlite"]
    paths = {name: root / name for name in names}
    paths["sources.json"] = source
    files = [
        {"path": name, "bytes": path.stat().st_size,
         "sha256": verified_hashes[name] if name in verified_hashes else file_hash(path)}
        for name, path in paths.items()
    ]
    bundle = {
        "schema": "bobcat-corpus-bundle-v1", "source_manifest_sha256": source_hash,
        "tokenizer_sha256": tokenizer.digest, "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "raw_source_parquet_included": False,
    }
    bundle["content_sha256"] = digest(bundle)
    return bundle, paths


def validate(bundle: dict, max_bytes: int) -> None:
    if (bundle.get("schema") != "bobcat-corpus-bundle-v1"
            or bundle.get("content_sha256") != digest(
                {k: v for k, v in bundle.items() if k != "content_sha256"}
            )):
        raise ValueError("Invalid frozen corpus bundle.")
    names = set()
    total = 0
    for item in bundle["files"]:
        name = item["path"]
        if (not isinstance(name, str) or Path(name).is_absolute()
                or "\\" in name or any(p in {"", ".", ".."} for p in name.split("/"))
                or name in names or name.endswith(".partial")):
            raise ValueError("Unsafe or incomplete corpus filename.")
        if (type(item["bytes"]) is not int or item["bytes"] < 1
                or not re.fullmatch("[0-9a-f]{64}", item["sha256"])):
            raise ValueError("Invalid corpus file identity.")
        names.add(name)
        total += item["bytes"]
    required = {"tokenizer.json", "tokenizer.manifest.json", "sources.json"}
    required |= {f"packed-512/{lang}.manifest.json" for lang in ("ko", "en")}
    required |= {f"packed-512/{lang}-{split}.bin"
                 for lang in ("ko", "en") for split in ("train", "validation")}
    if (not required <= names or total != bundle["total_bytes"]
            or not 0 < total <= max_bytes):
        raise ValueError("Corpus bundle is incomplete or exceeds the byte limit.")


def publish(root: Path, source: Path, store: S3CheckpointStore) -> dict:
    bundle, paths = collect(root, source)
    validate(bundle, 64 * 1024**3)
    receipts = []
    for item in bundle["files"]:
        key = f"{store.prefix}/blobs/{item['sha256']}"
        store.client.upload_file(str(paths[item["path"]]), store.bucket, key, ExtraArgs={
            "ChecksumAlgorithm": "SHA256", "Metadata": {"sha256": item["sha256"]},
        })
        head = store.client.head_object(Bucket=store.bucket, Key=key, ChecksumMode="ENABLED")
        if (head["ContentLength"] != item["bytes"]
                or head.get("Metadata", {}).get("sha256") != item["sha256"]):
            raise ValueError("Uploaded corpus identity mismatch.")
        receipts.append({
            "path": item["path"], "key": key, "version_id": head.get("VersionId"),
        })
        print(json.dumps({"event": "corpus_file_uploaded", **item}), flush=True)
    key = f"{store.prefix}/bundles/{bundle['content_sha256']}.json"
    store.client.put_object(
        Bucket=store.bucket, Key=key,
        Body=json.dumps(bundle, ensure_ascii=False, allow_nan=False).encode(),
        ContentType="application/json", ChecksumAlgorithm="SHA256",
    )
    receipt = {
        "schema": "bobcat-corpus-bundle-published-v1", "bundle": bundle, "manifest_key": key,
        "files": receipts,
        "verification": "SDK transfer checksums and S3 identity; restore verifies full SHA256",
    }
    atomic_json(root / "bundle-published.json", receipt)
    return receipt


def restore(store: S3CheckpointStore, bundle_hash: str, out: Path,
            max_bytes: int = 64 * 1024**3) -> dict:
    if not re.fullmatch("[0-9a-f]{64}", bundle_hash):
        raise ValueError("Use a frozen bundle SHA256, not a mutable latest alias.")
    body = store.client.get_object(
        Bucket=store.bucket, Key=f"{store.prefix}/bundles/{bundle_hash}.json",
    )["Body"].read(1024 * 1024 + 1)
    if len(body) > 1024 * 1024:
        raise ValueError("Oversized bundle manifest.")
    bundle = json.loads(body)
    validate(bundle, max_bytes)
    if bundle["content_sha256"] != bundle_hash:
        raise ValueError("Wrong bundle for this immutable key.")
    if out.exists():
        raise ValueError("Restore to a new directory; preserve existing local data.")
    out.mkdir(parents=True)
    for item in bundle["files"]:
        path = out / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        try:
            store.client.download_file(
                store.bucket, f"{store.prefix}/blobs/{item['sha256']}", str(temporary),
            )
            if temporary.stat().st_size != item["bytes"] or file_hash(temporary) != item["sha256"]:
                raise ValueError("Restored corpus checksum mismatch.")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    # Consumers see a completion receipt only after every file has been verified.
    collected, _ = collect(out, out / "sources.json")
    if collected["content_sha256"] != bundle_hash:
        raise ValueError("Restored corpus manifests do not agree.")
    atomic_json(out / "bundle-restored.json", bundle)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("publish", "restore"):
        command = sub.add_parser(name)
        command.add_argument("--s3", required=True)
        command.add_argument("--region")
        if name == "publish":
            command.add_argument("--root", type=Path, required=True)
            command.add_argument("--source", type=Path, required=True)
        else:
            command.add_argument("--bundle-sha256", required=True)
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    store = S3CheckpointStore(args.s3, region=args.region)
    if args.command == "publish":
        receipt = publish(args.root, args.source, store)
        print(json.dumps({
            "manifest_key": receipt["manifest_key"],
            "bytes": receipt["bundle"]["total_bytes"],
        }))
    else:
        bundle = restore(store, args.bundle_sha256, args.out)
        print(json.dumps({"bundle_sha256": bundle["content_sha256"],
                          "bytes": bundle["total_bytes"]}))


if __name__ == "__main__":
    main()
