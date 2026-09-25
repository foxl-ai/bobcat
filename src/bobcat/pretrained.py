"""Acquire a pinned pretrained checkpoint without executing repository code."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != "bobcat-pretrained-source-v1":
        raise ValueError("Unknown pretrained source manifest.")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Pin an immutable model revision.")
    if not manifest.get("license") or not manifest.get("files"):
        raise ValueError("Record the model license and all required files.")
    names = set()
    for item in manifest["files"]:
        name = item["path"]
        if (Path(name).name != name or name.startswith(".") or name in names
                or name.endswith((".py", ".sh", ".pkl", ".pt", ".bin"))):
            raise ValueError("Use unique flat data files; no downloaded executable code.")
        names.add(name)
        if (type(item["bytes"]) is not int or item["bytes"] < 1
                or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))):
            raise ValueError("Every file needs an exact size and SHA256.")
    if sum(item["bytes"] for item in manifest["files"]) != manifest["total_bytes"]:
        raise ValueError("Model byte budget does not match the file list.")
    if not {"config.json", "tokenizer.json", "LICENSE"}.issubset(names):
        raise ValueError("The original configuration, tokenizer and license are required.")


def acquire(source: Path, out: Path, workers: int = 4) -> dict:
    from huggingface_hub import hf_hub_download

    manifest = json.loads(source.read_text())
    validate_manifest(manifest)
    if not 1 <= workers <= 8:
        raise ValueError("Use between one and eight download workers.")
    out.mkdir(parents=True, exist_ok=True)
    identity = {"source_manifest_sha256": file_hash(source),
                "repo": manifest["repo"], "revision": manifest["revision"]}
    stamp = out / "bobcat-source.json"
    if stamp.exists() and json.loads(stamp.read_text()) != identity:
        raise ValueError("Output directory belongs to a different model manifest.")
    already_present = sum(
        min((out / item["path"]).stat().st_size, item["bytes"])
        for item in manifest["files"] if (out / item["path"]).is_file()
    )
    required = manifest["total_bytes"] - already_present + 64 * 1024**3
    if shutil.disk_usage(out).free < required:
        raise ValueError("Insufficient space for checkpoint files and 64 GiB working reserve.")
    atomic_json(stamp, identity)
    started = time.monotonic()

    def download(item):
        before = time.monotonic()
        path = Path(hf_hub_download(
            repo_id=manifest["repo"], revision=manifest["revision"],
            filename=item["path"], local_dir=out,
        ))
        if path.parent.resolve() != out.resolve():
            raise ValueError("Downloaded file escaped the model directory.")
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError(f"Model content failed verification: {item['path']}")
        receipt = {**item, "verified_seconds": round(time.monotonic() - before, 3)}
        atomic_json(out / "bobcat-receipts" / f"{item['path']}.json", receipt)
        print(json.dumps({"event": "model_file_verified", **receipt}), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(download, manifest["files"]))
    result = {
        "schema": "bobcat-pretrained-download-v1", **identity,
        "completed_at": datetime.now(UTC).isoformat(),
        "license": manifest["license"], "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "seconds": round(time.monotonic() - started, 3),
        "repository_code_executed": False,
        "inference_verified": False, "training_completed": False,
    }
    atomic_json(out / "bobcat-downloads.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = acquire(args.source, args.out, args.workers)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
