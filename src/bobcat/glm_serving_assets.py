"""Hash serving weights on the CPU while an independent GPU job is running.

Each unique inode is read once, including hardlinks shared by original and
derived models. Before launch, file identity, size and modification time must
still match the completed verification. This never loads a CUDA model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash


def identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Serving assets must be regular files, not symlinks.")
    stat = path.stat()
    return {
        "device": stat.st_dev, "inode": stat.st_ino,
        "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }


def planned_assets(plan):
    if plan.get("schema") != "bobcat-glm-serving-model-set-v1":
        raise ValueError("Use an explicit, frozen serving model set.")
    rows, names = [], set()
    for model in plan["models"]:
        if model["name"] in names:
            raise ValueError("Duplicate serving model names.")
        names.add(model["name"])
        root = Path(model["model_dir"])
        if not root.is_absolute() or root.is_symlink():
            raise ValueError("Use explicit, non-symlink model roots.")
        files = model["files"]
        if len({row["path"] for row in files}) != len(files):
            raise ValueError("Duplicate checkpoint filenames.")
        for row in files:
            if Path(row["path"]).name != row["path"]:
                raise ValueError("Checkpoint members must stay inside their model root.")
            path = root / row["path"]
            stamp = identity(path)
            if stamp["bytes"] != row["bytes"]:
                raise ValueError("Serving asset size changed.")
            rows.append({
                "model": model["name"], "path": str(path), "relative": row["path"],
                "sha256": row["sha256"], "identity": stamp,
            })
    return rows


def verify(plan_path, out, *, seconds=1200, workers=4):
    if out.exists() or not 0 < seconds <= 1800 or not 1 <= workers <= 4:
        raise ValueError("Use a new bounded CPU verification record.")
    started = time.monotonic()
    deadline = started + seconds
    plan = json.loads(plan_path.read_text())
    assets = planned_assets(plan)
    unique = {}
    for row in assets:
        key = (row["identity"]["device"], row["identity"]["inode"])
        if key in unique:
            if unique[key]["sha256"] != row["sha256"]:
                raise ValueError("One inode cannot represent two different expected weights.")
        else:
            unique[key] = row
    record = {
        "schema": "bobcat-glm-serving-asset-verification-v1",
        "status": "verifying", "started_at": datetime.now(UTC).isoformat(),
        "model_set_sha256": file_hash(plan_path),
        "models": [model["name"] for model in plan["models"]],
        "planned_unique_files": len(unique),
        "planned_unique_bytes": sum(row["identity"]["bytes"] for row in unique.values()),
        "model_file_references": len(assets), "verified_unique_files": 0,
        "verified_unique_bytes": 0, "cpu_workers": workers, "gpu_used": False,
        "verification": "fresh full-file SHA256; one read per shared inode",
        "source_code_sha256": file_hash(Path(__file__)),
    }
    atomic_json(out, record)

    def check(row):
        path = Path(row["path"])
        if identity(path) != row["identity"]:
            raise ValueError("Serving asset changed before hashing.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                if time.monotonic() >= deadline:
                    raise TimeoutError("CPU weight-verification allowance expired.")
                digest.update(block)
        if digest.hexdigest() != row["sha256"] or identity(path) != row["identity"]:
            raise ValueError("Serving asset SHA256 or identity changed during verification.")
        return row["identity"]["bytes"]

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(check, row) for row in unique.values()]):
                record["verified_unique_bytes"] += future.result()
                record["verified_unique_files"] += 1
                record["updated_at"] = datetime.now(UTC).isoformat()
                atomic_json(out, record)
        record.update(status="completed", assets=assets)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - started)
        atomic_json(out, record)
    return record


def validate_verified_assets(plan_path, verification_path):
    record = json.loads(verification_path.read_text())
    if (record.get("schema") != "bobcat-glm-serving-asset-verification-v1"
            or record.get("status") != "completed"
            or record["model_set_sha256"] != file_hash(plan_path)
            or record["verified_unique_files"] != record["planned_unique_files"]
            or record["verified_unique_bytes"] != record["planned_unique_bytes"]):
        raise ValueError("Require completed full-byte verification for this exact model set.")
    assets = planned_assets(json.loads(plan_path.read_text()))
    if assets != record["assets"]:
        raise ValueError("A serving asset changed after the completed CPU verification.")
    return record


def filesystem_identity(path):
    """Use a persistent filesystem UUID; Linux device numbers change across hosts."""
    raw = subprocess.check_output(
        ["findmnt", "--json", "--target", str(path), "--output", "UUID,FSTYPE,OPTIONS"],
        text=True, timeout=10,
    )
    filesystems = json.loads(raw)["filesystems"]
    if (len(filesystems) != 1 or not filesystems[0].get("uuid")
            or filesystems[0].get("fstype") != "ext4"):
        raise ValueError("Use the explicitly prepared ext4 staging filesystem.")
    row = filesystems[0]
    return {"uuid": row["uuid"], "fstype": row["fstype"],
            "read_only": "ro" in row["options"].split(",")}


def portable_receipt(plan_path, verification_path, out):
    """Bind a completed CPU hash pass to its filesystem before moving the EBS volume."""
    if out.exists():
        raise ValueError("Preserve the original portable receipt.")
    verified = validate_verified_assets(plan_path, verification_path)
    plan = json.loads(plan_path.read_text())
    mounts = [filesystem_identity(model["model_dir"]) for model in plan["models"]]
    if len({mount["uuid"] for mount in mounts}) != 1:
        raise ValueError("All models must belong to the same verified staging filesystem.")
    record = {
        "schema": "bobcat-portable-asset-verification-v1", "status": "completed",
        "at": datetime.now(UTC).isoformat(), "filesystem_uuid": mounts[0]["uuid"],
        "model_set_sha256": file_hash(plan_path),
        "cpu_verification_sha256": file_hash(verification_path),
        "assets": verified["assets"], "cpu_sha256_pass_finished_at": verified["finished_at"],
        "verification": "completed full-byte CPU pass; UUID/inode/size/mtime bound for transfer",
        "gpu_used": False,
    }
    atomic_json(out, record)
    return record


def validate_portable_assets(plan_path, receipt_path):
    """Admit the same read-only filesystem, never an unverified replacement copy."""
    record = json.loads(receipt_path.read_text())
    if (record.get("schema") != "bobcat-portable-asset-verification-v1"
            or record.get("status") != "completed"
            or record["model_set_sha256"] != file_hash(plan_path)):
        raise ValueError("Use a completed portable receipt for this exact model set.")
    plan = json.loads(plan_path.read_text())
    for model in plan["models"]:
        mount = filesystem_identity(model["model_dir"])
        if mount["uuid"] != record["filesystem_uuid"] or not mount["read_only"]:
            raise ValueError("The verified staging filesystem must be unchanged and read-only.")
    actual = planned_assets(plan)

    def persistent_rows(rows):
        return [{**row, "identity": {
            key: value for key, value in row["identity"].items() if key != "device"
        }} for row in rows]

    if persistent_rows(actual) != persistent_rows(record["assets"]):
        raise ValueError("Asset inode/size/mtime or expected bytes changed after CPU verification.")
    return record


def validate_native_source(plan_path, receipt_path, model_dir, source):
    """Use the CPU proof for the exact original source, not a derived serving treatment."""
    verified = validate_portable_assets(plan_path, receipt_path)
    plan = json.loads(plan_path.read_text())
    models = [model for model in plan["models"] if model["model_dir"] == str(model_dir)]

    def content_files(files):
        # Acquisition metadata such as a Git blob ID is not a filesystem
        # identity. Both manifests must agree on every filename, byte count
        # and full-file SHA256; duplicate filenames are never admissible.
        rows = [(row["path"], row["bytes"], row["sha256"]) for row in files]
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError("Duplicate original source filenames.")
        return sorted(rows)

    if (len(models) != 1 or models[0]["role"] != "untuned_original_fp8"
            or content_files(models[0]["files"]) != content_files(source["files"])):
        raise ValueError("Preverified native loading must use the exact original source files.")
    return {
        "schema": "bobcat-native-source-preverification-v1",
        "at": datetime.now(UTC).isoformat(), "revision": source["revision"],
        "files": {row["path"]: row["sha256"] for row in source["files"]},
        "bytes": sum(row["bytes"] for row in source["files"]),
        "portable_receipt_sha256": file_hash(receipt_path),
        "cpu_sha256_pass_finished_at": verified["cpu_sha256_pass_finished_at"],
        "filesystem_uuid": verified["filesystem_uuid"],
        "full_sha256_repeated_on_gpu_host": False,
        "read_only_inode_size_mtime_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    record = verify(args.plan, args.out, seconds=args.seconds)
    print(json.dumps({key: record[key] for key in (
        "status", "verified_unique_files", "verified_unique_bytes",
        "model_file_references", "elapsed_seconds", "gpu_used",
    )}))


if __name__ == "__main__":
    main()
