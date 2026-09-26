"""Upload a staged Bobcat package to Hugging Face as a PUBLIC model repository.

The package is a folder written by `scripts/stage_hf_release.py`. This script is the upload
step that the stager deliberately leaves out. Run it on the dev node, where the staged
folders live, never on a laptop:

    python scripts/hf_publish.py --package /path/to/stage/bobcat-1.1 --dry-run
    HF_TOKEN=... python scripts/hf_publish.py --package /path/to/stage/bobcat-1.1
    python scripts/hf_publish.py --package /path/to/stage/bobcat-1.1-nvfp4 \
        --repo sanghwa-na/bobcat-1.1-nvfp4 --dry-run

Before anything is sent:
  * the target repository is `sanghwa-na/<name>` for the manifest's `name` (adapters), or
    `--repo` for a ready-to-serve variant of that release; only the repositories approved for
    public upload are accepted (bobcat-1 stays private), each with the package kind it takes
    (an adapter, or a checkpoint with its `compiler/` folder);
  * every file of the package is listed in its SHA256SUMS.json and matches it, and nothing
    else is in the folder;
  * the card and the manifest pass the stager's blocked-pattern scan.
Upload order: create the repository private (or keep an existing one private), upload the
folder in one commit, list the remote files and check each against SHA256SUMS.json (Git LFS
files by their SHA-256, the others by their Git blob id), and only then make it public and
confirm. A failed check leaves the repository private.

The token is read only from the HF_TOKEN environment variable and passed to the client in
memory: it is never printed, logged, written to disk or put on a command line, and errors
are reported by type only. `--dry-run` needs no token and changes nothing; with HF_TOKEN
set it also reports, read-only, whether the repository exists and whether it is private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stage_hf_release import blocked, front_matter  # noqa: E402

OWNER = "sanghwa-na"
# Owner decisions 2026-09-26: adapters and ready-to-serve weights; bobcat-1 stays private.
APPROVED = {  # repository -> (release manifest name, package kind)
    f"{OWNER}/bobcat-1.1": ("bobcat-1.1", "adapter"),
    f"{OWNER}/bobcat-flash-1.1": ("bobcat-flash-1.1", "adapter"),
    f"{OWNER}/bobcat-1.1-nvfp4": ("bobcat-1.1", "weights"),
    f"{OWNER}/bobcat-flash-1.1-merged": ("bobcat-flash-1.1", "weights"),
}
SUMS = "SHA256SUMS.json"
REMOTE_EXTRA = {".gitattributes"}  # created by the Hub with every repository


class PublishError(Exception):
    """A refusal or a failed check; the message never contains the token."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(path: Path) -> str:
    """The Git object id of a file (SHA-1 over "blob <size>\\0" + content)."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_kind(package: Path) -> str:
    if (package / "adapter_model.safetensors").is_file():
        return "adapter"
    if (package / "config.json").is_file() and (package / "compiler").is_dir():
        return "weights"
    return "unknown"


def check_package(package: Path, repo: str | None = None) -> dict:
    """The package's name, repository and files, after every local check."""
    if not (package / SUMS).is_file():
        raise PublishError(f"{package} has no {SUMS}; stage it with stage_hf_release.py")
    sums = json.loads((package / SUMS).read_text())
    present = {str(p.relative_to(package)) for p in package.rglob("*") if p.is_file()}
    listed = set(sums) | {SUMS}
    if present != listed:
        raise PublishError(f"files not in {SUMS}: {sorted(present - listed)}; "
                           f"missing: {sorted(listed - present)}")
    bad = [name for name, digest in sums.items() if sha256(package / name) != digest]
    if bad:
        raise PublishError(f"files differ from {SUMS}: {bad}")
    manifest = json.loads((package / "bobcat-release-manifest.json").read_text())
    name = manifest.get("name")
    repo = repo or f"{OWNER}/{name}"
    if repo not in APPROVED:
        raise PublishError(f"{repo!r} is not approved for public upload")
    release, kind = APPROVED[repo]
    if name != release or package_kind(package) != kind:
        raise PublishError(f"{repo} takes a {kind} package of {release}; this is a "
                           f"{package_kind(package)} package of {name}")
    card = (package / "README.md").read_text()
    meta = front_matter(card)
    if repo not in card or not meta.get("base_model") or (
            kind == "adapter" and meta.get("base_model") != manifest["base"]["repo"]):
        raise PublishError("the card does not match the package's manifest")
    manifest_text = (package / "bobcat-release-manifest.json").read_text()
    for label, text, is_manifest in (("card", card, False), ("manifest", manifest_text, True)):
        hits = blocked(text, manifest=is_manifest)
        if hits:
            raise PublishError(f"the {label} matches blocked patterns: {', '.join(hits)}")
    files = {name: {"sha256": digest, "blob_id": git_blob_id(package / name),
                    "size": (package / name).stat().st_size}
             for name, digest in sums.items()}
    files[SUMS] = {"sha256": sha256(package / SUMS), "blob_id": git_blob_id(package / SUMS),
                   "size": (package / SUMS).stat().st_size}
    return {"name": name, "repo": repo, "kind": kind, "files": files}


def field(item, name, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def remote_files(api, repo: str) -> dict:
    """path -> {size, blob_id, lfs_sha256} of every file in the repository's main branch."""
    out = {}
    for item in api.list_repo_tree(repo, repo_type="model", recursive=True):
        if field(item, "type", "file") == "directory" or field(item, "blob_id") is None:
            continue
        lfs = field(item, "lfs")
        out[field(item, "path")] = {
            "size": field(item, "size"), "blob_id": field(item, "blob_id"),
            "lfs_sha256": field(lfs, "sha256") if lfs else None}
    return out


def verify_remote(expected: dict, remote: dict) -> list[str]:
    """Differences between the uploaded files and the package (empty when they match)."""
    problems = []
    unexpected = set(remote) - set(expected) - REMOTE_EXTRA
    if unexpected:
        problems.append(f"unexpected remote files: {sorted(unexpected)}")
    for name, local in expected.items():
        have = remote.get(name)
        if have is None:
            problems.append(f"missing remotely: {name}")
        elif have["lfs_sha256"] is not None:
            if have["lfs_sha256"] != local["sha256"]:
                problems.append(f"LFS sha256 differs: {name}")
        elif have["blob_id"] != local["blob_id"]:
            problems.append(f"Git blob differs: {name}")
    return problems


def repo_state(api, repo: str) -> dict:
    try:
        info = api.repo_info(repo, repo_type="model")
    except Exception as error:  # noqa: BLE001 - reported by type only
        if type(error).__name__ in ("RepositoryNotFoundError", "GatedRepoError"):
            return {"exists": False}
        raise PublishError(f"could not read {repo} ({type(error).__name__})") from None
    return {"exists": True, "private": bool(field(info, "private"))}


def set_private(api, repo: str, private: bool) -> None:
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(repo, private=private, repo_type="model")
    else:
        api.update_repo_visibility(repo, private=private, repo_type="model")


def publish(package: Path, api, *, dry_run: bool, repo: str | None = None) -> dict:
    plan = check_package(package, repo)
    repo = plan["repo"]
    report = {"repo": repo, "kind": plan["kind"], "package": str(package),
              "files": len(plan["files"]),
              "bytes": sum(f["size"] for f in plan["files"].values()),
              "target_visibility": "public", "dry_run": dry_run}
    if plan["kind"] == "adapter":
        report["adapter_model_sha256"] = plan["files"]["adapter_model.safetensors"]["sha256"]
    else:
        report["weights_sha256"] = {n: f["sha256"] for n, f in plan["files"].items()
                                    if n.endswith(".safetensors")}
    if dry_run:
        report["remote"] = repo_state(api, repo) if api is not None else "not checked (no token)"
        report["uploaded"] = False
        return report
    state = repo_state(api, repo)
    if not state["exists"]:
        api.create_repo(repo, repo_type="model", private=True, exist_ok=False)
    elif not state["private"]:
        raise PublishError(f"{repo} is already public; review it before uploading over it")
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(package),
                      commit_message=f"{repo.split('/')[1]}: {plan['kind']}, card, release "
                                     "manifest, identifiers and SHA256SUMS")
    problems = verify_remote(plan["files"], remote_files(api, repo))
    if problems:
        raise PublishError(f"{repo} stays private; remote check failed: {problems}")
    set_private(api, repo, False)
    if repo_state(api, repo) != {"exists": True, "private": False}:
        raise PublishError(f"{repo} did not become public")
    report.update(uploaded=True, remote={"exists": True, "private": False}, verified=True)
    return report


def client(token: str | None):
    if token is None:
        return None
    from huggingface_hub import HfApi

    return HfApi(token=token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True,
                        help="a folder written by scripts/stage_hf_release.py")
    parser.add_argument("--repo", help="target repository (default: sanghwa-na/<release name>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the package and print the plan; upload nothing")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    token = os.environ.get("HF_TOKEN") or None
    if token is None and not args.dry_run:
        print("Set HF_TOKEN in the environment to upload (it is read from nowhere else).")
        return 2
    try:
        report = publish(args.package, client(token), dry_run=args.dry_run, repo=args.repo)
    except PublishError as error:
        print(f"Refusing: {error}")
        return 1
    except Exception as error:  # noqa: BLE001 - never echo messages that may carry headers
        print(f"Upload not confirmed ({type(error).__name__}); the repository was left "
              "private unless the output above says otherwise.")
        return 1
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
