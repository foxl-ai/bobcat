import hashlib
import json
from pathlib import Path

import pytest

from scripts import hf_publish as hp

TOKEN = "hf_" + "Q" * 34
CARD = """---
license: apache-2.0
base_model: org/base
library_name: peft
pipeline_tag: zero-shot-classification
---

# Bobcat 1.1

Adapter at sanghwa-na/{name}; serve it on 127.0.0.1.
"""


def make_package(tmp_path: Path, name: str = "bobcat-1.1") -> Path:
    package = tmp_path / name
    (package / "assets").mkdir(parents=True)
    (package / "README.md").write_text(CARD.format(name=name))
    (package / "bobcat-release-manifest.json").write_text(json.dumps(
        {"name": name, "base": {"repo": "org/base", "revision": "a" * 40}}))
    (package / "adapter_model.safetensors").write_bytes(b"\x00weights" * 100)
    (package / "assets/fig.png").write_bytes(b"png")
    sums = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*")) if p.is_file()}
    (package / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1) + "\n")
    return package


class FakeHub:
    """In-memory Hub: LFS for binary files, Git blob ids for the rest."""

    def __init__(self, existing=None, corrupt=None):
        self.repos = dict(existing or {})
        self.files: dict[str, dict] = {}
        self.calls = []
        self.corrupt = corrupt

    def repo_info(self, repo, repo_type):
        self.calls.append(("repo_info", repo))
        if repo not in self.repos:
            raise type("RepositoryNotFoundError", (Exception,), {})()
        return {"private": self.repos[repo]}

    def create_repo(self, repo, repo_type, private, exist_ok):
        self.calls.append(("create_repo", repo, private))
        self.repos[repo] = private
        self.files[".gitattributes"] = {"path": ".gitattributes", "size": 3, "blob_id": "g" * 40,
                                        "lfs": None}

    def upload_folder(self, repo_id, repo_type, folder_path, commit_message):
        self.calls.append(("upload_folder", repo_id, self.repos[repo_id]))
        for path in sorted(Path(folder_path).rglob("*")):
            if not path.is_file():
                continue
            name = str(path.relative_to(folder_path))
            lfs = None
            if path.suffix in (".safetensors", ".png"):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lfs = {"sha256": "0" * 64 if name == self.corrupt else digest}
            self.files[name] = {"path": name, "size": path.stat().st_size,
                                "blob_id": hp.git_blob_id(path), "lfs": lfs}

    def list_repo_tree(self, repo, repo_type, recursive):
        return list(self.files.values()) + [{"path": "assets", "type": "directory"}]

    def update_repo_settings(self, repo, private, repo_type):
        self.calls.append(("update_repo_settings", repo, private))
        self.repos[repo] = private


def test_dry_run_checks_and_uploads_nothing(tmp_path):
    report = hp.publish(make_package(tmp_path), None, dry_run=True)
    assert report["repo"] == "sanghwa-na/bobcat-1.1" and report["uploaded"] is False
    assert report["files"] == 5 and report["target_visibility"] == "public"


def test_dry_run_with_a_token_is_read_only_and_never_prints_it(tmp_path, monkeypatch, capsys):
    hub = FakeHub()
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(hp, "client", lambda token: hub)
    assert hp.main(["--package", str(make_package(tmp_path)), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert TOKEN not in out and json.loads(out)["remote"] == {"exists": False}
    assert [c[0] for c in hub.calls] == ["repo_info"]


def test_upload_goes_private_then_verifies_then_public(tmp_path):
    hub = FakeHub()
    report = hp.publish(make_package(tmp_path), hub, dry_run=False)
    assert report["uploaded"] and report["verified"]
    names = [c[0] for c in hub.calls]
    assert names.index("create_repo") < names.index("upload_folder") < \
        names.index("update_repo_settings")
    assert ("create_repo", "sanghwa-na/bobcat-1.1", True) in hub.calls
    assert ("upload_folder", "sanghwa-na/bobcat-1.1", True) in hub.calls
    assert hub.repos["sanghwa-na/bobcat-1.1"] is False


def test_a_remote_hash_mismatch_keeps_the_repository_private(tmp_path):
    hub = FakeHub(corrupt="adapter_model.safetensors")
    with pytest.raises(hp.PublishError, match="stays private"):
        hp.publish(make_package(tmp_path), hub, dry_run=False)
    assert hub.repos["sanghwa-na/bobcat-1.1"] is True
    assert not any(c[0] == "update_repo_settings" for c in hub.calls)


def test_stale_remote_files_keep_the_repository_private(tmp_path):
    hub = FakeHub(existing={"sanghwa-na/bobcat-1.1": True})
    hub.files["old.bin"] = {"path": "old.bin", "size": 1, "blob_id": "b" * 40, "lfs": None}
    with pytest.raises(hp.PublishError, match="unexpected remote files"):
        hp.publish(make_package(tmp_path), hub, dry_run=False)
    assert hub.repos["sanghwa-na/bobcat-1.1"] is True


def test_an_already_public_repository_is_not_overwritten(tmp_path):
    hub = FakeHub(existing={"sanghwa-na/bobcat-1.1": False})
    with pytest.raises(hp.PublishError, match="already public"):
        hp.publish(make_package(tmp_path), hub, dry_run=False)
    assert not any(c[0] == "upload_folder" for c in hub.calls)


def test_bobcat_1_stays_private(tmp_path):
    with pytest.raises(hp.PublishError, match="not approved"):
        hp.publish(make_package(tmp_path, "bobcat-1"), FakeHub(), dry_run=True)


def test_unlisted_or_changed_files_are_refused(tmp_path):
    package = make_package(tmp_path)
    (package / "notes.txt").write_text("x")
    with pytest.raises(hp.PublishError, match="not in SHA256SUMS"):
        hp.check_package(package)
    (package / "notes.txt").unlink()
    (package / "assets/fig.png").write_bytes(b"other")
    with pytest.raises(hp.PublishError, match="differ"):
        hp.check_package(package)


def test_a_card_with_a_price_is_refused(tmp_path):
    package = make_package(tmp_path)
    card = package / "README.md"
    card.write_text(card.read_text() + "\nAbout $0.01 per million.\n")
    sums = json.loads((package / "SHA256SUMS.json").read_text())
    sums["README.md"] = hashlib.sha256(card.read_bytes()).hexdigest()
    (package / "SHA256SUMS.json").write_text(json.dumps(sums))
    with pytest.raises(hp.PublishError, match="price"):
        hp.check_package(package)


def test_upload_needs_the_token_in_the_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hp.main(["--package", str(make_package(tmp_path))]) == 2
    assert "HF_TOKEN" in capsys.readouterr().out


def test_errors_are_reported_by_type_without_the_token(tmp_path, monkeypatch, capsys):
    class Leaky(FakeHub):
        def upload_folder(self, *args, **kwargs):
            raise RuntimeError(f"401 for Authorization: Bearer {TOKEN}")

    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(hp, "client", lambda token: Leaky())
    assert hp.main(["--package", str(make_package(tmp_path))]) == 1
    out = capsys.readouterr().out
    assert TOKEN not in out and "RuntimeError" in out


def make_weights_package(tmp_path: Path, name: str = "bobcat-1.1",
                         repo: str = "sanghwa-na/bobcat-1.1-nvfp4") -> Path:
    package = tmp_path / repo.split("/")[1]
    (package / "compiler").mkdir(parents=True)
    card = CARD.format(name=repo.split("/")[1])
    (package / "README.md").write_text(card.replace(
        "base_model: org/base",
        "base_model: sanghwa-na/bobcat-1.1\nbase_model_relation: quantized"))
    (package / "bobcat-release-manifest.json").write_text(json.dumps(
        {"name": name, "base": {"repo": "org/base", "revision": "a" * 40}}))
    (package / "config.json").write_text("{}")
    (package / "model.safetensors").write_bytes(b"\x01w" * 50)
    (package / "compiler/tokenizer.json").write_text("{}")
    sums = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*")) if p.is_file()}
    (package / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1) + "\n")
    return package


def test_a_weights_package_goes_to_its_variant_repository(tmp_path):
    hub = FakeHub()
    package = make_weights_package(tmp_path)
    report = hp.publish(package, hub, dry_run=False, repo="sanghwa-na/bobcat-1.1-nvfp4")
    assert report["kind"] == "weights" and report["verified"]
    assert set(report["weights_sha256"]) == {"model.safetensors"}
    assert hub.repos["sanghwa-na/bobcat-1.1-nvfp4"] is False


@pytest.mark.parametrize("repo", ["sanghwa-na/bobcat-1.1", "sanghwa-na/bobcat-flash-1.1-merged",
                                  "sanghwa-na/bobcat-1-nvfp4"])
def test_a_weights_package_is_refused_elsewhere(tmp_path, repo):
    package = make_weights_package(tmp_path)
    with pytest.raises(hp.PublishError, match="takes a|not approved"):
        hp.check_package(package, repo)


def test_an_adapter_package_is_refused_by_a_weights_repository(tmp_path):
    with pytest.raises(hp.PublishError, match="takes a weights package"):
        hp.check_package(make_package(tmp_path), "sanghwa-na/bobcat-1.1-nvfp4")
