import hashlib
import json
from pathlib import Path

import pytest

from scripts import stage_hf_release as shr

INTERNAL = {
    "name": "bobcat-x",
    "base": {"repo": "org/base", "revision": "a" * 40, "license": "Apache-2.0"},
    "adapter": {"adapter_model_sha256": None},
    "weights": {"s3": "runs/x/model/", "sha256": {"adapter/adapter_model.safetensors": None}},
    "source": {"git_head": "b" * 40,
               "files_sha256": {"src/bobcat/a.py": "c" * 64, "infra/cloud/job.sh": "d" * 64}},
    "serving": {"note": "measured on one RTX PRO 6000 (g7e.2xlarge); dev rows",
                "tool": "infra/cloud/nvfp4_quantize.py, llmcompressor",
                "usd_per_mtok_at_9_99_per_hour": 0.5,
                "cost_estimate": {"usd": 1.0}, "gpu_memory_nvidia_smi": ["1 MiB"],
                "w1_p50_ms": 24.8},
    "selection": {"preregistration": {"file": ".aws-local/prereg-x.json (internal)",
                                      "sha256": "e" * 64}},
}
CARD = """---
license: apache-2.0
base_model: org/base
base_model_revision: {revision}
library_name: peft
pipeline_tag: zero-shot-classification
---

# Bobcat X

Adapter at sanghwa-na/bobcat-x. Serve it with `$PY -m bobcat.api_server`.
"""


def write_release(tmp_path: Path, adapter_bytes: bytes = b"weights") -> shr.Release:
    manifest = json.loads(json.dumps(INTERNAL))
    digest = hashlib.sha256(adapter_bytes).hexdigest()
    manifest["adapter"]["adapter_model_sha256"] = digest
    manifest["weights"]["sha256"]["adapter/adapter_model.safetensors"] = digest
    internal = tmp_path / "release/bobcat-x-manifest.json"
    internal.parent.mkdir(parents=True)
    internal.write_text(json.dumps(manifest))
    folder = tmp_path / "release/hf/bobcat-x"
    (folder / "assets").mkdir(parents=True)
    (folder / "assets/fig.png").write_bytes(b"png")
    (folder / "README.md").write_text(CARD.format(revision="a" * 40))
    release = shr.Release("bobcat-x", "sanghwa-na/bobcat-x", folder / "README.md",
                          folder / "assets", internal, ("adapter", "adapter_model_sha256"),
                          folder / "bobcat-release-manifest.json")
    release.public_manifest.write_text(shr.expected_public(release))
    return release


def write_adapter(folder: Path, data: bytes = b"weights") -> Path:
    folder.mkdir(parents=True)
    (folder / "adapter_model.safetensors").write_bytes(data)
    (folder / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "/mnt/nvme/base", "revision": None, "r": 16}))
    return folder


def test_public_manifest_drops_storage_cost_and_orchestration():
    out = shr.public_manifest(INTERNAL, "e" * 64)
    text = shr.render(out)
    assert "s3" not in out["weights"] and "git_head" not in out["source"]
    assert list(out["source"]["files_sha256"]) == ["src/bobcat/a.py"]
    serving = out["serving"]
    assert set(serving) == {"note", "tool", "w1_p50_ms"}
    assert serving["note"] == "measured on one RTX PRO 6000; dev rows"
    assert serving["tool"].startswith("scripts/nvfp4_quantize.py")
    assert out["selection"]["preregistration"]["file"] == "an internal file"
    assert out["public_copy"]["derived_from_sha256"] == "e" * 64
    assert shr.blocked(text, manifest=True) == []
    assert "instance-type" in shr.blocked(shr.render(INTERNAL), manifest=True)


@pytest.mark.parametrize("text, name", [
    ("copy s3://bucket/runs/x", "storage-uri"),
    ("hosted in eu-west-9", "cloud-region"),
    ("instance i-0" + "123456789abcdef0", "instance-id"),  # split: the publish scan
    ("at $9.99 per GPU", "price"),
    ("$1.23/MTok", "price"),
    ("a Spot host", "market"),
    ("/home/ubuntu/bobcat", "local-path"),
    ("origin at 10.2.3.4", "ip-address"),
    ("token hf_" + "a" * 34, "credential"),
])
def test_blocked_patterns(text, name):
    assert name in shr.blocked(text, manifest=False)


def test_instance_types_only_blocked_in_manifests():
    text = ("for example `g7e.2xlarge` on Amazon EC2; run `$PY -m bobcat.api_server "
            "--host 127.0.0.1` or bind 0.0.0.0")
    assert shr.blocked(text, manifest=False) == []
    assert shr.blocked(text, manifest=True) == ["instance-type"]


def test_stage_writes_the_package(tmp_path):
    release = write_release(tmp_path)
    adapter = write_adapter(tmp_path / "adapter")
    result = shr.stage(release, adapter, tmp_path / "out")
    out = tmp_path / "out"
    config = json.loads((out / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "org/base"
    assert config["revision"] == "a" * 40 and config["r"] == 16
    assert (out / "README.md").read_text() == release.card.read_text()
    assert (out / "bobcat-release-manifest.json").read_text() == \
        release.public_manifest.read_text()
    assert (out / "assets/fig.png").read_bytes() == b"png"
    assert json.loads((out / "bobcat-identifiers.json").read_text())["glm_preferred_identifiers"]
    sums = json.loads((out / "SHA256SUMS.json").read_text())
    assert set(sums) == {"adapter_model.safetensors", "adapter_config.json", "README.md",
                         "bobcat-release-manifest.json", "assets/fig.png",
                         "bobcat-identifiers.json"}
    assert sums["adapter_model.safetensors"] == hashlib.sha256(b"weights").hexdigest()
    assert result["uploaded"] is False and result["files"] == 7


def test_stage_refuses_a_different_adapter(tmp_path):
    release = write_release(tmp_path)
    adapter = write_adapter(tmp_path / "adapter", b"other")
    with pytest.raises(SystemExit, match="does not match"):
        shr.stage(release, adapter, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_stage_refuses_a_drifted_public_manifest(tmp_path):
    release = write_release(tmp_path)
    data = json.loads(release.internal_manifest.read_text())
    data["base"]["revision"] = "f" * 40
    release.internal_manifest.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="differs from the internal"):
        shr.stage(release, write_adapter(tmp_path / "adapter"), tmp_path / "out")


def test_stage_refuses_a_card_with_a_price_or_wrong_base(tmp_path):
    release = write_release(tmp_path)
    adapter = write_adapter(tmp_path / "adapter")
    release.card.write_text(CARD.format(revision="a" * 40) + "\nAbout $0.01 per million.\n")
    with pytest.raises(SystemExit, match="price"):
        shr.stage(release, adapter, tmp_path / "out")
    release.card.write_text(CARD.format(revision="f" * 40))
    with pytest.raises(SystemExit, match="base_model_revision"):
        shr.stage(release, adapter, tmp_path / "out")


def test_main_refuses_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    with pytest.raises(SystemExit, match="credentials"):
        shr.main(["--release", "bobcat-1.1", "--adapter-dir", str(tmp_path),
                  "--out", str(tmp_path / "out")])


def test_download_s3_fetches_the_two_adapter_files(tmp_path):
    calls = []

    class Client:
        def download_file(self, bucket, key, path):
            calls.append((bucket, key))
            Path(path).write_text(key)

    folder = shr.download_s3("s3://bucket/runs/x/adapter", tmp_path / "dl", client=Client())
    assert calls == [("bucket", "runs/x/adapter/adapter_model.safetensors"),
                     ("bucket", "runs/x/adapter/adapter_config.json")]
    assert (folder / "adapter_config.json").read_text().endswith("adapter_config.json")
    with pytest.raises(SystemExit):
        shr.download_s3("bucket/runs", tmp_path / "dl2", client=Client())


@pytest.mark.parametrize("name", sorted(shr.RELEASES))
def test_committed_cards_and_manifests_are_publishable(name):
    release = shr.RELEASES[name]
    manifest, text = shr.published_manifest(release)
    card = release.card.read_text()
    shr.check_card(release, card, manifest)
    assert shr.blocked(card, manifest=False) == []
    assert shr.blocked(text, manifest=True) == []
    if release.kind == "adapter":
        assert len(shr.dig(manifest, release.adapter_hash_path)) == 64
    else:
        hashes = shr.dig(manifest, release.weight_hashes_path)
        assert any(name.endswith(".safetensors") for name in hashes)
        assert all(len(digest) == 64 for digest in hashes.values())
        assert release.card.with_name("NOTICE").is_file()


def write_weights(tmp_path: Path, release_manifest: dict) -> tuple[Path, Path]:
    weights, base = tmp_path / "weights", tmp_path / "base"
    for folder in (weights, base):
        folder.mkdir()
        for name in shr.TOKENIZER_FILES:
            (folder / name).write_text(f"{name}\n")
    (base / "config.json").write_text("{}\n")
    (weights / "config.json").write_text('{"quantization_config": {}}\n')
    (weights / "model.safetensors").write_bytes(b"w" * 64)
    (weights / "README.md").write_text("upstream card")
    (weights / "bobcat-merge.json").write_text(json.dumps({"base": "/mnt/nvme/base", "rank": 64}))
    (base / "LICENSE").write_text("Apache License\n")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    (base / "bobcat-download.json").write_text(json.dumps({
        "repo": "org/base", "revision": "a" * 40,
        "files": {n: {"status": "ok", "sha256": digest(base / n)}
                  for n in (*shr.TOKENIZER_FILES, "config.json", "LICENSE")}}))
    release_manifest["tokenizer"] = {n: digest(weights / n) for n in shr.TOKENIZER_FILES}
    release_manifest["serving_builds"] = {"nvfp4": {"files_sha256": {
        "model.safetensors": digest(weights / "model.safetensors")}}}
    return weights, base


def weights_release(tmp_path: Path, internal: dict) -> shr.Release:
    path = tmp_path / "release/bobcat-x-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(internal))
    folder = tmp_path / "release/hf/bobcat-x-nvfp4"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(CARD.format(revision="a" * 40).replace(
        "base_model: org/base\nbase_model_revision: " + "a" * 40,
        "base_model: sanghwa-na/bobcat-x\nbase_model_relation: quantized").replace(
        "sanghwa-na/bobcat-x.", "sanghwa-na/bobcat-x-nvfp4."))
    (folder / "NOTICE").write_text("notice\n")
    public = tmp_path / "release/hf/bobcat-x/bobcat-release-manifest.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    release = shr.Release("bobcat-x-nvfp4", "sanghwa-na/bobcat-x-nvfp4", folder / "README.md",
                          folder / "assets", path, None, public, kind="weights",
                          weight_hashes_path=("serving_builds", "nvfp4", "files_sha256"),
                          base_model="sanghwa-na/bobcat-x", license_from="compiler",
                          source_label="the merged adapter")
    public.write_text(shr.expected_public(release))
    return release


def test_stage_weights_writes_a_servable_package(tmp_path):
    internal = json.loads(json.dumps(INTERNAL))
    weights, base = write_weights(tmp_path, internal)
    release = weights_release(tmp_path, internal)
    result = shr.stage_weights(release, weights, base, tmp_path / "out")
    out = tmp_path / "out"
    names = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert names == {"config.json", "model.safetensors", "bobcat-merge.json", *shr.TOKENIZER_FILES,
                     *(f"compiler/{n}" for n in shr.COMPILER_FILES), "LICENSE", "NOTICE",
                     "README.md", "bobcat-release-manifest.json", "bobcat-identifiers.json",
                     "SHA256SUMS.json"}
    assert json.loads((out / "bobcat-merge.json").read_text())["base"] == "the merged adapter"
    assert (out / "LICENSE").read_text() == "Apache License\n"
    assert result["uploaded"] is False and set(result["weights_sha256"]) == {"model.safetensors"}


def test_stage_weights_refuses_changed_weights_or_tokenizer(tmp_path):
    for case in ("weights", "tokenizer"):
        root = tmp_path / case
        root.mkdir()
        internal = json.loads(json.dumps(INTERNAL))
        weights, base = write_weights(root, internal)
        release = weights_release(root, internal)
        if case == "weights":
            (weights / "model.safetensors").write_bytes(b"x")
            expected = "model.safetensors does not match"
        else:
            (base / "tokenizer.json").write_text("other\n")
            expected = "tokenizer"
        with pytest.raises(SystemExit, match=expected):
            shr.stage_weights(release, weights, base, root / "out")
        assert not (root / "out").exists()
