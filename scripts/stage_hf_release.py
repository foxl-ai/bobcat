"""Stage the Hugging Face package of a Bobcat release (never uploads).

Assembles exactly what the Hugging Face repository holds: the LoRA adapter, the model card,
the public release manifest, the identifier list the compiler needs, the card's figures and
a SHA-256 manifest of every file. The adapter must match the release manifest's hash, and
`adapter_config.json` is rewritten to name the upstream base repository and its pinned
revision instead of the local path it was trained against. It refuses to run with Hugging
Face credentials in the environment, so staging can never turn into an upload by accident;
uploading is a separate, deliberate step.

Releases (`--release`), all ready-to-serve checkpoints:
  bobcat-1.1         release/hf/bobcat-1.1/ (BF16, the adapter merged into its base)
  bobcat-1.1-nvfp4   release/hf/bobcat-1.1-nvfp4/ (NVFP4 build of the same weights)
  bobcat-flash-1.1   release/hf/bobcat-flash-1.1/ (BF16, the adapter merged into its base)
Weight packages (`--weights-dir`, `--compiler-dir`) hold a served checkpoint: every weight
file the release manifest hashes must match, the tokenizer files must match the manifest,
receipts lose their local paths, and `compiler/` carries the base's pinned tokenizer,
template, config and download receipt that the Bobcat compiler checks. LICENSE and
NOTICE are added.
The 1.1 and Flash 1.1 card folders also hold `bobcat-release-manifest.json`, the public copy
of the internal release manifest with storage locations, cost figures, host details,
orchestration paths and comparisons with other models removed. On the development line,
`--refresh-public-manifest` rewrites it from the internal manifest, and staging refuses when
the two have drifted apart. The card and
the public manifest are scanned for storage URIs, cloud regions, instance ids, prices and
local paths before anything is written.

    python scripts/stage_hf_release.py --release bobcat-1.1 --refresh-public-manifest
    python scripts/stage_hf_release.py --release bobcat-1.1-nvfp4 --weights-dir DIR \
        --compiler-dir BASE_DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from bobcat.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = ROOT / "reports/2026-09-22-glm-readout-preflight.json"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
CREDENTIAL_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
                        "HF_API_TOKEN")


@dataclass(frozen=True)
class Release:
    name: str
    repo_id: str
    card: Path
    assets: Path
    internal_manifest: Path
    adapter_hash_path: tuple[str, ...] | None
    public_manifest: Path | None = None  # None: the internal manifest is already public
    kind: str = "adapter"  # or "weights"
    weight_hashes_path: tuple[str, ...] | None = None  # manifest dict {file: sha256}
    base_model: str | None = None  # the card's base_model when it is not the manifest base
    license_from: str = "repo"  # "compiler": the upstream LICENSE fetched with the compiler
    source_label: str = ""  # replaces local paths in the weights' receipts


RELEASES = {
    "bobcat-1.1": Release(
        "bobcat-1.1", "sanghwa-na/bobcat-1.1", ROOT / "release/hf/bobcat-1.1/README.md",
        ROOT / "release/hf/bobcat-1.1/assets", ROOT / "release/bobcat-1.1-manifest.json", None,
        ROOT / "release/hf/bobcat-1.1/bobcat-release-manifest.json", kind="weights",
        weight_hashes_path=("serving_builds", "merged_bf16", "files_sha256"),
        license_from="compiler",
        source_label="Qwen/Qwen3.8-27B at revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "bobcat-1.1-nvfp4": Release(
        "bobcat-1.1-nvfp4", "sanghwa-na/bobcat-1.1-nvfp4",
        ROOT / "release/hf/bobcat-1.1-nvfp4/README.md", ROOT / "release/hf/bobcat-1.1-nvfp4/assets",
        ROOT / "release/bobcat-1.1-manifest.json", None,
        ROOT / "release/hf/bobcat-1.1/bobcat-release-manifest.json", kind="weights",
        weight_hashes_path=("serving_builds", "nvfp4", "files_sha256"),
        base_model="sanghwa-na/bobcat-1.1", license_from="compiler",
        source_label="the Bobcat 1.1 adapter merged into Qwen/Qwen3.8-27B at revision "
                     "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 (bobcat.student_merge, BF16)"),
    "bobcat-flash-1.1": Release(
        "bobcat-flash-1.1", "sanghwa-na/bobcat-flash-1.1",
        ROOT / "release/hf/bobcat-flash-1.1/README.md",
        ROOT / "release/hf/bobcat-flash-1.1/assets",
        ROOT / "release/bobcat-flash-1.1-manifest.json", None,
        ROOT / "release/hf/bobcat-flash-1.1/bobcat-release-manifest.json", kind="weights",
        weight_hashes_path=("serving_artifacts", "merged_bf16", "files_sha256"),
        source_label="google/gemma-4-26B-A4B-it at revision "
                     "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"),
}
COMPILER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json",
                  "bobcat-download.json")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
# Never copied from a weights folder: upstream cards and receipts, and the files the stager
# itself writes (so a staged package can be restaged from its own weights).
SKIPPED_WEIGHT_FILES = {"README.md", ".gitattributes", "bobcat-download.json", "LICENSE",
                        "crc32.txt", "NOTICE", "SHA256SUMS.json", "bobcat-release-manifest.json",
                        "bobcat-identifiers.json"}
RECEIPT_PATH_FIELDS = {"bobcat-nvfp4.json": "source_model", "bobcat-merge.json": "base"}

# ---------------------------------------------------------------- public manifest

DROP_KEYS = {"s3", "cost", "cost_estimate", "git_head", "gpu_memory_nvidia_smi", "report",
             "comparison"}
# Public materials compare each model with its untrained base and Jev's published figures
# only (owner decision 2026-09-26): keys naming the earlier model's results are dropped, and a
# key holding a teacher's provenance is renamed.
EARLIER_MODEL = re.compile(r"Bobcat\s1(?![.\d])|bobcat-?[1](?![.\d])")
KEY_RENAMES = {"teacher_bobcat1": "teacher"}
DROP_KEY_PREFIXES = ("usd", "infra/")
REWRITES = (
    (re.compile(r"infra/\w+/nvfp4_quantize\.py"), "scripts/nvfp4_quantize.py"),
    (re.compile(r"\s*\((?:g7e|g6e|p5|p5e|p5en|p6-b200|p6-b300)\.[0-9a-z]+\)"), ""),
    (re.compile(r"\.aws-local/[\w.-]+(?: \(internal\))?"), "an internal file"),
    (re.compile(r"bobcat-[1]: Qwen3\.8-27B \+ release LoRA"), "Qwen3.8-27B + Bobcat LoRA"),
    (re.compile(r"bobcat1_mixture"), "teacher_training_mixture"),
    (re.compile(r";\s*Bobcat\s1 stays unchanged"), ""),
    (re.compile(r"was used by Bobcat\s1 training"), "was used by an earlier training build"),
    (re.compile(r"\s+and vs Bobcat\s1(?![.\d])"), ""),
    (re.compile(r"Flash corpus / Bobcat\s1 mixture 51"), "the Flash corpus 51"),
    (re.compile(r"\s*\(Bobcat\s1 39/85\)"), " (untrained base 58/85)"),
    (re.compile(r"is unchanged\sfrom the frozen file"), "is identical to the frozen file"),
)
# Named so a hit can be reported without echoing the text around it.
BLOCKED = {
    "storage-uri": r"s3://|\.amazonaws\.com|arn:aws:",
    "cloud-region": r"\b(?:us|eu|ap|sa|ca|me|af|il|mx)-(?:north|south|east|west|central|"
                    r"northeast|northwest|southeast|southwest)-\d\b",
    "instance-id": r"\bi-0[0-9a-f]{16}\b|\b(?:vol|sg|ami|subnet|vpc)-0[0-9a-f]{8,17}\b",
    "ip-address": r"\b(?!127\.0\.0\.1\b|0\.0\.0\.0\b)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "price": r"\$\s?\d|\bUSD\b|\busd_|per[ _-]hour|/MTok|\bMTok\b",
    "market": r"\bSpot\b|\bOn-Demand\b",
    "local-path": r"/opt/dlami|/home/ubuntu|\.aws-local|/Users/",
    "credential": r"AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|"
                  r"sk-[A-Za-z0-9]{20,}|fxb_[0-9A-Za-z]{32}|BEGIN [A-Z ]*PRIVATE KEY",
    "private-email": r"[A-Za-z0-9._%+-]+@(?:gmail|naver|amazon)\.com",
    "earlier-model": EARLIER_MODEL.pattern,
}
# Instance types are allowed in a card (its "Running on AWS" how-to) but not in a manifest.
MANIFEST_ONLY = {"instance-type": r"\b(?:g7e|g6e|g6|g5|p4d|p5|p5en|p6-b200|p6-b300)\.\d*x?large\b"}


def blocked(text: str, *, manifest: bool) -> list[str]:
    rules = {**BLOCKED, **(MANIFEST_ONLY if manifest else {})}
    return [name for name, pattern in rules.items() if re.search(pattern, text)]


def public_manifest(internal: dict, internal_sha256: str) -> dict:
    """The internal release manifest without storage, cost, host and orchestration detail."""

    def clean(value):
        if isinstance(value, dict):
            return {KEY_RENAMES.get(k, k): clean(v) for k, v in value.items()
                    if k not in DROP_KEYS and not k.startswith(DROP_KEY_PREFIXES)
                    and (k in KEY_RENAMES or not EARLIER_MODEL.search(k))}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            for pattern, replacement in REWRITES:
                value = pattern.sub(replacement, value)
        return value

    out = clean(internal)
    out["public_copy"] = {
        "derived_from_sha256": internal_sha256,
        "changes": "storage locations, cost figures, host and instance details, internal "
                   "report and orchestration paths, and results of models other than this "
                   "release and its untrained base removed; the NVFP4 quantizer named by its "
                   "public path, scripts/nvfp4_quantize.py; nothing else changed",
    }
    return out


def render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def expected_public(release: Release) -> str | None:
    """The public manifest text the internal manifest implies (None off the dev line)."""
    if not release.internal_manifest.exists():
        return None
    internal = json.loads(release.internal_manifest.read_text())
    return render(public_manifest(internal, file_hash(release.internal_manifest)))


def published_manifest(release: Release) -> tuple[dict, str]:
    if release.public_manifest is None:
        text = release.internal_manifest.read_text()
        return json.loads(text), text
    if not release.public_manifest.exists():
        raise SystemExit(f"{release.public_manifest.relative_to(ROOT)} is missing; run "
                         "--refresh-public-manifest on the development line.")
    text = release.public_manifest.read_text()
    expected = expected_public(release)
    if expected is not None and expected != text:
        raise SystemExit("The public manifest differs from the internal release manifest; "
                         "run --refresh-public-manifest and review the change.")
    return json.loads(text), text


def dig(data: dict, path: tuple[str, ...]):
    for key in path:
        data = data[key]
    return data


# ---------------------------------------------------------------- card checks


def front_matter(card: str) -> dict[str, str]:
    """Top-level scalar keys of the card's YAML front matter (enough for the checks)."""
    if not card.startswith("---\n"):
        raise SystemExit("The model card has no YAML front matter.")
    block = card[4:card.index("\n---", 4)]
    out = {}
    for line in block.splitlines():
        match = re.match(r"^([A-Za-z_]+):\s*(\S.*)?$", line)
        if match and match.group(2):
            out[match.group(1)] = match.group(2).strip()
    return out


def check_card(release: Release, card: str, manifest: dict) -> None:
    meta = front_matter(card)
    base = manifest["base"]
    expected = release.base_model or base["repo"]
    if meta.get("base_model") != expected:
        raise SystemExit("The card's base_model differs from the release manifest.")
    revision = meta.get("base_model_revision")
    if revision is not None and (expected != base["repo"] or revision != base["revision"]):
        raise SystemExit("The card's base_model_revision differs from the release manifest.")
    for key in ("license", "pipeline_tag", "library_name"):
        if key not in meta:
            raise SystemExit(f"The card's front matter has no {key}.")
    if release.repo_id not in card:
        raise SystemExit(f"The card never names its repository {release.repo_id}.")


# ---------------------------------------------------------------- adapter source


def download_s3(uri: str, folder: Path, client=None) -> Path:
    """Copy the adapter files under an s3://bucket/prefix/ into folder."""
    match = re.match(r"^s3://([^/]+)/(.*)$", uri)
    if not match:
        raise SystemExit("--from-s3 takes s3://bucket/prefix/")
    bucket, prefix = match.group(1), match.group(2).rstrip("/") + "/"
    if client is None:
        import boto3

        client = boto3.client("s3", region_name=os.environ.get("AWS_REGION")
                              or os.environ.get("AWS_DEFAULT_REGION"))
    folder.mkdir(parents=True)
    for name in ADAPTER_FILES:
        client.download_file(bucket, prefix + name, str(folder / name))
    return folder


def refuse_credentials() -> None:
    if any(os.environ.get(name) for name in CREDENTIAL_VARIABLES):
        raise SystemExit("Refusing to stage with Hugging Face credentials in the environment.")


def stage(release: Release, adapter_dir: Path, out: Path) -> dict:
    manifest, manifest_text = published_manifest(release)
    card = release.card.read_text()
    check_card(release, card, manifest)
    for label, text, is_manifest in (("model card", card, False),
                                     ("release manifest", manifest_text, True)):
        hits = blocked(text, manifest=is_manifest)
        if hits:
            raise SystemExit(f"The {label} matches blocked patterns: {', '.join(hits)}")
    have = file_hash(adapter_dir / "adapter_model.safetensors")
    if have != dig(manifest, release.adapter_hash_path):
        raise SystemExit("The adapter does not match the release manifest.")
    out.mkdir(parents=True)
    shutil.copy(adapter_dir / "adapter_model.safetensors", out / "adapter_model.safetensors")
    # PEFT records the local path the adapter was trained against; publish the upstream
    # repository and the pinned revision instead, so `PeftModel.from_pretrained` resolves.
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    config["base_model_name_or_path"] = manifest["base"]["repo"]
    config["revision"] = manifest["base"]["revision"]
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "README.md").write_text(card)
    (out / "bobcat-release-manifest.json").write_text(manifest_text)
    if release.assets.exists():
        shutil.copytree(release.assets, out / "assets")
    write_identifiers(out)
    files = {str(p.relative_to(out)): file_hash(p)
             for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "SHA256SUMS.json").write_text(json.dumps(files, indent=1) + "\n")
    return {"release": release.name, "repo_id": release.repo_id, "staged": str(out),
            "files": len(files) + 1, "adapter_model_sha256": have, "uploaded": False}


def link_or_copy(source: Path, target: Path) -> None:
    """A hard link where the file system allows it (checkpoints are tens of GB), else a copy.
    The hashes are checked before and recorded after, so either gives the same package."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy(source, target)


def stage_weights(release: Release, weights_dir: Path, compiler_dir: Path, out: Path) -> dict:
    """A ready-to-serve checkpoint package (see the module docstring)."""
    manifest, manifest_text = published_manifest(release)
    card = release.card.read_text()
    check_card(release, card, manifest)
    for label, text, is_manifest in (("model card", card, False),
                                     ("release manifest", manifest_text, True)):
        hits = blocked(text, manifest=is_manifest)
        if hits:
            raise SystemExit(f"The {label} matches blocked patterns: {', '.join(hits)}")
    expected = dig(manifest, release.weight_hashes_path)
    if not expected:
        raise SystemExit("The release manifest records no hashes for these weights.")
    for name, digest in expected.items():
        if not (weights_dir / name).is_file() or file_hash(weights_dir / name) != digest:
            raise SystemExit(f"{name} does not match the release manifest.")
    for name in TOKENIZER_FILES:
        for folder in (weights_dir, compiler_dir):
            if file_hash(folder / name) != manifest["tokenizer"][name]:
                raise SystemExit(f"{folder.name}/{name} differs from the manifest's tokenizer.")
    receipt = json.loads((compiler_dir / "bobcat-download.json").read_text())
    if (receipt["repo"], receipt["revision"]) != (manifest["base"]["repo"],
                                                  manifest["base"]["revision"]):
        raise SystemExit("The compiler folder is not the manifest's pinned base.")
    for name in COMPILER_FILES[:-1]:
        if file_hash(compiler_dir / name) != receipt["files"][name]["sha256"]:
            raise SystemExit(f"compiler {name} differs from its download receipt.")
    out.mkdir(parents=True)
    for path in sorted(weights_dir.iterdir()):
        if not path.is_file() or path.name in SKIPPED_WEIGHT_FILES:
            continue
        if path.name in RECEIPT_PATH_FIELDS:
            data = json.loads(path.read_text())
            data[RECEIPT_PATH_FIELDS[path.name]] = release.source_label
            text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            if blocked(text, manifest=True):
                raise SystemExit(f"{path.name} still holds host details after rewriting.")
            (out / path.name).write_text(text)
        else:
            link_or_copy(path, out / path.name)
    (out / "compiler").mkdir()
    for name in COMPILER_FILES:
        shutil.copy(compiler_dir / name, out / "compiler" / name)
    if release.license_from == "compiler":
        if file_hash(compiler_dir / "LICENSE") != receipt["files"]["LICENSE"]["sha256"]:
            raise SystemExit("The upstream LICENSE differs from its download receipt.")
        shutil.copy(compiler_dir / "LICENSE", out / "LICENSE")
    else:
        shutil.copy(ROOT / "LICENSE", out / "LICENSE")
    shutil.copy(release.card.with_name("NOTICE"), out / "NOTICE")
    (out / "README.md").write_text(card)
    (out / "bobcat-release-manifest.json").write_text(manifest_text)
    if release.assets.exists():
        shutil.copytree(release.assets, out / "assets")
    write_identifiers(out)
    files = {str(p.relative_to(out)): file_hash(p)
             for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "SHA256SUMS.json").write_text(json.dumps(files, indent=1) + "\n")
    return {"release": release.name, "repo_id": release.repo_id, "staged": str(out),
            "files": len(files) + 1, "weights_sha256": expected, "uploaded": False}


def write_identifiers(out: Path) -> None:
    # The key every loader reads (bobcat.api_server.load_compiler and the other servers), so
    # `--identifiers bobcat-identifiers.json` works from the downloaded package.
    identifiers = json.loads(IDENTIFIERS.read_text())["identifiers"]
    (out / "bobcat-identifiers.json").write_text(json.dumps(
        {"identifiers": identifiers,
         "note": "Bobcat keeps these where they are single ordinary tokens of the base "
                 "tokenizer, then extends with fixed Greek/Cyrillic/Hebrew/Latin-1/"
                 "Armenian/Georgian letters (bobcat.student_readout.identifier_scheme)."},
        ensure_ascii=False, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", choices=sorted(RELEASES), required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--adapter-dir", type=Path,
                        help="folder with adapter_config.json and adapter_model.safetensors")
    source.add_argument("--from-s3", metavar="S3_URI",
                        help="s3://bucket/prefix/ holding the two adapter files")
    source.add_argument("--weights-dir", type=Path,
                        help="weight packages: the served checkpoint folder")
    parser.add_argument("--compiler-dir", type=Path,
                        help="weight packages: the pinned base download (receipt, tokenizer)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--refresh-public-manifest", action="store_true",
                        help="rewrite the card folder's public manifest (development line)")
    args = parser.parse_args(argv)
    release = RELEASES[args.release]
    if args.refresh_public_manifest:
        if release.public_manifest is None:
            raise SystemExit(f"{release.name} publishes its internal manifest unchanged.")
        text = expected_public(release)
        if text is None:
            raise SystemExit("The internal release manifest is not in this checkout.")
        hits = blocked(text, manifest=True)
        if hits:
            raise SystemExit(f"The public manifest still matches: {', '.join(hits)}")
        release.public_manifest.parent.mkdir(parents=True, exist_ok=True)
        release.public_manifest.write_text(text)
        print(json.dumps({"written": str(release.public_manifest.relative_to(ROOT)),
                          "sha256": file_hash(release.public_manifest)}))
        return
    refuse_credentials()
    if args.out is not None and args.out.exists():
        raise SystemExit("Choose a new output folder.")
    if release.kind == "weights":
        if args.out is None or args.weights_dir is None or args.compiler_dir is None:
            parser.error("weight packages need --weights-dir, --compiler-dir and --out")
        print(json.dumps(stage_weights(release, args.weights_dir, args.compiler_dir, args.out)))
        return
    if args.out is None or (args.adapter_dir is None and args.from_s3 is None):
        parser.error("staging needs --out and one of --adapter-dir / --from-s3")
    download = None
    adapter_dir = args.adapter_dir
    if args.from_s3:
        download = args.out.with_name(args.out.name + ".download")
        if download.exists():
            raise SystemExit(f"Remove the leftover download folder {download} first.")
        adapter_dir = download_s3(args.from_s3, download)
    try:
        result = stage(release, adapter_dir, args.out)
    finally:
        if download is not None:
            shutil.rmtree(download, ignore_errors=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
