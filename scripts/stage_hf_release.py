"""Stage the Hugging Face package for the Bobcat 1 adapter (never uploads).

Assembles exactly what the Hugging Face repository holds: the LoRA adapter, the model card,
the public release manifest, the identifier list the compiler needs, the card's figures and
a SHA-256 manifest of every file. The adapter must match the release manifest's hash. It
refuses to run with Hugging Face credentials in the environment, so staging can never turn
into an upload by accident; uploading is a separate, deliberate step.

    python scripts/stage_hf_release.py --adapter-dir path/to/adapter --out path/to/stage
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from bobcat.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release/bobcat-1-manifest.json"
IDENTIFIERS = ROOT / "reports/2026-09-22-glm-readout-preflight.json"
CARD = ROOT / "release/hf/README.md"
ASSETS = ROOT / "release/hf/assets"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", type=Path, required=True,
                        help="folder with adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise SystemExit("Refusing to stage with Hugging Face credentials in the environment.")
    if args.out.exists():
        raise SystemExit("Choose a new output folder.")
    release = json.loads(MANIFEST.read_text())
    have = file_hash(args.adapter_dir / "adapter_model.safetensors")
    if have != release["adapter"]["adapter_model_sha256"]:
        raise SystemExit("The adapter does not match the release manifest.")
    args.out.mkdir(parents=True)
    shutil.copy(args.adapter_dir / "adapter_model.safetensors",
                args.out / "adapter_model.safetensors")
    # PEFT records the local path the adapter was trained against; publish the upstream
    # repository and the pinned revision instead, so `PeftModel.from_pretrained` resolves.
    config = json.loads((args.adapter_dir / "adapter_config.json").read_text())
    config["base_model_name_or_path"] = release["base"]["repo"]
    config["revision"] = release["base"]["revision"]
    (args.out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy(CARD, args.out / "README.md")
    shutil.copy(MANIFEST, args.out / "bobcat-release-manifest.json")
    if ASSETS.exists():
        shutil.copytree(ASSETS, args.out / "assets")
    identifiers = json.loads(IDENTIFIERS.read_text())["identifiers"]
    (args.out / "bobcat-identifiers.json").write_text(json.dumps(
        {"glm_preferred_identifiers": identifiers,
         "note": "Bobcat keeps these where they are single ordinary tokens of the base "
                 "tokenizer, then extends with fixed Greek/Cyrillic/Hebrew/Latin-1/"
                 "Armenian/Georgian letters (bobcat.student_readout.identifier_scheme)."},
        ensure_ascii=False, indent=1) + "\n")
    files = {str(p.relative_to(args.out)): file_hash(p)
             for p in sorted(args.out.rglob("*")) if p.is_file()}
    (args.out / "SHA256SUMS.json").write_text(json.dumps(files, indent=1) + "\n")
    print(json.dumps({"staged": str(args.out), "files": len(files), "uploaded": False}))


if __name__ == "__main__":
    main()
