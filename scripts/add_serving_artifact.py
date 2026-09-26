"""Add one entry under `serving_artifacts` of a frozen release manifest, changing nothing else.

The frozen fields are checked before and after: the manifest without `serving_artifacts`,
serialised as it was frozen (json indent 2 + newline), must keep the sha256 recorded at the
freeze (for bobcat-flash-1.1: 88056ce0..., also bound by the final-opening record), and every
existing `serving_artifacts` entry must stay byte-identical. The file is rewritten in the
same serialisation (json indent 2 + newline), so an unchanged manifest round-trips exactly.

    python scripts/add_serving_artifact.py --manifest release/bobcat-flash-1.1-manifest.json \
        --frozen-sha256 88056ce0628ce36d3a1d5907d97016520a0c81a437facaad701abb527fae612d \
        --key routing_length_rule --entry entry.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def dumps(value) -> str:
    return json.dumps(value, indent=2) + "\n"


def frozen_sha256(manifest: dict) -> str:
    base = {k: v for k, v in manifest.items() if k != "serving_artifacts"}
    return hashlib.sha256(dumps(base).encode()).hexdigest()


def add(path: Path, key: str, entry: dict, expected: str, replace: bool = False) -> dict:
    raw = path.read_text()
    manifest = json.loads(raw)
    if dumps(manifest) != raw:
        raise ValueError("The manifest does not round-trip; refusing to rewrite it.")
    if manifest.get("status") != "frozen":
        raise ValueError("Not a frozen manifest.")
    if frozen_sha256(manifest) != expected:
        raise ValueError("The frozen fields already differ from the recorded sha256.")
    artifacts = manifest.setdefault("serving_artifacts", {})
    if key in artifacts and not replace:
        raise ValueError(f"serving_artifacts.{key} exists.")
    before = {k: json.dumps(v, sort_keys=True) for k, v in artifacts.items() if k != key}
    artifacts[key] = entry
    after = {k: json.dumps(v, sort_keys=True) for k, v in artifacts.items() if k != key}
    if before != after or frozen_sha256(manifest) != expected:
        raise AssertionError("A frozen field or another serving entry changed.")
    path.write_text(dumps(manifest))
    return {"frozen_sha256": frozen_sha256(json.loads(path.read_text())),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "serving_artifacts": sorted(artifacts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-sha256", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--entry", type=Path, required=True, help="JSON object to add")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    entry = json.loads(args.entry.read_text())
    print(json.dumps(add(args.manifest, args.key, entry, args.frozen_sha256, args.replace)))


if __name__ == "__main__":
    main()
