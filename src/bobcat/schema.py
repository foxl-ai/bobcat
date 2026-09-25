from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

NONE = "__none__"
INSUFFICIENT = "__insufficient__"
SENTINELS = (NONE, INSUFFICIENT)
SENTINEL_TEXT = {
    NONE: "None of the offered choices applies.",
    INSUFFICIENT: "There is insufficient information to determine the answer.",
}
KINDS = ("choice", "boolean", "ordinal")


@dataclass(frozen=True)
class Choice:
    id: str
    text: str


@dataclass
class Example:
    id: str
    group_id: str
    family: str
    split: str
    context: str
    instruction: str
    choices: list[Choice]
    target: str | None
    kind: str = "choice"
    pair_id: str = ""
    variant: str = "base"
    metadata: dict[str, Any] = field(default_factory=dict)
    supervision: str = "hard_label"
    score_target: float | None = None

    def validate(self) -> None:
        ids = [choice.id for choice in self.choices]
        if not self.id or not self.group_id or self.kind not in KINDS:
            raise ValueError(f"Invalid example identity/kind: {self.id}")
        if not self.context.strip() or not self.instruction.strip():
            raise ValueError(f"Empty context/instruction: {self.id}")
        if len(ids) < 2 or len(set(ids)) != len(ids) or set(ids) & set(SENTINELS):
            raise ValueError(f"Candidates need >=2 distinct, non-reserved IDs: {self.id}")
        if any(not c.text.strip() or not c.id for c in self.choices):
            raise ValueError(f"Empty candidate: {self.id}")
        if self.supervision == "score_mean":
            from bobcat.supervision import target_values
            target_values({
                "candidate_ids": ids, "kind": self.kind, "target": self.target,
                "supervision": self.supervision, "score_target": self.score_target,
            })
        elif (self.supervision != "hard_label" or self.score_target is not None
              or self.target not in {*ids, *SENTINELS}):
            raise ValueError(f"Target is outside the schema: {self.id}")

    def to_dict(self) -> dict:
        value = asdict(self)
        # Keep the original categorical dataset serialization unchanged.
        if self.supervision == "hard_label" and self.score_target is None:
            value.pop("supervision")
            value.pop("score_target")
        return value

    @classmethod
    def from_dict(cls, value: dict) -> Example:
        copy = dict(value)
        copy["choices"] = [Choice(**choice) for choice in copy["choices"]]
        example = cls(**copy)
        example.validate()
        return example

    def input_fingerprint(self) -> str:
        # IDs, targets, ordering, and all oracle metadata are excluded.
        prompt = {
            "context": self.context,
            "instruction": self.instruction,
            "choices": ([c.text for c in self.choices] if self.kind == "ordinal"
                        else sorted(c.text for c in self.choices)),
            "kind": self.kind,
        }
        return hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()

    @property
    def context_id(self) -> str:
        return hashlib.sha256(self.context.encode()).hexdigest()


def read_examples(path: str | Path) -> list[Example]:
    with Path(path).open() as stream:
        return [Example.from_dict(json.loads(line)) for line in stream if line.strip()]


def write_examples(path: str | Path, examples: list[Example]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for example in examples:
            example.validate()
            stream.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
