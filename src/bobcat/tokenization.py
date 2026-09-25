from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from bobcat.schema import SENTINEL_TEXT, Example, file_hash

SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]
PAD, UNK, BOS, EOS, SEP, MASK = range(len(SPECIAL))
ENCODING_PROFILE = "literal_special_strings_v1"


class ScratchTokenizer:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.backend = Tokenizer.from_file(str(path))
        if any(self.backend.token_to_id(token) != i for i, token in enumerate(SPECIAL)):
            raise ValueError("Tokenizer special IDs do not match the Bobcat format.")
        # Source text is data. Only the host may insert boundary/mask IDs.
        # tokenizers calls this flag encode_special_tokens (split their spelling
        # into ordinary tokens rather than recognizing the reserved token).
        self.backend.encode_special_tokens = True
        self.encoding_profile = ENCODING_PROFILE
        self.digest = file_hash(path)
        self.vocab_size = self.backend.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text, add_special_tokens=False).ids

    @classmethod
    def train(cls, examples: list[Example], path: Path, vocab_size: int = 2048):
        if not examples or any(example.split != "train" for example in examples):
            raise ValueError("Tokenizer training accepts only the training partition.")
        if path.exists():
            raise ValueError("Refusing to overwrite an existing tokenizer.")
        backend = Tokenizer(models.BPE(unk_token="[UNK]"))
        backend.normalizer = normalizers.NFKC()
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        # Only public input fields, never targets, IDs, split names or oracle programs.
        texts = set(SENTINEL_TEXT.values())
        for example in examples:
            texts.update([example.context, example.instruction])
            texts.update(choice.text for choice in example.choices)
        backend.train_from_iterator(sorted(texts), trainer=trainer)
        path.parent.mkdir(parents=True, exist_ok=True)
        backend.save(str(path))
        tokenizer = cls(path)
        provenance = {
            "origin": "trained_from_scratch",
            "training_partition": "train",
            "training_examples": len(examples),
            "unique_input_texts": len(texts),
            "vocab_size": tokenizer.vocab_size,
            "sha256": tokenizer.digest,
        }
        path.with_suffix(".manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return tokenizer
