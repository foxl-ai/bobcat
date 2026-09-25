from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from bobcat.model import ModelConfig
from bobcat.schema import SENTINEL_TEXT, SENTINELS, Example
from bobcat.supervision import MEAN_TARGET
from bobcat.tokenization import BOS, EOS, SEP, ScratchTokenizer


def padded(rows: list[list[int]]) -> Tensor:
    result = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(row)
    return result


@dataclass
class Batch:
    tensors: dict[str, Tensor]
    examples: list[Example]
    candidate_ids: list[list[str]]
    nonpadding_tokens: int
    unique_contexts: int
    padded_tokens: int

    def model_inputs(self) -> dict[str, Tensor]:
        """Only encoded evidence reaches the model; labels remain loss inputs."""
        return {key: self.tensors[key] for key in (
            "context_ids", "context_index", "schema_ids", "schema_candidate_mask",
            "candidate_keep", "joint_ids", "candidate_positions", "joint_candidate_mask",
        )}

    def to(self, device: torch.device) -> Batch:
        return Batch(
            # Keep original means in float64 on the host; the supervised loss
            # transfers them explicitly. Inference does not need these labels.
            tensors={k: v if k == "score_targets" else v.to(device)
                     for k, v in self.tensors.items()},
            examples=self.examples,
            candidate_ids=self.candidate_ids,
            nonpadding_tokens=self.nonpadding_tokens,
            unique_contexts=self.unique_contexts,
            padded_tokens=self.padded_tokens,
        )


class EncodedDataset:
    def __init__(
        self, examples: list[Example], tokenizer: ScratchTokenizer, config: ModelConfig,
        *, include_sentinels: bool = True,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.config = config
        self.sentinel_ids = list(SENTINELS) if include_sentinels else []
        self.sentinel_texts = [SENTINEL_TEXT[key] for key in self.sentinel_ids]
        for example in examples:
            if example.supervision == "score_mean":
                example.validate()
                if include_sentinels:
                    raise ValueError("Ordinal means cannot add unobserved sentinel levels.")
        self.contexts = {
            example.context_id: [BOS, *tokenizer.encode(example.context), EOS]
            for example in examples
        }
        self.instructions = {e.id: tokenizer.encode(e.instruction) for e in examples}
        self.choices = {
            text: tokenizer.encode(text)
            for text in {*self.sentinel_texts, *(c.text for e in examples for c in e.choices)}
        }
        self.groups = defaultdict(list)
        for index, example in enumerate(examples):
            self.groups[example.context_id].append(index)
            if len(self.contexts[example.context_id]) > config.max_context_tokens:
                raise ValueError(f"Context too long; refusing to truncate policy: {example.id}")
            lengths = [
                len(self.instructions[example.id]) + len(self.choices[text]) + 3
                for text in [*(c.text for c in example.choices), *self.sentinel_texts]
            ]
            if max(lengths) > config.max_schema_tokens:
                raise ValueError(f"Schema too long: {example.id}")
            joint_length = (
                len(self.contexts[example.context_id])
                + len(self.instructions[example.id])
                + 1
                + sum(
                    len(self.choices[text]) + 2
                    for text in [
                        *(c.text for c in example.choices),
                        *self.sentinel_texts,
                    ]
                )
            )
            if config.architecture == "joint" and joint_length > config.max_joint_tokens:
                raise ValueError(f"Joint input too long: {example.id}")
        self.group_keys = list(self.groups)
        self.world_groups = defaultdict(list)
        for index, example in enumerate(examples):
            self.world_groups[example.group_id].append(index)
        self.world_keys = list(self.world_groups)

    def training_indices(
        self, step: int, groups_per_batch: int, seed: int, strategy: str = "context"
    ) -> list[int]:
        rng = random.Random(seed + step * 1000003)
        if strategy == "context":
            keys, groups = self.group_keys, self.groups
        elif strategy == "world":
            keys, groups = self.world_keys, self.world_groups
        else:
            raise ValueError("Grouping strategy must be context or world.")
        sampled = rng.sample(keys, min(groups_per_batch, len(keys)))
        return [index for key in sampled for index in groups[key]]

    def collate(
        self,
        indices: list[int],
        *,
        shuffle_seed: int | None = None,
        reuse_context: bool = True,
    ) -> Batch:
        examples = [self.examples[index] for index in indices]
        contexts: list[list[int]] = []
        context_map = {}
        context_indices = []
        schema_rows = []
        candidate_ids = []
        joint_rows = []
        positions = []
        schema_masks = []
        joint_spans = []
        targets = []
        score_targets = []
        rng = random.Random(shuffle_seed)
        max_candidates = max(len(e.choices) + len(self.sentinel_ids) for e in examples)
        for example in examples:
            key = example.context_id if reuse_context else example.id
            if key not in context_map:
                context_map[key] = len(contexts)
                contexts.append(self.contexts[example.context_id])
            context_indices.append(context_map[key])
            choices = list(example.choices)
            if shuffle_seed is not None and example.kind != "ordinal":
                rng.shuffle(choices)
            ids = [c.id for c in choices] + self.sentinel_ids
            texts = [c.text for c in choices] + self.sentinel_texts
            candidate_ids.append(ids)
            targets.append(MEAN_TARGET if example.supervision == "score_mean"
                           else ids.index(example.target))
            score_targets.append(example.score_target if example.score_target is not None else 0.0)
            instruction = self.instructions[example.id]
            schema = [[BOS, *instruction, SEP, *self.choices[text], EOS] for text in texts]
            # Padded candidates are masked from set attention and from the output.
            schema.extend([[BOS, EOS]] * (max_candidates - len(schema)))
            schema_rows.extend(schema)
            for index, row in enumerate(schema):
                mask = [0] * len(row)
                if index < len(texts):
                    mask[len(instruction) + 2 : -1] = [1] * len(self.choices[texts[index]])
                schema_masks.append(mask)
            joint = [*self.contexts[example.context_id], *instruction, SEP]
            anchors = []
            spans = []
            for text in texts:
                anchors.append(len(joint))
                spans.append((len(joint) + 1, len(joint) + 1 + len(self.choices[text])))
                joint.extend([BOS, *self.choices[text], EOS])
            anchors.extend([0] * (max_candidates - len(anchors)))
            joint_rows.append(joint)
            positions.append(anchors)
            joint_spans.append(spans)
        context_ids = padded(contexts)
        schema_ids = padded(schema_rows).reshape(len(examples), max_candidates, -1)
        joint_ids = padded(joint_rows)
        schema_candidate_mask = (
            padded(schema_masks).bool().reshape(len(examples), max_candidates, -1)
        )
        joint_candidate_mask = torch.zeros(
            len(examples), max_candidates, joint_ids.shape[1], dtype=torch.bool
        )
        for batch_index, spans in enumerate(joint_spans):
            for choice_index, (start, end) in enumerate(spans):
                joint_candidate_mask[batch_index, choice_index, start:end] = True
        candidate_keep = torch.tensor(
            [[True] * len(ids) + [False] * (max_candidates - len(ids)) for ids in candidate_ids]
        )
        tensors = {
            "context_ids": context_ids,
            "context_index": torch.tensor(context_indices),
            "schema_ids": schema_ids,
            "schema_candidate_mask": schema_candidate_mask,
            "candidate_keep": candidate_keep,
            "joint_ids": joint_ids,
            "candidate_positions": torch.tensor(positions),
            "joint_candidate_mask": joint_candidate_mask,
            "targets": torch.tensor(targets),
            "score_targets": torch.tensor(score_targets, dtype=torch.float64),
        }
        relevant = (
            [context_ids, schema_ids] if self.config.architecture == "shared" else [joint_ids]
        )
        return Batch(
            tensors=tensors,
            examples=examples,
            candidate_ids=candidate_ids,
            nonpadding_tokens=sum(int(tensor.ne(0).sum()) for tensor in relevant),
            unique_contexts=len(contexts),
            padded_tokens=sum(tensor.numel() for tensor in relevant),
        )

    def evaluation_batches(self, batch_size: int = 48):
        # Each context's questions stay adjacent, allowing actual reuse in evaluation.
        indices = [index for group in self.groups.values() for index in group]
        for start in range(0, len(indices), batch_size):
            yield self.collate(indices[start : start + batch_size])
