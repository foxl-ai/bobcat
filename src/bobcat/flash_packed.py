"""Packed scoring for pure full-attention students (Bobcat Flash, 2026-09-26).

All questions of one request go into ONE sequence: the shared prefix (template + state, the
longest common token prefix of the compiled questions) once, then every question's own
suffix. A block attention mask lets each question token attend to the shared prefix and to
earlier tokens of its own question only, and every question's positions continue from the
prefix end, so each question is computed exactly as if it were sent alone (the question
isolation rule), while the state is computed once. The first answer position of each
question (the last token of its suffix) is read with the LM head restricted to the offered
identifiers, as in the unpacked path.

Two execution shapes:
  single   one forward over [prefix | q1 | q2 | ...] with a [T, T] block mask; requests are
           batched (right-padded, pads masked out) when their packs are short;
  staged   long packs: the prefix once with a KV cache, then groups of question suffixes
           against it with a [S, P + S] block mask (the cache is cropped back after a group).
Gated DeltaNet hybrids and sliding-window layers are refused: a recurrent state or a window
cannot be shared this way. `api_engine` wraps the scorer for `bobcat.api_server.create_api`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

SINGLE_MAX_TOKENS = 12288   # a pack up to this length runs as one masked forward
GROUP_TOKENS = 8192         # staged: question tokens per suffix group
BATCH_TOKENS = 65536        # single: padded tokens per batched forward


def common_prefix(sequences) -> int:
    first, length = sequences[0], min(map(len, sequences))
    for index in range(length):
        if any(s[index] != first[index] for s in sequences[1:]):
            return index
    return length


@dataclass
class Pack:
    ids: list[int]
    positions: list[int]
    segments: list[int]      # -1 = shared prefix, k = question k
    readouts: list[int]      # packed index of each question's first answer position
    prefix: int
    unique: list[int]        # question -> index of the distinct sequence it shares

    @property
    def length(self) -> int:
        return len(self.ids)


def pack(sequences: list[list[int]]) -> Pack:
    """Shared prefix once, then each distinct question suffix. Identical compiled questions
    (same content under different IDs) are scored once."""
    distinct, unique = [], []
    seen = {}
    for sequence in sequences:
        key = tuple(sequence)
        if key not in seen:
            seen[key] = len(distinct)
            distinct.append(list(sequence))
        unique.append(seen[key])
    prefix = common_prefix(distinct) if len(distinct) > 1 else len(distinct[0]) - 1
    prefix = min(prefix, min(len(s) for s in distinct) - 1)  # every question keeps >= 1 token
    ids = list(distinct[0][:prefix])
    positions = list(range(prefix))
    segments = [-1] * prefix
    readouts = []
    for k, sequence in enumerate(distinct):
        suffix = sequence[prefix:]
        ids += suffix
        positions += list(range(prefix, len(sequence)))
        segments += [k] * len(suffix)
        readouts.append(len(ids) - 1)
    return Pack(ids, positions, segments, readouts, prefix, unique)


def block_mask(torch, segments, positions, *, keys_segments=None, keys_positions=None,
               device=None):
    """Boolean [q, k] mask: a query sees prefix keys (segment -1) at earlier-or-equal
    positions and keys of its own segment at earlier-or-equal positions."""
    q_seg = torch.as_tensor(segments, device=device)
    q_pos = torch.as_tensor(positions, device=device)
    k_seg = q_seg if keys_segments is None else torch.as_tensor(keys_segments, device=device)
    k_pos = q_pos if keys_positions is None else torch.as_tensor(keys_positions, device=device)
    same = (k_seg[None, :] == -1) | (k_seg[None, :] == q_seg[:, None])
    causal = k_pos[None, :] <= q_pos[:, None]
    if keys_segments is None:
        # Within one packed sequence the index order also has to be causal for the prefix.
        index = torch.arange(len(segments), device=device)
        causal = causal & (index[None, :] <= index[:, None])
    return same & causal


class PackedScorer:
    def __init__(self, model_dir, *, device="cuda", dtype=None, attn="sdpa"):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        self.torch = torch
        config = AutoConfig.from_pretrained(model_dir)
        text = getattr(config, "text_config", None) or config
        types = set(getattr(text, "layer_types", None) or ["full_attention"])
        if types != {"full_attention"} or getattr(text, "sliding_window", None) and \
                getattr(text, "use_sliding_window", False):
            raise NotImplementedError(f"Packed scoring needs pure full attention, not {types}.")
        self.softcap = getattr(text, "final_logit_softcapping", None)
        dtype = dtype or torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, device_map={"": device}, attn_implementation=attn).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = self.model.device
        self.backbone = self.model.get_decoder()
        self.lm_weight = self.model.get_output_embeddings().weight
        self.name = f"packed ({attn}, {str(dtype).removeprefix('torch.')})"

    # ------------------------------------------------------------------ readout
    def _read(self, hidden, index, options):
        torch = self.torch
        with torch.inference_mode():
            z = (self.lm_weight[torch.as_tensor(options, device=self.device)]
                 @ hidden[index].to(self.lm_weight.dtype)).float()
            if self.softcap:
                z = torch.tanh(z / self.softcap) * self.softcap
        return z.tolist()

    # ------------------------------------------------------------------ shapes
    def _single(self, packs: list[Pack]):
        """One right-padded batch of packs, each with its own block mask."""
        torch = self.torch
        longest = max(p.length for p in packs)
        ids = torch.zeros((len(packs), longest), dtype=torch.long, device=self.device)
        pos = torch.zeros((len(packs), longest), dtype=torch.long, device=self.device)
        mask = torch.zeros((len(packs), 1, longest, longest), dtype=torch.bool,
                           device=self.device)
        for b, p in enumerate(packs):
            n = p.length
            ids[b, :n] = torch.as_tensor(p.ids, device=self.device)
            pos[b, :n] = torch.as_tensor(p.positions, device=self.device)
            mask[b, 0, :n, :n] = block_mask(torch, p.segments, p.positions, device=self.device)
            if n < longest:  # pad queries attend to themselves only (never read)
                mask[b, 0, torch.arange(n, longest), torch.arange(n, longest)] = True
        with torch.inference_mode():
            hidden = self.backbone(input_ids=ids, position_ids=pos, attention_mask=mask,
                                   use_cache=False).last_hidden_state
        return [hidden[b] for b in range(len(packs))]

    def _staged(self, p: Pack, options_by_segment):
        """Prefix once with a KV cache, then suffix groups against it."""
        torch = self.torch
        from transformers import DynamicCache

        bounds = {}
        for index in range(p.prefix, p.length):
            bounds.setdefault(p.segments[index], [index, index])[1] = index
        groups, current, tokens = [], [], 0
        for k in sorted(bounds):
            size = bounds[k][1] - bounds[k][0] + 1
            if current and tokens + size > GROUP_TOKENS:
                groups.append(current)
                current, tokens = [], 0
            current.append(k)
            tokens += size
        if current:
            groups.append(current)
        values = {}
        with torch.inference_mode():
            cache = DynamicCache()
            self.backbone(input_ids=torch.as_tensor([p.ids[:p.prefix]], device=self.device),
                          position_ids=torch.arange(p.prefix, device=self.device)[None],
                          past_key_values=cache, use_cache=True)
            for group in groups:
                lo, hi = bounds[group[0]][0], bounds[group[-1]][1] + 1
                q_seg, q_pos = p.segments[lo:hi], p.positions[lo:hi]
                mask = block_mask(torch, q_seg, q_pos, keys_segments=[-1] * p.prefix + q_seg,
                                  keys_positions=list(range(p.prefix)) + q_pos,
                                  device=self.device)
                hidden = self.backbone(
                    input_ids=torch.as_tensor([p.ids[lo:hi]], device=self.device),
                    position_ids=torch.as_tensor([q_pos], device=self.device),
                    attention_mask=mask[None, None], past_key_values=cache,
                    use_cache=True).last_hidden_state[0]
                cache.crop(p.prefix)
                for k in group:
                    values[k] = self._read(hidden, p.readouts[k] - lo, options_by_segment[k])
        return values

    # ------------------------------------------------------------------ entry points
    def score_requests(self, requests):
        """requests: [(sequences, option_ids)] -> [[logits per question]] (request order)."""
        packs = [pack(seqs) for seqs, _ in requests]
        results = [None] * len(requests)
        short = [i for i, p in enumerate(packs) if p.length <= SINGLE_MAX_TOKENS]
        batch = []
        for i in sorted(short, key=lambda i: packs[i].length) + [None]:
            if i is not None and (not batch or (len(batch) + 1) * max(
                    packs[i].length, max(packs[j].length for j in batch)) <= BATCH_TOKENS):
                batch.append(i)
                continue
            if batch:
                hiddens = self._single([packs[j] for j in batch])
                for j, hidden in zip(batch, hiddens, strict=True):
                    results[j] = self._collect(packs[j], requests[j][1], hidden=hidden)
            batch = [i] if i is not None else []
        for i, p in enumerate(packs):
            if results[i] is None:
                options = self._segment_options(p, requests[i][1])
                values = self._staged(p, options)
                results[i] = [values[p.unique[q]] for q in range(len(p.unique))]
        return results

    @staticmethod
    def _segment_options(p: Pack, option_ids):
        options = {}
        for q, k in enumerate(p.unique):
            options.setdefault(k, option_ids[q])
        return options

    def _collect(self, p: Pack, option_ids, *, hidden):
        options = self._segment_options(p, option_ids)
        values = {k: self._read(hidden, p.readouts[k], opts) for k, opts in options.items()}
        return [values[p.unique[q]] for q in range(len(p.unique))]

    def score_unpacked(self, sequences, option_ids):
        """Reference path: every question alone (for equivalence checks)."""
        torch = self.torch
        out = []
        with torch.inference_mode():
            for sequence, options in zip(sequences, option_ids, strict=True):
                hidden = self.backbone(input_ids=torch.as_tensor([sequence], device=self.device),
                                       use_cache=False).last_hidden_state[0]
                out.append(self._read(hidden, len(sequence) - 1, options))
        return out


class PackedEngine:
    """`bobcat.api_server.create_api` engine: requests arriving within `window_ms` share one
    batched packed forward on one GPU."""

    def __init__(self, scorer: PackedScorer, *, window_ms: float = 2.0, max_batch: int = 64):
        self.scorer, self.window, self.max_batch = scorer, window_ms / 1000, max_batch
        self.name = scorer.name
        self.queue: asyncio.Queue | None = None
        self.worker = None
        self.stats = {"forward_batches": 0, "requests": 0, "seconds": 0.0}

    async def _run(self):
        while True:
            first = await self.queue.get()
            items = [first]
            deadline = time.perf_counter() + self.window
            while len(items) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), remaining))
                except TimeoutError:
                    break
            tick = time.perf_counter()
            try:
                results = await asyncio.to_thread(
                    self.scorer.score_requests, [(s, o) for s, o, _ in items])
                for (_, _, future), result in zip(items, results, strict=True):
                    future.set_result(result)
            except Exception as error:  # every waiting request fails closed
                for _, _, future in items:
                    if not future.done():
                        future.set_exception(error)
            self.stats["forward_batches"] += 1
            self.stats["requests"] += len(items)
            self.stats["seconds"] += time.perf_counter() - tick

    async def logits(self, sequences, option_ids, prefix: int):
        if self.queue is None:
            self.queue = asyncio.Queue()
            self.worker = asyncio.create_task(self._run())
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((list(map(list, sequences)), list(map(list, option_ids)), future))
        return await future
