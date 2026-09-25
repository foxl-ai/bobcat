"""Deterministic pooled selection reference for a future GLM runtime control.

This is a correctness oracle, not a fast kernel or an installed serving patch.
Equal scores prefer the lower logical group id; selected groups are then ordered
by logical position. Paged and ragged mappings are applied after that selection.
Changing which tied groups are kept can change a model's answer. Full-model
quality, repeated inference, and performance must be measured before adoption.
"""
from __future__ import annotations

import torch


def _integer_vector(value, *, name, rows, device, allow_column=False):
    if allow_column and value.ndim == 2 and value.shape == (rows, 1):
        value = value[:, 0]
    if (value.shape != (rows,) or value.device != device
            or value.dtype not in (torch.int32, torch.int64)):
        raise ValueError(f"{name} must contain one integer per row on the logits device.")
    return value.to(torch.int64)


@torch.no_grad()
def pooled_topk_reference(
    logits,
    group_lengths,
    pool_size,
    topk,
    page_table=None,
    topk_offsets=None,
    seq_lens=None,
    row_starts=None,
    out_rows=None,
    page_table_row_index=None,
    *,
    maximum_score_elements=8_000_000,
):
    """Return int32 token positions with fixed tie handling and canonical ordering.

    ``group_lengths`` is a local count; ``row_starts`` locates that local range
    within concatenated score columns. Page-table columns index logical tokens,
    not pages. No metadata is ignored and invalid requests are rejected.

    Validation synchronizes CUDA tensors and stable full sorting is expensive.
    This function is intended to identify causes, before implementing a fast path.
    """
    if (logits.ndim != 2 or logits.dtype != torch.float32
            or type(pool_size) is not int or pool_size <= 0
            or type(topk) is not int or topk <= 0 or topk % pool_size
            or type(maximum_score_elements) is not int or maximum_score_elements < 1
            or logits.numel() > maximum_score_elements):
        raise ValueError("Require bounded float32 pooled scores and a valid group budget.")
    rows, columns = logits.shape
    device = logits.device
    if out_rows is None:
        out_rows = rows
    if type(out_rows) is not int or out_rows < rows:
        raise ValueError("Padded output cannot discard query rows.")
    output_width = topk + (pool_size - 1 if seq_lens is not None else 0)
    if (topk > maximum_score_elements or pool_size > maximum_score_elements
            or out_rows * output_width > maximum_score_elements):
        raise ValueError("The reference output exceeds its explicit allocation limit.")
    if page_table is not None and topk_offsets is not None:
        raise ValueError("Paged and ragged output mappings are mutually exclusive.")
    if page_table_row_index is not None and page_table is None:
        raise ValueError("Page-table row mapping needs a page table.")
    lengths = _integer_vector(
        group_lengths, name="group_lengths", rows=rows, device=device)
    starts = (torch.zeros_like(lengths) if row_starts is None else _integer_vector(
        row_starts, name="row_starts", rows=rows, device=device))
    if bool(((lengths < 0) | (starts < 0) | (starts > columns)
             | (lengths > columns - starts)).any()):
        raise ValueError("A query's pooled score range is outside the supplied columns.")
    tail_lengths = torch.zeros_like(lengths)
    if seq_lens is not None:
        seq_lens = _integer_vector(
            seq_lens, name="seq_lens", rows=rows, device=device)
        tail_lengths = seq_lens - lengths * pool_size
        if bool(((tail_lengths < 0) | (tail_lengths >= pool_size)).any()):
            raise ValueError("Sequence length and local pooled count disagree.")

    budget = topk // pool_size
    counts = lengths.clamp(max=budget)
    positions = torch.arange(columns, device=device).expand(rows, -1)
    valid = (positions >= starts[:, None]) & (positions < (starts + lengths)[:, None])
    if bool((torch.isnan(logits) & valid).any()):
        raise ValueError("A valid pooled score is NaN.")
    # Stable score sort retains ascending group id on ties. A second stable
    # validity sort keeps valid -inf scores ahead of invalid padding as well.
    clean = torch.where(valid, logits, torch.zeros_like(logits))
    by_score = torch.argsort(clean, dim=1, descending=True, stable=True)
    ordered_valid = valid.gather(1, by_score).to(torch.int32)
    by_validity = torch.argsort(ordered_valid, dim=1, descending=True, stable=True)
    ordered = by_score.gather(1, by_validity)[:, :min(columns, budget)]
    local = ordered - starts[:, None]
    selected = torch.full((rows, budget), columns + 1, dtype=torch.int64, device=device)
    if local.shape[1]:
        local_valid = torch.arange(local.shape[1], device=device)[None, :] < counts[:, None]
        selected[:, :local.shape[1]] = torch.where(local_valid, local, columns + 1)
    selected = selected.sort(dim=1).values
    group_valid = torch.arange(budget, device=device)[None, :] < counts[:, None]
    token_offsets = torch.arange(pool_size, device=device)
    tokens = (selected[:, :, None] * pool_size + token_offsets).reshape(rows, topk)
    token_valid = group_valid[:, :, None].expand(-1, -1, pool_size).reshape(rows, topk)
    tokens = torch.where(token_valid, tokens, -1)
    if seq_lens is not None and pool_size > 1:
        tokens = torch.cat([
            tokens,
            torch.full((rows, pool_size - 1), -1, dtype=torch.int64, device=device),
        ], dim=1)
        offsets = torch.arange(pool_size - 1, device=device)[None, :].expand(rows, -1)
        tail_positions = counts[:, None] * pool_size + offsets
        tail_values = lengths[:, None] * pool_size + offsets
        tokens.scatter_(1, tail_positions, torch.where(
            offsets < tail_lengths[:, None], tail_values, -1))

    token_valid = tokens >= 0
    if page_table is not None:
        if (page_table.ndim != 2 or page_table.device != device
                or page_table.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Page table must be an integer matrix on the logits device.")
        if page_table_row_index is None:
            if page_table.shape[0] != rows:
                raise ValueError("Page table needs one row per query or explicit row indices.")
            mapped_rows = torch.arange(rows, device=device)
        else:
            mapped_rows = _integer_vector(
                page_table_row_index, name="page_table_row_index", rows=rows, device=device)
        if bool(((mapped_rows < 0) | (mapped_rows >= page_table.shape[0])).any()):
            raise ValueError("A page-table row index is invalid.")
        if bool((token_valid & (tokens >= page_table.shape[1])).any()):
            raise ValueError("A selected logical token is outside the page table.")
        if page_table.shape[1]:
            mapped = page_table[mapped_rows[:, None], tokens.clamp(min=0)]
            tokens = torch.where(token_valid, mapped.to(torch.int64), -1)
    elif topk_offsets is not None:
        offsets = _integer_vector(
            topk_offsets, name="topk_offsets", rows=rows, device=device, allow_column=True)
        if bool((offsets < 0).any()):
            raise ValueError("Ragged offsets must be nonnegative.")
        tokens = torch.where(token_valid, tokens + offsets[:, None], -1)
    if bool((token_valid & ((tokens < 0) | (tokens > torch.iinfo(torch.int32).max))).any()):
        raise ValueError("Mapped token addresses must fit nonnegative int32.")
    result = tokens.to(torch.int32)
    if out_rows > rows:
        result = torch.cat([
            result,
            torch.full((out_rows - rows, result.shape[1]), -1,
                       dtype=result.dtype, device=device),
        ], dim=0)
    return result
