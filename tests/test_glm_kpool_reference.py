import pytest
import torch

from bobcat.glm_kpool_kernel_probe import check_selection
from bobcat.glm_kpool_reference import pooled_topk_reference


def integers(values):
    return torch.tensor(values, dtype=torch.int32)


def test_stable_ties_and_canonical_order_preserve_complete_groups_and_tail():
    scores = torch.tensor([[2., 5., 5., 5.], [9., 1., 3., 7.]])
    lengths, seq = integers([4, 2]), integers([9, 5])
    out = pooled_topk_reference(scores, lengths, 2, 4, seq_lens=seq)
    assert out.tolist() == [[2, 3, 4, 5, 8], [0, 1, 2, 3, 4]]
    assert check_selection(
        scores.tolist(), lengths.tolist(), seq.tolist(), out.tolist(), pool_size=2, topk=4)
    for _ in range(4):
        assert torch.equal(out, pooled_topk_reference(scores, lengths, 2, 4, seq_lens=seq))


def test_ragged_score_ranges_ignore_other_requests_and_preserve_output_offsets():
    scores = torch.tensor([[99., 1., 8., 8., 99.], [3., 2., float("nan"), 99., 99.]])
    out = pooled_topk_reference(
        scores, integers([3, 2]), 2, 4, row_starts=integers([1, 0]),
        seq_lens=integers([7, 4]), topk_offsets=integers([[100], [200]]), out_rows=3)
    assert out.tolist() == [
        [102, 103, 104, 105, 106], [200, 201, 202, 203, -1], [-1] * 5,
    ]


def test_paged_mapping_uses_logical_token_columns_and_explicit_request_rows():
    table = torch.tensor([[40, 41, 42, 43, 44], [90, 91, 92, 93, 94]], dtype=torch.int64)
    out = pooled_topk_reference(
        torch.tensor([[1., 2.], [3., 1.]]), integers([2, 1]), 2, 2,
        seq_lens=integers([5, 3]), page_table=table, page_table_row_index=integers([1, 0]))
    assert out.tolist() == [[92, 93, 94], [40, 41, 42]]


def test_logical_order_does_not_sort_physical_cache_addresses():
    out = pooled_topk_reference(
        torch.tensor([[1., 2.]]), integers([2]), 2, 2,
        seq_lens=integers([5]), page_table=integers([[50, 2, 30, 1, 60]]))
    assert out.tolist() == [[30, 1, 60]]


def test_negative_infinity_is_valid_and_padding_cannot_steal_its_slot():
    out = pooled_topk_reference(
        torch.tensor([[float("nan"), -float("inf"), -float("inf"), 99.]]),
        integers([2]), 2, 4, row_starts=integers([1]), seq_lens=integers([4]))
    assert out.tolist() == [[0, 1, 2, 3, -1]]


def test_empty_history_and_tail_are_valid_even_when_there_are_no_score_columns():
    out = pooled_topk_reference(
        torch.empty((2, 0), dtype=torch.float32), integers([0, 0]), 4, 8,
        seq_lens=integers([0, 3]))
    assert out.tolist() == [[-1] * 11, [0, 1, 2] + [-1] * 8]
    empty = pooled_topk_reference(torch.empty((0, 0)), integers([]), 4, 8)
    assert empty.shape == (0, 8)


def test_batch_shape_does_not_change_fixed_input_selection_at_real_glm_budget():
    generator = torch.Generator().manual_seed(19)
    scores = torch.stack([torch.randperm(600, generator=generator) // 16 for _ in range(4)])
    scores = scores.float()
    lengths, seq = integers([127, 511, 519, 600]), integers([508, 2045, 2078, 2403])
    combined = pooled_topk_reference(scores, lengths, 4, 2048, seq_lens=seq)
    singles = torch.cat([
        pooled_topk_reference(scores[i:i + 1], lengths[i:i + 1], 4, 2048,
                             seq_lens=seq[i:i + 1]) for i in range(4)
    ])
    assert torch.equal(combined, singles)
    assert check_selection(scores.tolist(), lengths.tolist(), seq.tolist(), combined.tolist())


@pytest.mark.parametrize("kwargs", [
    {"row_starts": integers([2])},
    {"seq_lens": integers([9])},
    {"page_table": integers([[1, 2]])},
    {"page_table_row_index": integers([0])},
    {"topk_offsets": integers([-1])},
    {"topk_offsets": torch.tensor([2**31], dtype=torch.int64)},
    {"out_rows": 0},
    {"maximum_score_elements": 1},
])
def test_invalid_metadata_cannot_be_silently_dropped(kwargs):
    with pytest.raises(ValueError):
        pooled_topk_reference(torch.tensor([[1., 2.]]), integers([2]), 2, 4, **kwargs)


def test_valid_nan_and_conflicting_address_spaces_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        pooled_topk_reference(torch.tensor([[float("nan")]]), integers([1]), 2, 2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        pooled_topk_reference(torch.tensor([[1.]]), integers([1]), 2, 2,
                             page_table=integers([[0, 1]]), topk_offsets=integers([0]))


def test_output_padding_and_group_budget_are_bounded_before_allocation():
    with pytest.raises(ValueError, match="allocation"):
        pooled_topk_reference(torch.tensor([[1.]]), integers([1]), 2, 2, out_rows=10**12)
    with pytest.raises(ValueError, match="allocation"):
        pooled_topk_reference(torch.empty((0, 0)), integers([]), 2, 10**12)
