import pytest

from bobcat.glm_kpool_kernel_probe import check_selection


def test_complete_selected_groups_and_tail():
    assert check_selection(
        [[1., 4., 3.]], [3], [7], [[2, 3, 4, 5, 6]], pool_size=2, topk=4)
    assert check_selection(
        [[5., 1., 3.]], [1], [3], [[0, 1, -1, -1, 2]], pool_size=2, topk=4)


def test_ties_allow_different_optimal_sets():
    for row in ([0, 1, 2, 3, -1], [2, 3, 4, 5, -1]):
        assert check_selection([[1., 1., 1.]], [3], [6], [row], pool_size=2, topk=4)


@pytest.mark.parametrize("row", [
    [0, 1, 2, 3, 6],  # Suboptimal group.
    [2, 3, 4, 4, 6],  # Duplicated position.
    [2, 3, 4, 5, -1],  # Missing unpooled tail.
    [2, 3, 4, 5, 7],  # Position outside the sequence.
])
def test_invalid_selection_is_not_treated_as_nondeterminism(row):
    with pytest.raises(ValueError):
        check_selection([[1., 4., 3.]], [3], [7], [row], pool_size=2, topk=4)
