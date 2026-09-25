import numpy as np
import pytest

from bobcat.glm_rotation_study import aligned_logmean


def test_logmean_preserves_labels_not_slots_and_uses_geometric_probabilities():
    a = [.9, .1]
    b = [.4, .6]  # reverse label order: canonical probability is [.6, .4].
    result = aligned_logmean([np.log(a), np.log(b)], [["가", "나"], ["나", "가"]], ["가", "나"])
    expected = np.sqrt(np.array([.9, .1]) * np.array([.6, .4]))
    expected /= expected.sum()
    np.testing.assert_allclose(np.exp(result), expected)
    assert not np.allclose(np.exp(result), [.75, .25])
    shifted = aligned_logmean([np.log(a) + 400, np.log(b) - 200],
                             [["가", "나"], ["나", "가"]], ["가", "나"])
    np.testing.assert_allclose(result, shifted, atol=1e-12)


@pytest.mark.parametrize("labels,scores", [
    (["가", "가"], [1., 0.]), (["가"], [1.]), (["가", "다"], [1., 0.]),
    (["가", "나"], [1., float("nan")]),
])
def test_missing_duplicate_or_nonfinite_choices_cannot_be_filled_in(labels, scores):
    with pytest.raises(ValueError, match="option"):
        aligned_logmean([scores], [labels], ["가", "나"])
