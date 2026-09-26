from scripts.quant_drift import compare


def test_drift_is_paired_on_rows_outside_the_calibration_set():
    dev = {f"r{i}": {"candidate_ids": ["a", "b"], "target": "a", "task": "t" + str(i % 2),
                     "language": "ko" if i < 4 else "en", "group_id": f"g{i // 2}"}
           for i in range(8)}
    right, wrong = [2.0, 0.0], [0.0, 2.0]
    base = {f"r{i}": right for i in range(8)}
    other = {**base, "r1": wrong, "r5": wrong}
    result = compare(dev, {"fp8": base, "nvfp4": other}, "fp8", {"r5"}, 1.0)
    assert result["rows"] == 7 and result["excluded"] == 1
    nvfp4 = result["sets"]["nvfp4"]
    assert nvfp4["accuracy"] == 6 / 7
    assert nvfp4["argmax_agreement_with_base"] == 6 / 7
    assert abs(nvfp4["minus_base"] + 1 / 7) < 1e-12
    low, high = nvfp4["minus_base_95ci"]
    assert low <= -1 / 7 <= high <= 0
    assert result["sets"]["fp8"]["language_accuracy"] == {"en": 1.0, "ko": 1.0}


def test_near_ties_and_pairs_are_scored_on_the_reference_margin():
    from scripts.quant_drift import agreement, compare_pair

    dev = {f"r{i}": {"candidate_ids": ["a", "b"], "target": "a", "task": "t",
                     "language": "ko", "group_id": f"g{i}"} for i in range(6)}
    reference = {"r0": [0.05, 0.0], "r1": [0.0, 0.05], "r2": [3.0, 0.0], "r3": [3.0, 0.0],
                 "r4": [0.1, 0.0], "r5": [-3.0, 0.0]}
    other = {**reference, "r0": [0.0, 0.05], "r3": [2.5, 0.0]}  # one near-tie flip
    rows = sorted(dev)
    result = agreement(reference, other, rows, 1.0, 0.1)
    assert result["near_tie_rows"] == 3            # r0, r1, r4 have margins below 0.1
    assert result["near_tie_agreement"] == 2 / 3
    assert result["argmax_agreement"] == 5 / 6 and result["disagreements"] == 1
    pair = compare_pair(dev, other, reference, {"r5"}, 1.0, 0.1)
    assert pair["rows"] == 5 and pair["reference_accuracy"] == 4 / 5
    assert pair["accuracy"] == 3 / 5
