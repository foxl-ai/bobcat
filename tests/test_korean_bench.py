import pytest

from bobcat import korean_bench as kb
from bobcat.student_merge import module_map


def test_haerae_options_accept_list_literals_and_csat_pipes():
    assert kb.haerae_options("['a', 'b', 'c']") == ["a", "b", "c"]
    assert kb.haerae_options(" (A), (B)  |  (C), (D)  |  (A)") == ["(A), (B)", "(C), (D)", "(A)"]


def test_choice_rows_keep_benchmark_options_and_answer():
    row = kb.choice_row("kmmlu", "Accounting", 3, {"문제": "1+1은?"}, "정답을 고르라.",
                        ["1", "2", "3", "4"], 1, ["1+1은?"])
    assert row["candidate_ids"] == ["A", "B", "C", "D"] and row["target"] == "B"
    assert row["request"]["questions"]["q"]["criteria"]["B"] == "2"
    assert row["language"] == "ko" and row["text_keys"]
    with pytest.raises(ValueError):
        kb.choice_row("kmmlu", "x", 0, {"문제": "q"}, "i", ["only"], 0, ["q"])


def test_adapter_modules_map_to_unique_checkpoint_weights():
    keys = ["base_model.model.model.layers.0.mlp.up_proj.lora_A.weight",
            "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight"]
    names = {"model.language_model.layers.0.mlp.up_proj.weight"}
    assert module_map(keys, names) == {
        "model.language_model.layers.0.mlp.up_proj.weight": (keys[0], keys[1])}
    with pytest.raises(ValueError):
        module_map(keys, {"other.weight"})
