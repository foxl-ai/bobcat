import json

import pytest
from test_decision_train import decision_run  # noqa: F401
from test_kobest_eval import fixture as korean_fixture
from test_public_eval import data_fixture

from bobcat.kobest_eval import freeze as freeze_korean
from bobcat.public_eval import freeze
from bobcat.schema import file_hash
from bobcat.student_eval import run


def test_verified_checkpoint_reaches_public_judgment_evaluation(decision_run, tmp_path):  # noqa: F811
    data = data_fixture(tmp_path / "public", ["dev_public"])
    suite = tmp_path / "suite.json"
    freeze(data, suite, per_task=1)
    result = run(
        decision_run.initial_mlm, decision_run.tokenizer, suite, tmp_path / "evaluated",
        device="cpu", max_seconds=120,
    )
    assert result["status"] == "completed"
    assert result["checkpoint_format"] == "bobcat-real-mlm-v1"
    assert result["gpu_count_used"] == 0 and not result["release_gate_passed"]
    provenance = result["scorer_provenance"]
    assert provenance["checkpoint_sha256"] == file_hash(decision_run.initial_mlm)
    assert provenance["tokenizer_sha256"] == file_hash(decision_run.tokenizer)
    assert provenance["training_counters"]["tokens"] > 0
    public = json.loads((tmp_path / "evaluated/public/summary.json").read_text())
    assert public["hard_attempted"] == 2 and public["ordinal_mean_attempted"] == 1
    assert public["valid_response_metrics"]["ordinal_mean_metrics"]["categorical_nll"] is None


@pytest.mark.parametrize("decision_run", [{"max_schema_tokens": 512}], indirect=True)
def test_extra_korean_suite_reaches_the_same_verified_checkpoint(decision_run, tmp_path):  # noqa: F811
    data = data_fixture(tmp_path / "public", ["dev_public"])
    suite = tmp_path / "suite.json"
    freeze(data, suite, per_task=1)
    source, raw = korean_fixture(tmp_path / "korean")
    external = tmp_path / "korean-suite.json"
    freeze_korean(source, raw, external, per_task=1)
    result = run(
        decision_run.initial_mlm, decision_run.tokenizer, suite, tmp_path / "evaluated",
        device="cpu", max_seconds=120, external_suite=external,
    )
    assert result["status"] == "completed"
    assert result["evaluations"]["korean_external"]["attempted_questions"] == 5
    assert result["evaluations"]["korean_external"]["final_evaluation"] is False
    summary = json.loads((tmp_path / "evaluated/korean_external/summary.json").read_text())
    assert set(summary["valid_metrics_by_task"]) == {
        "kobest_boolq", "kobest_copa", "kobest_wic", "kobest_hellaswag", "kobest_sentineg",
    }


def test_corrupt_parent_is_rejected_before_evaluation(decision_run, tmp_path):  # noqa: F811
    with decision_run.initial_mlm.open("ab") as stream:
        stream.write(b"corrupt")
    out = tmp_path / "bad-evaluation"
    with pytest.raises(ValueError, match="checksum"):
        run(decision_run.initial_mlm, decision_run.tokenizer, tmp_path / "unused-suite", out,
            device="cpu")
    assert not out.exists()
