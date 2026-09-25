import copy
import json

import pytest

from bobcat.glm_serving_checkpoint import CompiledReadout, distribution_comparison


class Reply:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        pass

    def json(self):
        return self.value


class Client:
    def __init__(self):
        self.payloads = []
        self.missing = False

    def get(self, path):
        assert path == "/get_model_info"
        return Reply({"model_path": "/models/bobcat"})

    def post(self, path, *, json):
        assert path == "/generate"
        self.payloads.append(copy.deepcopy(json))
        return Reply([
            {"text": "", "output_ids": [], "meta_info": {
                "prompt_tokens": len(ids), "completion_tokens": 0, "cached_tokens": 0,
                "output_token_ids_logprobs": [[[-float(i), token, None]
                    for i, token in enumerate(options[:-1] if self.missing else options)]],
            }}
            for ids, options in zip(json["input_ids"], json["token_ids_logprob"], strict=True)
        ])


def row():
    return {
        "id": "private-evaluation-id", "group_id": "private-group",
        "input_sha256": "input-proof", "kind": "choice", "task": "routing",
        "language": "ko", "language_origin": "native", "supervision": "hard_label",
        "target_index": 1, "score_mean": None, "input_tokens": 3,
        "input_ids": [111, 112, 113], "option_token_ids": [32, 33, 34],
    }


def test_direct_request_never_sends_labels_or_example_metadata():
    client = Client()
    scorer = CompiledReadout(client, model_path="/models/bobcat", run_id="private-run")
    values, measured = scorer.read([row()], label="test")
    wire = client.payloads[0]
    assert wire["input_ids"] == [[111, 112, 113]]
    assert wire["token_ids_logprob"] == [[32, 33, 34]]
    assert wire["sampling_params"]["max_new_tokens"] == 0
    assert "target_index" not in json.dumps(wire)
    assert "private-evaluation-id" not in json.dumps(wire)
    assert values[0]["logits"] == [0., -1., -2.]
    assert values[0]["target_index"] == 1
    assert measured["native_completion_tokens"] == 0
    assert distribution_comparison(values, values)["passed"]


def test_missing_candidate_is_an_error_not_a_filled_probability():
    client = Client()
    client.missing = True
    scorer = CompiledReadout(client, model_path="/models/bobcat", run_id="test")
    with pytest.raises(ValueError, match="omitted requested options"):
        scorer.read([row()], label="test")
    assert len(client.payloads) == 1


def test_distribution_change_does_not_pass_just_because_argmax_is_the_same():
    a = [{**row(), "logits": [1., 0., 0.]}]
    b = [{**row(), "logits": [2., 0., 0.]}]
    result = distribution_comparison(a, b)
    assert result["argmax_changes"] == 0
    assert result["maximum_probability_tv"] > .001
    assert not result["passed"]
