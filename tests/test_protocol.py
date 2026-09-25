import math

import pytest

from bobcat.protocol import (
    answer,
    choice_confidence,
    decode_request,
    parse_request,
    probabilities,
    response,
    score_confidence,
)


def request(questions, state="중복 결제만 환급해 주세요."):
    return {"model": "bobcat-latest", "state": state, "questions": questions}


def test_public_adapter_fixture_is_not_entropy_or_probability_of_correctness():
    assert choice_confidence([0.8, 0.1, 0.1]) == pytest.approx(0.7)
    assert score_confidence([0.1, 0.3, 0.6]) == pytest.approx(0.25)
    # Score confidence depends on ordinal geometry, even at the same p_max.
    assert score_confidence([0.1, 0.8, 0.1]) == pytest.approx(0.7)
    assert score_confidence([0.8, 0.1, 0.1]) == pytest.approx(0.55)
    # SDK adapter normalizes inputs, including its all-zero convention.
    assert choice_confidence([8, 1, 1]) == pytest.approx(0.7)
    assert choice_confidence([0, 0, 0]) == pytest.approx(0)
    assert score_confidence([0, 0, 0]) == pytest.approx(0)


@pytest.mark.parametrize("count", [1, 2, 3, 10, 255])
def test_confidence_extremes_and_singleton_scores(count):
    assert choice_confidence([1.0] + [0.0] * (count - 1)) == pytest.approx(1)
    assert score_confidence([1.0] + [0.0] * (count - 1)) == pytest.approx(1)
    if count > 1:
        assert choice_confidence([1.0] * count) == pytest.approx(0)
    assert probabilities([1000.0] * count) == pytest.approx([1 / count] * count)


def test_korean_mixed_primitives_preserve_structured_rubrics_and_noul_criteria():
    source = request({
        "담당": {
            "type": "choice",
            "instructions": {"판단": "담당 부서", "제약": ["중복 결제"]},
            "criteria": {"결제": {"대상": ["환급", "청구"]}, "배송": None},
        },
        "동의": {
            "type": "noul",
            "criteria": {"true": "명시적 취소 동의", "false": ["취소 금지", "동의 없음"]},
        },
        "긴급": {"type": "score", "criteria": ["보통", {"시점": "오늘"}, ["지금"]]},
    })
    state, questions = parse_request(source)
    assert state == source["state"]
    assert "결제" in questions[0].descriptions[0]
    assert "환급" in questions[0].descriptions[0]
    assert "명시적 취소 동의" in questions[1].descriptions[1]
    assert "동의 없음" in questions[1].descriptions[0]
    assert questions[1].instructions is None
    result = response(
        "bobcat-fixture", questions,
        [[math.log(0.8), math.log(0.2)], [math.log(0.9), math.log(0.1)],
         [math.log(0.1), math.log(0.3), math.log(0.6)]],
        {}, input_tokens=120,
    )
    assert result["model"] == "bobcat-fixture"
    assert result["answers"]["담당"]["choice"] == "결제"
    assert set(result["answers"]["담당"]["probabilities"]) == {"결제", "배송"}
    assert result["answers"]["동의"] == {"type": "noul", "noul": pytest.approx(0.1)}
    assert result["answers"]["긴급"]["score"] == pytest.approx(1.5)
    assert result["answers"]["긴급"]["legend"] == {
        "0": "보통", "1": {"시점": "오늘"}, "2": ["지금"],
    }
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 0}


def test_sdk_optional_content_and_singletons():
    _, questions = parse_request(request({
        "": {"type": "choice", "criteria": {"계속": None}},
        "constant": {"type": "score", "instructions": None, "criteria": ["유일한 단계"]},
        "boolean": {"type": "noul", "criteria": {"true": None}},
    }, state=[]))
    result = response("bobcat-fixture", questions, [[7], [-1000], [0, 0]], {})
    assert result["answers"][""]["probabilities"] == {"계속": 1.0}
    assert result["answers"]["constant"]["score"] == 0
    assert result["answers"]["constant"]["confidence"] == 1
    assert result["answers"]["boolean"] == {"type": "noul", "noul": 0.5}


def test_all_255_choice_descriptions_survive_and_256_is_explicit_error():
    criteria = {f"팀{i}": f"구별되는 업무 설명 {i}" for i in range(255)}
    payload = request({"large": {"type": "choice", "criteria": criteria}})
    _, questions = parse_request(payload)
    assert questions[0].labels == tuple(criteria)
    assert "업무 설명 254" in questions[0].descriptions[-1]
    result = answer(questions[0], [0] * 254 + [1])
    assert result["choice"] == "팀254"
    assert len(result["probabilities"]) == 255
    criteria["팀255"] = "이 후보를 조용히 버리면 안 됨"
    with pytest.raises(ValueError, match="255"):
        parse_request(payload)


@pytest.mark.parametrize("spec", [
    {"type": "choice", "criteria": {}},
    {"type": "score", "criteria": []},
    {"type": "score", "criteria": ["x"] * 11},
    {"type": "noul", "criteria": {"maybe": "unknown"}},
    {"type": "noul", "criteria": {"true": float("nan")}},
    {"type": "noul", "instructions": {"bad": float("inf")}},
    {"type": "noul", "instructions": 3},
    {"type": "noul", "another_question": "do not leak this"},
])
def test_invalid_primitive_data_is_rejected(spec):
    with pytest.raises(ValueError):
        parse_request(request({"q": spec}))


@pytest.mark.parametrize("raw", [
    b'{"state":"a","state":"b"}',
    b'{"questions":{"q":{"criteria":{"x":1,"x":2}}}}',
    b'{"state": NaN}',
    b'{"state": Infinity}',
    b'[]',
])
def test_json_is_unambiguous_and_finite(raw):
    with pytest.raises(ValueError):
        decode_request(raw)


def test_backend_vectors_cannot_be_repaired_into_fake_valid_answers():
    _, questions = parse_request(request({"q": {"type": "choice", "criteria": {"a": None}}}))
    for invalid in ([0.7], [float("nan")], [1, 0]):
        with pytest.raises(ValueError):
            answer(questions[0], invalid)
    with pytest.raises(ValueError, match="finite"):
        response("broken", questions, [[float("nan")]], {})
    with pytest.raises(ValueError, match="integer"):
        response("broken", questions, [[0]], {}, input_tokens=None)
