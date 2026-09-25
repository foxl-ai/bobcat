import pytest

from bobcat.glm_curriculum import select_components


def test_component_sampling_tracks_tokens_not_row_count():
    pools = {
        task: {f"{task}-{i}": [
            {"group_id": f"{task}-{i}", "input_tokens": length, "observation": j}
            for j in range(3)
        ] for i in range(100)}
        for task, length in (("short", 10), ("long", 40))
    }
    rows = select_components(pools, 100, 7, {"short": .5, "long": .5})
    assert len({r["group_id"] for r in rows}) == 100
    short = sum(r["input_tokens"] for r in rows if r["group_id"].startswith("short"))
    long = sum(r["input_tokens"] for r in rows if r["group_id"].startswith("long"))
    assert abs(short - long) <= 40
    assert rows == select_components(pools, 100, 7, {"short": .5, "long": .5})


def test_shared_component_cannot_be_sampled_twice_across_tasks():
    pools = {task: {"same": [{"group_id": "same", "input_tokens": 10}]}
             for task in ("a", "b")}
    with pytest.raises(ValueError, match="exhausted"):
        select_components(pools, 2, 1, {"a": .5, "b": .5})
