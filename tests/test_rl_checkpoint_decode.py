import copy

import pytest
import torch

from bobcat.rl_checkpoint_decode import compare_local_actor


def fixture():
    names = ("model.layers.0.self_attn.q_proj.lora_A.weight",
             "model.layers.0.self_attn.q_proj.lora_B.weight")
    full = {name: (torch.arange(128).reshape(16, 8) + 10 * index).to(torch.bfloat16)
            for index, name in enumerate(names)}
    loop = {"arm": "proper_score_reinforce", "cursor": 16, "round": 0}
    states = []
    for rank in range(8):
        actor = {name.replace(".layers.0.", ".layers.0._checkpoint_wrapped_module."):
                 value.chunk(8, 0)[rank].clone() for name, value in full.items()}
        states.append({"updates": 16, "loop": loop.copy(), "actor": actor,
                       "optimizer": {"state": {i: {"step": torch.tensor(16.)}
                                               for i in range(len(full))}}})
    return full, states, loop


def test_independent_reconstruction_rejects_missing_or_reordered_shards():
    full, states, loop = fixture()
    proof = compare_local_actor(full, states, updates=16, loop=loop)
    assert proof["all_eight_local_actor_shards_match_dcp"]
    assert not proof["new_process_gpu_next_update_verified"]
    with pytest.raises(ValueError, match="all eight"):
        compare_local_actor(full, states[:-1], updates=16, loop=loop)
    states[0], states[1] = states[1], states[0]
    with pytest.raises(ValueError, match="shard differs"):
        compare_local_actor(full, states, updates=16, loop=loop)


def test_independent_reconstruction_rejects_counter_and_key_changes():
    full, states, loop = fixture()
    bad = copy.deepcopy(states)
    bad[3]["optimizer"]["state"][0]["step"] -= 1
    with pytest.raises(ValueError, match="optimizer steps"):
        compare_local_actor(full, bad, updates=16, loop=loop)
    bad = copy.deepcopy(states)
    name = next(iter(bad[3]["actor"]))
    bad[3]["actor"][name.replace("q_proj", "other_proj")] = bad[3]["actor"].pop(name)
    with pytest.raises(ValueError, match="actor key"):
        compare_local_actor(full, bad, updates=16, loop=loop)
    bad = copy.deepcopy(states)
    bad[2]["loop"]["cursor"] = 15
    with pytest.raises(ValueError, match="counter"):
        compare_local_actor(full, bad, updates=16, loop=loop)
