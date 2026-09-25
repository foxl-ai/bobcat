"""Native GLM outcome-feedback RL, proper-score control and correctness control.

One forward produces the whole categorical distribution. Four IID actions reuse
it; REINFORCE is applied immediately, without an old-policy forward, a reference
forward, text decoding, a critic or PPO clipping. The retained-judgment replay
is common to every arm and is reported separately from the RL objective.

Method reference: anthony-maio/eve-rlcd, MIT,
57a179b7b1bedc80f65bf42ccda129dd1888272f. This is a Bobcat adaptation, not a
reproduction of Eve's training setup or TypeSafe's proprietary RLCD.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import _same_tree
from bobcat.glm_native_train import tensor_digest
from bobcat.rl_calibration import categorical_brier, outcome_policy_loss

ARMS = ("proper_score_reinforce", "direct_brier", "correctness_reinforce")


def update(experiment, arm):
    e = experiment
    index = e.target_cursor + e.rank
    if index >= len(e.bandit):
        raise ValueError("The disjoint frozen bandit pool is exhausted.")
    e.optimizer.zero_grad(set_to_none=True)
    e.critic_optimizer.zero_grad(set_to_none=True)
    observation = e.bandit.observation(index)
    logits, value = e.forward(observation)
    stats = {"component": e.bandit.provenance(index), "target_component_index": index}
    if arm == "direct_brier":
        # The oracle control receives more information than sampled-feedback RL.
        target = e.bandit.reveal_for_supervised_control(index)
        loss = .5 * categorical_brier(logits, target)
        stats.update(objective="half_brier", full_label_observed_by_learner=True)
    elif arm in ("proper_score_reinforce", "correctness_reinforce"):
        actions = e.torch.multinomial(
            logits.detach().softmax(-1).cpu(), e.args.samples_per_question,
            replacement=True, generator=e.sampling,
        )
        outcomes = e.bandit.step(index, actions)
        result = outcome_policy_loss(
            logits, actions.to(e.device), outcomes.to(e.device),
            calibrated=arm == "proper_score_reinforce",
        )
        loss = result["loss"]
        stats.update(
            objective=result["method"], full_label_observed_by_learner=False,
            sampled_actions=actions.tolist(), sampled_outcomes=outcomes.tolist(),
            sampled_probabilities=result["sampled_probability"].cpu().tolist(),
            sampled_rewards=result["rewards"].cpu().tolist(),
            zero_group_advantage=result["zero_group_advantage"],
            sampled_correctness=float(outcomes.mean()),
        )
        e.transitions += len(actions)
    else:
        raise ValueError("Unknown study arm.")
    stats["primary_loss"] = float(loss.detach())
    loss.backward()
    del logits, value, loss
    if arm != "direct_brier":
        del result
    stats["retention_loss"] = e.replay_backward() if e.args.replay_coefficient else 0.
    stats["actor_gradient_norm"] = e.clip_and_step()
    e.target_cursor += e.world
    e.loop["cursor"] += 1
    return stats


def reset_arm(e, name):
    from bobcat.glm_native_rl import restore_local_parameters

    restore_local_parameters(e.parameters, e.reference)
    e.critic.load_state_dict(e.critic_initial)
    e.new_optimizers()
    e.torch.manual_seed(e.args.seed)
    e.torch.cuda.manual_seed_all(e.args.seed)
    e.sampling.manual_seed(e.args.seed + e.rank)
    e.lane = e.new_lane()
    e.replay_cursor = e.target_cursor = 0
    e.updates = e.transitions = e.critic_steps = 0
    e.loop = {"arm": name, "round": 0, "cursor": 0}
    e.study["arm_seconds"] = 0.
    e.study["stage"] = "training"


def drain(e, reason):
    e.checkpoint(f"{e.loop['arm']}-drain-{e.updates:04d}")
    e.record.update(status="interrupted_checkpointed", native_rl_executed=e.study["proof_done"],
                    study=e.study, interruption_reason=reason,
                    finished_at=datetime.now(UTC).isoformat())
    e.status("interrupted_checkpointed")


def run_study(e):
    if e.args.resume_from:
        # Recipe, original model, parent and local shard hashes are checked by
        # restore_checkpoint. A new execution uses a fresh output directory.
        marker = json.loads((e.args.resume_from / "complete.json").read_text())
        backend = marker["expert_backend"]
        if backend not in ("torch", "torch_mm") or (
                backend == "torch_mm" and e.args.expert_backend != "torch_mm"):
            raise ValueError("The resumed expert backend was not admitted on this worker.")
        for module in e.model.modules():
            if hasattr(module, "use_torch_mm"):
                module.use_torch_mm = backend == "torch_mm"
        e.record["expert_backend"] = backend
        e.restore_checkpoint(e.args.resume_from)
        e.record["resumed_from"] = {
            "checkpoint": str(e.args.resume_from), "producer_job_sha256": marker["job_sha256"],
            "same_recipe": True, "exact_local_state_restored": True,
            "next_update_cross_process_equivalence_verified": False,
        }
    else:
        e.expert_control()
        e.loop = {"arm": "fixed556", "round": 0, "cursor": 0}
        e.evaluation("parent556", expanded=True)
        e.study = {
            "arms": list(ARMS), "arm_index": 0, "stage": "arm_start",
            "primary_updates": None, "arm_seconds": 0., "reports": {}, "proof_done": False,
            "method_reference_revision": "57a179b7b1bedc80f65bf42ccda129dd1888272f",
            "method_is_typesafe_recipe": False, "group_std_normalization": False,
            "reference_kl_coefficient": 0., "critic_used": False,
            "retention_coefficient": e.args.replay_coefficient,
            "evaluation_is_development": True,
        }
    if e.study.get("arms") != list(ARMS):
        raise ValueError("The continuation study or its treatment order changed.")

    while e.study["arm_index"] < len(ARMS):
        arm = ARMS[e.study["arm_index"]]
        if e.should_stop(reserve=e.args.final_reserve_seconds):
            return drain(e, "allocation_or_signal_limit")
        if e.study["stage"] == "arm_start":
            reset_arm(e, arm)
            e.checkpoint(f"{arm}-start")
        if e.study["stage"] == "training":
            target = e.study["primary_updates"] or e.args.max_bandit_updates
            while e.updates < target:
                if e.interruption_requested():
                    return drain(e, "spot_or_external_signal")
                if e.should_stop(reserve=e.args.final_reserve_seconds):
                    return drain(e, "native_execution_limit")
                if arm == ARMS[0] and e.global_flag(
                        e.study["arm_seconds"] >= e.args.primary_arm_seconds):
                    break
                begin = time.monotonic()
                if not e.study["proof_done"]:
                    before = e.snapshots()
                    stats = update(e, arm)
                    expected = e.snapshots()
                    e.restore_checkpoint(e.args.out / f"checkpoint-{arm}-start")
                    update(e, arm)
                    exact = _same_tree(expected, e.snapshots())
                    failed = e.global_flag(not exact)
                    atomic_json(e.args.out / f"rl-update-replay-rank-{e.rank}.json", {
                        "same_process_next_reinforce_update_exact": exact,
                        "global_passed": not failed,
                        "actor_optimizer_rng_sampled_feedback_and_cursors_checked": True,
                        "cross_process_resume_verified": False,
                        "repeated_diagnostic_update_counts_as_compute_not_unique_training": True,
                    })
                    if failed:
                        e.restore(before)
                        raise ValueError("The first native REINFORCE update failed exact replay.")
                    del before, expected
                    e.study["proof_done"] = True
                else:
                    stats = update(e, arm)
                seconds = time.monotonic() - begin
                e.study["arm_seconds"] += seconds
                e.log_update(stats, seconds)
                if e.updates % e.args.checkpoint_every == 0:
                    before_save = time.monotonic()
                    e.checkpoint(f"{arm}-u{e.updates:04d}")
                    e.study["arm_seconds"] += time.monotonic() - before_save
            if not e.updates:
                raise ValueError("The primary RL arm completed no optimizer update.")
            if arm == ARMS[0]:
                e.study["primary_updates"] = e.updates
            e.study["reports"][arm] = {
                "actor_updates": e.updates, "global_target_components": e.target_cursor,
                "global_retention_components": e.replay_cursor,
                "local_sampled_outcomes": e.transitions,
                "training_seconds_including_checkpoints_and_first_update_proof":
                    e.study["arm_seconds"],
                "matched_parent_rows_updates_and_retention":
                    e.updates == e.study["primary_updates"],
                "full_label_observed_by_primary_learner": arm == "direct_brier",
                "rows_drawn_from_parent_training_curriculum": True,
                "incremental_new_gold_annotations": 0,
            }
            e.study["stage"] = "evaluation"
            e.checkpoint(f"{arm}-pre-evaluation")
        if e.study["stage"] == "evaluation":
            e.evaluation(f"{arm}-final", expanded=True)
            e.study["arm_index"] += 1
            e.study["stage"] = "arm_start"
            e.checkpoint(f"{arm}-final")
    e.status("verifying_frozen_original_state")
    after = {name: tensor_digest(value) for name, value in e.model.state_dict().items()
             if "lora_" not in name}
    if e.global_flag(after != e.frozen):
        raise ValueError("The original pretrained body or vocabulary head changed.")
    e.record.update(
        status="completed", frozen_original_state_preserved=True,
        native_rl_executed=e.study["proof_done"], study=e.study,
        finished_at=datetime.now(UTC).isoformat(),
    )
    e.status("completed")
