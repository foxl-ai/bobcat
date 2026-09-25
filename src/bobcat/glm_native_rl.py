"""Finite native GLM actor/critic experiment over real sandbox transitions.

The original frozen GLM body and vocabulary readout stay intact. Attention LoRA
is the actor; a separate scalar critic reads the detached last hidden state.
Neither free-form generation nor a hosted Jev teacher is used. Workflow reward
is not a calibrated event probability, and this pilot is not a release result.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import _same_tree, verify_source
from bobcat.glm_fsdp_adapter_probe import expected_local, local_copy
from bobcat.glm_native_data import single_rank_batch
from bobcat.glm_native_train import (
    REVISION,
    decision_loss,
    read_curriculum,
    read_monitoring_suite,
    tensor_digest,
)
from bobcat.rl_collection import (
    WorkflowLane,
    compile_observation,
    validate_world_splits,
    visible_teacher_actions,
)
from bobcat.rl_workflow import EvidenceWorkflow, make_world
from bobcat.schema import file_hash


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def restore_local_parameters(parameters, snapshot):
    import torch

    if set(parameters) != set(snapshot):
        raise ValueError("The actor parameter ownership changed.")
    with torch.no_grad():
        for name, parameter in parameters.items():
            target, value = local_tensor(parameter), snapshot[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError("An actor shard changed shape or precision.")
            target.copy_(value.to(target.device))
    if any(not torch.equal(local_tensor(p).detach().cpu(), snapshot[n])
           for n, p in parameters.items()):
        raise ValueError("Actor shard restoration is not exact.")


def restore_local_optimizer(optimizer, snapshot):
    """Rebuild DTensor moment ownership before restoring local optimizer values.

    A CPU snapshot intentionally stores ordinary local tensors. Feeding those
    directly to AdamW would mix Tensor moments and DTensor parameters. Scalar
    step counters retain their original CPU placement.
    """
    import copy

    from torch.distributed.tensor import DTensor

    state = copy.deepcopy(snapshot)
    if len(state["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Optimizer parameter groups changed.")
    for saved_group, group in zip(state["param_groups"], optimizer.param_groups, strict=True):
        if len(saved_group["params"]) != len(group["params"]):
            raise ValueError("Optimizer parameter ownership changed.")
        for identifier, parameter in zip(saved_group["params"], group["params"], strict=True):
            for key, value in list(state["state"].get(identifier, {}).items()):
                if key not in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    continue
                if value.shape != local_tensor(parameter).shape or value.dtype != parameter.dtype:
                    raise ValueError("Optimizer moment shape or dtype changed.")
                value = value.to(parameter.device)
                if isinstance(parameter, DTensor):
                    value = DTensor.from_local(
                        value, parameter.device_mesh, parameter.placements, run_check=False,
                        shape=parameter.shape, stride=parameter.stride(),
                    )
                state["state"][identifier][key] = value
    optimizer.load_state_dict(state)


def initialize_adam_state(optimizer):
    """Create zero moments without taking a hidden optimizer step during DCP save."""
    import torch

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if optimizer.state[parameter]:
                raise ValueError("Initialize only a fresh AdamW optimizer.")
            optimizer.state[parameter] = {
                "step": torch.tensor(0., dtype=torch.float32, device="cpu"),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }


@contextmanager
def frozen_reference(parameters, reference):
    """Swap only local LoRA values, retaining Parameter identity and optimizer state."""
    current = {name: local_copy(p) for name, p in parameters.items()}
    restore_local_parameters(parameters, reference)
    try:
        yield
    finally:
        restore_local_parameters(parameters, current)


class NativeExperiment:
    def __init__(self, args, record):
        import torch
        import torch.distributed as dist

        from bobcat.glm_native_loader import load_native_model
        from bobcat.glm_readout import GLMCompiler

        self.torch, self.dist = torch, dist
        self.args, self.record = args, record
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if self.world != 8:
            raise ValueError("This original GLM experiment requires eight exclusive ranks.")
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.started = time.monotonic()
        self.stop_requested = False
        self.logical_tokens = self.physical_tokens = self.forwards = 0
        self.updates = self.transitions = 0
        self.critic_steps = 0
        self.gradstats_admitted = None
        self.pending = None
        self.study = {}
        self.loop = {"arm": "initialization", "round": 0, "cursor": 0}
        self.replay_cursor = 0
        self.target_cursor = 0
        self.sampling = torch.Generator(device="cpu").manual_seed(args.seed + self.rank)
        self.curriculum, self.data = read_curriculum(args.curriculum)
        halfway = len(self.data["train"]) // 2
        self.bandit_rows = [
            row for row in self.data["train"][:halfway] if row["supervision"] == "hard_label"
        ]
        self.replay_rows = [
            *self.data["train"][halfway:],
            *(row for row in self.data["train"][:halfway]
              if row["supervision"] == "score_mean"),
        ] if args.rl_mechanism != "workflow" else self.data["train"]
        if {r["group_id"] for r in self.bandit_rows} & {
            r["group_id"] for r in self.replay_rows
        } and args.rl_mechanism != "workflow":
            raise ValueError("The bandit and retention replay pools must be disjoint.")
        from bobcat.rl_bandit import DecisionBandit

        self.bandit = DecisionBandit(self.bandit_rows)
        _, self.monitor = read_monitoring_suite(args.suite, self.data["train"])
        validate_world_splits(args.train_world_start, args.train_worlds,
                              args.eval_world_start, args.eval_worlds)
        self.source = json.loads(args.source.read_text())
        if self.source["revision"] != REVISION:
            raise ValueError("The original GLM source changed.")
        self.compiler = GLMCompiler(args.model_dir, self.source)
        self.model, self.adapters = load_native_model(
            args.model_dir, self.source, args.out, self.status, cpu_offload=False,
            expert_backend="torch", activation_checkpointing=True,
            source_model_set=args.source_model_set, source_verification=args.source_verification,
            parent_adapter_path=args.parent_adapter,
        )
        self.model.train()
        self.parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if (len(self.parameters) != 360
                or sum(p.numel() for p in self.parameters.values()) != 17649664
                or any("lora_" not in n for n in self.parameters)):
            raise ValueError("Only the verified full45 attention adapters may be trained.")
        self.status("loading_parent_actor")
        self.load_parent()
        self.reference = {n: local_copy(p) for n, p in self.parameters.items()}
        self.frozen = {n: tensor_digest(p) for n, p in self.model.state_dict().items()
                       if "lora_" not in n}
        atomic_json(args.out / f"frozen-state-hashes-rank-{self.rank}.json", self.frozen)
        self.hidden = None

        def capture(_module, inputs):
            value = inputs[0]
            if value.ndim != 3 or value.shape[0:2] != (1, 1) or value.shape[-1] != 4096:
                raise ValueError("The critic needs the real last scored GLM hidden state.")
            # Critic gradients do not alter the language adapter in this first pilot.
            self.hidden = value[0, -1].detach().float()

        self.hook = self.model.get_output_embeddings().register_forward_pre_hook(capture)
        self.critic = torch.nn.Sequential(
            torch.nn.LayerNorm(4096, elementwise_affine=False),
            torch.nn.Linear(4096, 1),
        ).to(self.device)
        with torch.no_grad():
            self.critic[1].weight.zero_()
            self.critic[1].bias.zero_()
        self.critic_initial = local_copy(self.critic.state_dict())
        self.new_optimizers()
        self.lane = self.new_lane()
        record.update(
            pretrained_weights_loaded=True, parent_actor_values_exact=True,
            parent_adapter_sha256=args.parent_adapter_sha256,
            parent_complete_sha256=args.parent_complete_sha256,
            curriculum_sha256=file_hash(args.curriculum / "manifest.json"),
            monitoring_suite_sha256=file_hash(args.suite / "manifest.json"),
            actor_trainable_parameters=17649664, critic_trainable_parameters=4097,
            parent_optimizer_continued=False, fresh_optimizer_per_arm=True,
            original_vocabulary_head_retained=True, generated_text_tokens=0,
            policy_probabilities_are_calibrated_event_probabilities=False,
            evaluator_private_fields_enter_model=False, release_quality_passed=False,
            rl_mechanism=args.rl_mechanism,
            rlcd_identity_claimed=False,
            bandit_components=len(self.bandit_rows), retention_components=len(self.replay_rows),
        )

    def status(self, phase, **fields):
        self.record.update(phase=phase, **fields, updated_at=datetime.now(UTC).isoformat(),
                           elapsed_seconds=time.monotonic() - self.started)
        self.record.update(actor_optimizer_updates=self.updates, transitions=self.transitions,
                           critic_optimizer_updates=self.critic_steps,
                           counted_forward_prompt_tokens=self.logical_tokens,
                           counted_forward_padded_tokens=self.physical_tokens,
                           counted_local_forward_calls=self.forwards)
        atomic_json(self.args.out / f"rank-{self.rank}.json", self.record)
        if self.rank == 0:
            print(json.dumps({"phase": phase, **fields,
                              "elapsed_seconds": self.record["elapsed_seconds"]}), flush=True)

    def load_parent(self):
        from safetensors.torch import load_file

        from bobcat.glm_parent_adapter import parent_adapter_bindings

        if (file_hash(self.args.parent_adapter) != self.args.parent_adapter_sha256
                or file_hash(self.args.parent_complete) != self.args.parent_complete_sha256):
            raise ValueError("The independently decoded parent actor changed.")
        marker = json.loads(self.args.parent_complete.read_text())
        if marker["step"] != 556 or marker["source_revision"] != REVISION:
            raise ValueError("This pilot is anchored to completed full45 checkpoint 556.")
        full = load_file(str(self.args.parent_adapter), device="cpu")
        bindings = parent_adapter_bindings(self.model, full)
        if set(bindings) != set(self.parameters):
            raise ValueError("Parent bindings differ from the optimizer's native actor.")
        parts = {n: expected_local(bindings[n], p).clone() for n, p in self.parameters.items()}
        restore_local_parameters(self.parameters, parts)

    def new_optimizers(self):
        self.optimizer = self.torch.optim.AdamW(
            self.parameters.values(), lr=self.args.learning_rate, weight_decay=.01, foreach=False,
        )
        self.critic_optimizer = self.torch.optim.AdamW(
            self.critic.parameters(), lr=self.args.critic_learning_rate,
            weight_decay=0, foreach=False,
        )
        initialize_adam_state(self.optimizer)
        initialize_adam_state(self.critic_optimizer)

    def new_lane(self):
        return WorkflowLane(first_seed=self.args.train_world_start,
                            world_count=self.args.train_worlds,
                            language="ko" if self.rank < 4 else "en",
                            lane=self.rank % 4)

    def agreed_length(self, row):
        length = self.torch.tensor(len(row["input_ids"]), device=self.device)
        self.dist.all_reduce(length, op=self.dist.ReduceOp.MAX)
        return (int(length) + 127) // 128 * 128

    def forward(self, row):
        """Identical train-mode, grad-enabled numerical path for rollout and PPO."""
        padded = self.agreed_length(row)
        if padded > self.args.max_input_tokens:
            raise ValueError("A full branch exceeds the frozen pilot length; never truncate.")
        batch = single_rank_batch({"inputs": {"input_ids": row["input_ids"]}},
                                  padded, device=self.device)
        self.hidden = None
        with self.torch.enable_grad():
            output = self.model(**batch).logits
            if output.shape[:2] != (1, 1) or output.shape[-1] != 154880:
                raise ValueError("The original one-position vocabulary readout changed.")
            logits = output[0, 0].index_select(
                0, self.torch.tensor(row["option_token_ids"], device=self.device),
            ).float()
            if self.hidden is None or not bool(self.torch.isfinite(logits).all()):
                raise ValueError("Missing critic feature or nonfinite original actor scores.")
            value = self.critic(self.hidden).squeeze(-1)
        self.forwards += 1
        self.logical_tokens += len(row["input_ids"])
        self.physical_tokens += padded
        return logits, value

    def sample_action(self, logits):
        probabilities = logits.detach().float().softmax(-1).cpu()
        return int(self.torch.multinomial(probabilities, 1, generator=self.sampling))

    def global_flag(self, flag):
        vote = self.torch.tensor(int(flag), device=self.device)
        self.dist.all_reduce(vote, op=self.dist.ReduceOp.MAX)
        return bool(vote)

    def should_stop(self, reserve=0):
        # A signal or stop file is shared at a safe distributed boundary.
        return self.global_flag(
            self.stop_requested or self.args.stop_file.exists()
            or time.monotonic() - self.started >= self.args.max_seconds - reserve
        )

    def interruption_requested(self):
        return self.global_flag(self.stop_requested or self.args.stop_file.exists())

    def snapshots(self):
        return {
            "actor": {n: local_copy(p) for n, p in self.parameters.items()},
            "optimizer": local_copy(self.optimizer.state_dict()),
            "critic": local_copy(self.critic.state_dict()),
            "critic_optimizer": local_copy(self.critic_optimizer.state_dict()),
            "cpu_rng": self.torch.get_rng_state(),
            "cuda_rng": self.torch.cuda.get_rng_state(self.device),
            "sampling_rng": self.sampling.get_state(), "lane": self.lane.state_dict(),
            "loop": dict(self.loop), "replay_cursor": self.replay_cursor,
            "target_cursor": self.target_cursor,
            "updates": self.updates, "transitions": self.transitions,
            "critic_steps": self.critic_steps, "pending": self.pending,
            "study": copy.deepcopy(self.study),
        }

    def restore(self, state):
        restore_local_parameters(self.parameters, state["actor"])
        restore_local_optimizer(self.optimizer, state["optimizer"])
        self.critic.load_state_dict(state["critic"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.torch.set_rng_state(state["cpu_rng"])
        self.torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        self.sampling.set_state(state["sampling_rng"])
        self.lane.load_state_dict(state["lane"])
        self.loop, self.replay_cursor = dict(state["loop"]), state["replay_cursor"]
        self.target_cursor = state["target_cursor"]
        self.updates, self.transitions = state["updates"], state["transitions"]
        self.critic_steps, self.pending = state["critic_steps"], state["pending"]
        self.study = copy.deepcopy(state["study"])
        if not _same_tree(state, self.snapshots()):
            raise ValueError("Actor/optimizer/critic/RNG/environment restoration is not exact.")

    def checkpoint(self, name):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

        folder = self.args.out / f"checkpoint-{name}"
        if folder.exists():
            raise ValueError("Never overwrite a previous checkpoint.")
        self.status("saving_rl_checkpoint", checkpoint=name)
        before = self.snapshots()
        options = StateDictOptions(ignore_frozen_params=True, cpu_offload=True, strict=False)
        actor, optimizer = get_state_dict(self.model, self.optimizer, options=options)
        dcp.save({"model": actor, "optimizer": optimizer}, checkpoint_id=folder)
        # Local shards duplicate only the small adapter state, allowing an exact
        # same-layout continuation plus portable DCP for independent reconstruction.
        self.torch.save(self.snapshots(), folder / f"state-rank-{self.rank}.pt")
        if not _same_tree(before, self.snapshots()):
            raise ValueError("Checkpoint serialization changed the learning state.")
        self.dist.barrier()
        if self.rank == 0:
            atomic_json(folder / "complete.json", {
                "schema": "bobcat-native-rl-checkpoint-v1", "world_size": 8,
                "source_revision": REVISION, "parent_complete_sha256":
                    self.args.parent_complete_sha256,
                "parent_adapter_sha256": self.args.parent_adapter_sha256,
                "job_sha256": self.args.job_sha256,
                "recipe_sha256": self.args.recipe_sha256,
                "expert_backend": self.record.get("expert_backend", "torch"),
                "loop": dict(self.loop), "actor_optimizer_updates": self.updates,
                "transitions": self.transitions, "critic_optimizer_updates": self.critic_steps,
                "producer_runtime": {"python": list(sys.version_info[:2]),
                                     "torch": self.torch.__version__},
                "files": {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()},
            })
        self.dist.barrier()
        return folder

    def expert_control(self):
        """A full-original-model treatment, separate from the small expert fixture."""
        import statistics

        from bobcat.glm_expert_backend import inspect_grouped_experts

        if self.args.expert_backend != "torch_mm":
            self.record["expert_backend"] = "torch"
            return
        modules = [m for m in self.model.modules() if hasattr(m, "use_torch_mm")]
        inspect_grouped_experts(self.model, "torch")
        saved = self.snapshots()
        row = self.data["train"][self.rank]
        samples, reference, relative_errors = [], None, []
        self.status("full_model_expert_comparison")
        for index, use_grouped in enumerate((False, True, False, True, True, False)):
            self.restore(saved)
            for module in modules:
                module.use_torch_mm = use_grouped
            self.optimizer.zero_grad(set_to_none=True)
            self.torch.cuda.synchronize(self.device)
            started = time.monotonic()
            logits, value = self.forward(row)
            loss = decision_loss(logits, row)
            loss.backward()
            gradients = {n: local_copy(p.grad) for n, p in self.parameters.items()}
            self.clip_and_step()
            elapsed = time.monotonic() - started
            actual = logits.detach().softmax(-1).cpu()
            if not use_grouped and reference is None:
                reference = {"probabilities": actual, "gradients": gradients}
            # Gradient error is pooled over all intended trainable shards; do
            # not let a near-zero tensor dominate a relative error quotient.
            diff = sum(float((g.float() - reference["gradients"][n].float()).square().sum())
                       for n, g in gradients.items())
            energy = sum(float(g.float().square().sum())
                         for g in reference["gradients"].values())
            tv = float((actual - reference["probabilities"]).abs().sum() / 2)
            relative_errors.append((diff / max(energy, 1e-30)) ** .5)
            samples.append({
                "grouped": use_grouped, "warmup": index < 2, "seconds": elapsed,
                "probability_tv": tv,
                "argmax_equal": int(actual.argmax()) == int(reference["probabilities"].argmax()),
                "adapter_gradient_relative_rms": relative_errors[-1],
            })
            del logits, value, loss, gradients
        self.restore(saved)
        reports = [None] * 8
        self.dist.all_gather_object(reports, {"rank": self.rank, "samples": samples})
        slowest = [max(r["samples"][i]["seconds"] for r in reports) for i in range(6)]
        speedup = statistics.median([slowest[2], slowest[5]]) / statistics.median(
            [slowest[3], slowest[4]],
        )
        numerical = all(
            s["probability_tv"] <= .001 and s["argmax_equal"]
            and s["adapter_gradient_relative_rms"] <= .03
            for r in reports for s in r["samples"]
        )
        admitted = numerical and speedup >= 1.05
        for module in modules:
            module.use_torch_mm = admitted
        result = {
            "schema": "bobcat-full-model-expert-admission-v1",
            "activated": admitted, "numerical_gate_passed": numerical,
            "measured_update_speedup": speedup, "reports": reports,
            "maximum_probability_tv": .001, "maximum_gradient_relative_rms": .03,
            "minimum_speedup": 1.05, "same_parent_restored_between_trials": True,
            "timing_includes_forward_backward_gradient_checks_and_optimizer": True,
            "test_input_components": 8, "broad_quality_equivalence_claimed": False,
            "fallback": None if admitted else "original_torch_loop",
        }
        if self.rank == 0:
            atomic_json(self.args.out / "full-model-expert-admission.json", result)
        self.record["expert_backend"] = "torch_mm" if admitted else "torch"

    def restore_checkpoint(self, folder):
        marker = json.loads((folder / "complete.json").read_text())
        if (marker.get("schema") != "bobcat-native-rl-checkpoint-v1"
                or marker["source_revision"] != REVISION
                or marker.get("recipe_sha256") != self.args.recipe_sha256
                or marker["world_size"] != 8
                or marker["parent_adapter_sha256"] != self.args.parent_adapter_sha256):
            raise ValueError("The RL resume checkpoint belongs to another frozen experiment.")
        member = f"state-rank-{self.rank}.pt"
        if file_hash(folder / member) != marker["files"][member]:
            raise ValueError("The saved local continuation state changed.")
        state = self.torch.load(folder / member, map_location="cpu", weights_only=True)
        self.restore(state)

    def clip_and_step(self, *, critic=False):
        from bobcat.gradient_statistics import (
            admit_gradient_statistics,
            compare_gradient_statistics,
            gradient_norm_squared,
        )

        gradients = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    raise ValueError(f"An intended actor gradient is missing: {name}")
                gradients.append(parameter.grad)
            elif parameter.grad is not None:
                raise ValueError("A frozen original parameter received a gradient.")
        if self.gradstats_admitted is None:
            comparison = compare_gradient_statistics(gradients, device=self.device)
            comparison["rank"] = self.rank
            reports = [None] * 8
            self.dist.all_gather_object(reports, comparison)
            decision = admit_gradient_statistics(reports)
            self.gradstats_admitted = decision["activated"]
            if self.rank == 0:
                atomic_json(self.args.out / "gradient-statistics-admission.json", decision)
        norm = self.torch.tensor(
            gradient_norm_squared(gradients, batched=self.gradstats_admitted), device=self.device,
            dtype=self.torch.float64,
        )
        self.dist.all_reduce(norm)
        value = float(norm.sqrt())
        for parameter in self.parameters.values():
            local_tensor(parameter.grad).mul_(min(1., 1. / max(value, 1e-12)))
        if critic:
            for parameter in self.critic.parameters():
                if parameter.grad is None or not bool(self.torch.isfinite(parameter.grad).all()):
                    raise ValueError("Critic gradients are missing or nonfinite.")
                self.dist.all_reduce(parameter.grad)
                parameter.grad.div_(self.world)
            self.torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.,
                                               error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if critic:
            self.critic_optimizer.step()
            self.critic_steps += 1
        self.hidden = None
        self.torch.cuda.synchronize(self.device)
        return value

    def replay_backward(self):
        rows = self.replay_rows
        cursor = self.replay_cursor + self.rank
        if cursor >= len(rows):
            raise ValueError("The frozen judgment replay curriculum is exhausted.")
        row = rows[cursor]
        logits, value = self.forward(row)
        loss = decision_loss(logits, row) * self.args.replay_coefficient
        loss.backward()
        self.replay_cursor += self.world
        result = float(loss.detach())
        del logits, value, loss
        return result

    def collect_bandit(self):
        """On-policy sampled categorical feedback from real, train-only judgments.

        The reward is a detached marginal of a proper score, not +1 accuracy.
        Its full-action expectation equals the direct Brier gradient at the
        behavior policy. Sampling and PPO clipping are the experimental part.
        """
        from bobcat.rl_calibration import marginal_brier_feedback

        self.status("collecting_calibrated_bandit_feedback", round=self.loop["round"])
        rows = []
        for _ in range(self.args.rollout_steps):
            index = self.target_cursor + self.rank
            if index >= len(self.bandit_rows):
                raise ValueError("The frozen categorical bandit curriculum is exhausted.")
            row = self.bandit_rows[index]
            logits, value = self.forward(row)
            old_logs = logits.detach().log_softmax(-1).cpu()
            actions = self.torch.multinomial(
                old_logs.exp(), self.args.samples_per_question, replacement=True,
                generator=self.sampling,
            )
            feedback = marginal_brier_feedback(old_logs, row["target_index"], actions)
            rows.append({
                "mechanism": "marginal_brier_bandit", "row": row,
                "sampled_actions": actions.tolist(), "old_log_probs": old_logs.tolist(),
                "sample_rewards": feedback["rewards"].tolist(),
                "sample_advantages": feedback["advantages"].tolist(),
                "exact_baseline": float(feedback["exact_action_independent_baseline"]),
                "negative_brier_score": float(feedback["negative_brier_score"]),
                "full_information_gold_used": True, "terminal_single_step": True,
                "reward_is_policy_dependent_marginal_score": True,
                "proper_gradient_identity_applies_at_behavior_policy_only": True,
            })
            self.target_cursor += self.world
            self.transitions += len(actions)
            del logits, value
        with frozen_reference(self.parameters, self.reference):
            for row in rows:
                logits, value = self.forward(row["row"])
                row["reference_log_probs"] = logits.detach().log_softmax(-1).cpu().tolist()
                del logits, value
        self.pending = rows
        self.loop["cursor"] = 0
        folder = self.args.out / f"rollout-{self.loop['round']:04d}"
        folder.mkdir(exist_ok=True)
        atomic_json(folder / f"rank-{self.rank}.json", {
            "schema": "bobcat-native-proper-score-bandit-rollout-v1",
            "rank": self.rank, "round": self.loop["round"], "rows": rows,
            "source_components": [row["row"]["group_id"] for row in rows],
            "target_cursor_after": self.target_cursor,
            "retention_replay_components_disjoint": True,
            "calibration_guaranteed": False, "is_typesafe_rlcd_recipe": False,
        })

    def collect(self):
        """One active environment lane per rank, without invalid padding transitions."""
        from bobcat.rl_objective import generalized_advantages

        self.status("collecting_on_policy_rollout", arm="ppo", round=self.loop["round"])
        transitions = []
        for _ in range(self.args.rollout_steps):
            payload = self.lane.observation()
            row = compile_observation(payload, self.compiler)
            logits, value = self.forward(row)
            logs = logits.detach().log_softmax(-1).cpu()
            action_index = self.sample_action(logits)
            old_value = float(value.detach())
            del logits, value
            result = self.lane.advance(row["candidate_ids"][action_index])
            transitions.append({
                "row": row, "payload": payload, "action_index": action_index,
                "old_log_probs": logs.tolist(), "old_value": old_value,
                "reward": result["reward"], "terminated": result["terminated"],
                "truncated": result["truncated"],
                "bootstrap_allowed": not (result["terminated"] or result["truncated"]),
                "world_seed": result["world_seed"],
                "evaluator_info": result["info"], "episode_trace": result["episode_trace"],
            })
            self.transitions += 1
        if transitions[-1]["bootstrap_allowed"]:
            next_row = compile_observation(self.lane.observation(), self.compiler)
            logits, value = self.forward(next_row)
            next_value = float(value.detach())
            del logits, value
            transitions[-1]["truncated"] = True
            transitions[-1]["collector_cutoff"] = True
        else:
            next_value = 0.
        values = [row["old_value"] for row in transitions]
        next_values = [
            values[i + 1] if i + 1 < len(values) else next_value for i in range(len(values))
        ]
        def tensor(values):
            return self.torch.tensor(values, dtype=self.torch.float32)

        def flag(field):
            return self.torch.tensor([row[field] for row in transitions], dtype=self.torch.bool)

        advantages, returns = generalized_advantages(
            tensor([row["reward"] for row in transitions]), tensor(values), tensor(next_values),
            terminated=flag("terminated"), truncated=flag("truncated"),
            bootstrap_allowed=flag("bootstrap_allowed"),
        )
        moments = self.torch.tensor(
            [float(advantages.sum()), float(advantages.square().sum()), len(advantages)],
            device=self.device, dtype=self.torch.float64,
        )
        self.dist.all_reduce(moments)
        mean = moments[0] / moments[2]
        std = (moments[1] / moments[2] - mean.square()).clamp_min(0).sqrt().clamp_min(1e-8)
        normalized = (advantages.double() - float(mean)) / float(std)
        # The reference is checkpoint 556, not a stale rollout or an external API.
        self.status("scoring_frozen_reference", round=self.loop["round"])
        with frozen_reference(self.parameters, self.reference):
            for i, row in enumerate(transitions):
                logits, value = self.forward(row["row"])
                row["reference_log_probs"] = logits.detach().log_softmax(-1).cpu().tolist()
                row["advantage"] = float(normalized[i])
                row["return"] = float(returns[i])
                del logits, value
        folder = self.args.out / f"rollout-{self.loop['round']:04d}"
        folder.mkdir(exist_ok=True)
        atomic_json(folder / f"rank-{self.rank}.json", {
            "schema": "bobcat-native-on-policy-rollout-v1",
            "rank": self.rank, "round": self.loop["round"], "rows": transitions,
            "lane_after": self.lane.state_dict(),
            "old_policy_actor_updates": self.updates,
            "parent_adapter_sha256": self.args.parent_adapter_sha256,
            "policy_seed": self.args.seed, "language": self.lane.settings["language"],
            "calibrated_event_probabilities": False,
        })
        self.pending = transitions
        self.loop["cursor"] = 0

    def ppo_update(self, row, *, check_old=False):
        from bobcat.rl_objective import clipped_actor_critic_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        logits, value = self.forward(row["row"])
        def t(values):
            return self.torch.tensor(values, device=self.device, dtype=self.torch.float32)

        old = t(row["old_log_probs"])
        if check_old:
            difference = float((logits.detach().log_softmax(-1) - old).abs().max())
            failure = self.global_flag(difference > 1e-5)
            atomic_json(self.args.out / f"old-policy-recompute-rank-{self.rank}.json", {
                "maximum_absolute_logprob_difference": difference,
                "allowed_absolute_difference": 1e-5, "global_passed": not failure,
                "same_train_mode_and_grad_enabled_path": True,
            })
            if failure:
                raise ValueError("Old policy logits changed before the first PPO update.")
        bandit = row.get("mechanism") == "marginal_brier_bandit"
        if bandit:
            actions = self.torch.tensor(row["sampled_actions"], device=self.device,
                                        dtype=self.torch.long)
            size = len(actions)
            matrix = logits[None].expand(size, -1)
            zeros = t([0.] * size)
            result = clipped_actor_critic_loss(
                matrix, self.torch.ones_like(matrix, dtype=self.torch.bool), actions,
                old[actions], t(row["sample_advantages"]), zeros, zeros, zeros,
                t(row["reference_log_probs"])[None].expand(size, -1),
                value_coefficient=0.,
                reference_kl_coefficient=self.args.reference_kl_coefficient,
            )
        else:
            index = row["action_index"]
            result = clipped_actor_critic_loss(
                logits[None], self.torch.ones_like(logits[None], dtype=self.torch.bool),
                self.torch.tensor([index], device=self.device, dtype=self.torch.long),
                old[index][None], t([row["advantage"]]), value[None], t([row["old_value"]]),
                t([row["return"]]), t(row["reference_log_probs"])[None],
                reference_kl_coefficient=self.args.reference_kl_coefficient,
            )
        result["loss"].backward()
        stats = {key: float(v.detach()) for key, v in result.items()
                 if self.torch.is_tensor(v)}
        del logits, value, result
        stats["replay_loss"] = self.replay_backward()
        stats["actor_gradient_norm"] = self.clip_and_step(critic=not bandit)
        return stats

    def supervised_update(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.args.rl_mechanism == "calibrated_bandit":
            from bobcat.rl_calibration import categorical_brier

            index = self.target_cursor + self.rank
            if index >= len(self.bandit_rows):
                raise ValueError("The frozen proper-score control pool is exhausted.")
            row = self.bandit_rows[index]
            # Score the identical original reference before constructing the
            # differentiable current forward; never mutate live saved weights.
            with frozen_reference(self.parameters, self.reference):
                reference, value = self.forward(row)
                reference_logs = reference.detach().log_softmax(-1)
                del reference, value
            logits, value = self.forward(row)
            logs = logits.log_softmax(-1)
            kl = (logs.exp() * (logs - reference_logs)).sum()
            loss = categorical_brier(logits, row["target_index"]) \
                + self.args.reference_kl_coefficient * kl
            loss.backward()
            result = {"direct_brier_and_kl_loss": float(loss.detach())}
            del logits, value, loss, logs, reference_logs, kl
            result["replay_loss"] = self.replay_backward()
            result["actor_gradient_norm"] = self.clip_and_step()
            self.target_cursor += self.world
            # These are supervised judgments, not policy transitions.
            result["supervised_target_cursor"] = self.target_cursor
            return result
        payload = self.lane.observation()
        row = compile_observation(payload, self.compiler)
        selected = visible_teacher_actions(payload)
        logits, value = self.forward(row)
        indices = [row["candidate_ids"].index(name) for name in selected]
        # Mean log-likelihood for every equally valid read action avoids teaching
        # an arbitrary alphabetic ordering as if it were the sole correct action.
        loss = -logits.log_softmax(-1)[indices].mean()
        loss.backward()
        result = {"workflow_loss": float(loss.detach())}
        action = selected[int(self.torch.randint(len(selected), (1,), generator=self.sampling))]
        outcome = self.lane.advance(action)
        self.transitions += 1
        del logits, value, loss
        result["replay_loss"] = self.replay_backward()
        result["actor_gradient_norm"] = self.clip_and_step()
        result["reward"] = outcome["reward"]
        return result

    def log_update(self, stats, seconds):
        row = {"at": datetime.now(UTC).isoformat(), **self.loop, **stats,
               "actor_update": self.updates, "critic_update": self.critic_steps,
               "local_transitions": self.transitions, "elapsed_seconds": seconds}
        with (self.args.out / f"updates-rank-{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        self.status("training_" + self.loop["arm"], update=self.updates,
                    round=self.loop["round"], cursor=self.loop["cursor"])

    def evaluation(self, label, *, expanded=False):
        """Fixed bilingual worlds plus the frozen public judgment development suite."""
        before = self.snapshots()
        self.status("evaluating", evaluation=label)
        predictions, episodes = [], []
        monitor = self.monitor if expanded else self.data["dev_train"]
        for start in range(0, len(monitor), 8):
            row = monitor[start + self.rank]
            logits, value = self.forward(row)
            predictions.append({
                **{key: row[key] for key in (
                    "id", "group_id", "language", "task", "kind", "supervision",
                    "target_index", "score_mean", "candidate_ids", "input_sha256",
                ) if key in row},
                "logits": logits.detach().cpu().tolist(),
                "probabilities": logits.detach().softmax(-1).cpu().tolist(),
                "input_tokens": row["input_tokens"],
            })
            del logits, value
        for offset in range(0, self.args.eval_worlds, 4):
            seed = self.args.eval_world_start + offset + self.rank % 4
            env = EvidenceWorkflow(make_world(seed, "ko" if self.rank < 4 else "en"))
            reward, steps, trace = 0., 0, []
            last = None
            placeholder = compile_observation(env.observation(), self.compiler)
            for _ in range(4):
                active = not (env.terminated or env.truncated)
                row = compile_observation(
                    env.observation(), self.compiler,
                ) if active else placeholder
                logits, value = self.forward(row)
                choice = int(logits.detach().argmax())
                del logits, value
                if active:
                    last = env.step(row["candidate_ids"][choice])
                    reward += last["reward"]
                    steps += 1
                    trace.append({"action": row["candidate_ids"][choice],
                                  "observation_sha256": row["observation_sha256"],
                                  "reward": last["reward"]})
            episodes.append({
                "seed": seed, "component_id": env.world.component_id,
                "language": env.world.language, "reward": reward, "steps": steps,
                "success": last["info"]["success"], "failure": last["info"]["failure"],
                "trace": trace, "extra_collective_padding_forwards": 4 - steps,
            })
        folder = self.args.out / f"evaluation-{label}"
        folder.mkdir(exist_ok=True)
        atomic_json(folder / f"rank-{self.rank}.json", {
            "label": label, "predictions": predictions, "workflow_episodes": episodes,
            "development_only": True, "same_generator_workflow_holdout": True,
            "generated_tokens": 0, "production_latency_measured": False,
        })
        self.restore(before)
        self.dist.barrier()
        if self.rank == 0:
            atomic_json(folder / "complete.json", {
                "schema": "bobcat-native-rl-evaluation-v1", "label": label,
                "files": {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()},
                "same_process_restored_rng_and_actor": True,
            })
        self.dist.barrier()

    def run(self):
        if self.args.rl_mechanism == "proper_score_reinforce":
            from bobcat.glm_reinforce_study import run_study

            return run_study(self)
        self.expert_control()
        self.loop = {"arm": "fixed556", "round": 0, "cursor": 0}
        self.evaluation("parent556", expanded=True)
        self.loop = {"arm": "ppo", "round": 0, "cursor": 0}
        self.status("starting_ppo")
        arm_start = time.monotonic()
        proof_done = False
        for r in range(self.args.rounds):
            if self.should_stop(reserve=self.args.final_reserve_seconds + 900):
                break
            self.loop.update(round=r, cursor=0)
            if self.args.rl_mechanism == "calibrated_bandit":
                self.collect_bandit()
            else:
                self.collect()
            self.checkpoint(f"ppo-r{r:03d}-collected")
            for i, transition in enumerate(self.pending):
                self.loop["cursor"] = i
                if not proof_done:
                    before = self.snapshots()
                    begin = time.monotonic()
                    stats = self.ppo_update(transition, check_old=True)
                    expected = self.snapshots()
                    self.restore_checkpoint(self.args.out / f"checkpoint-ppo-r{r:03d}-collected")
                    self.ppo_update(transition, check_old=True)
                    actual = self.snapshots()
                    exact = _same_tree(expected, actual)
                    failed = self.global_flag(not exact)
                    atomic_json(self.args.out / f"rl-update-replay-rank-{self.rank}.json", {
                        "same_process_next_ppo_update_exact": exact,
                        "global_passed": not failed,
                        "all_actor_critic_optimizer_rng_lane_checked": True,
                        "cross_process_resume_verified": False,
                    })
                    if failed:
                        self.restore(before)
                        raise ValueError("The first native PPO update did not replay exactly.")
                    del before, expected, actual
                    proof_done = True
                else:
                    begin = time.monotonic()
                    stats = self.ppo_update(transition)
                self.loop["cursor"] = i + 1
                self.log_update(stats, time.monotonic() - begin)
                if (i + 1) % 4 == 0:
                    self.checkpoint(f"ppo-r{r:03d}-u{i + 1:03d}")
                if self.should_stop(reserve=self.args.final_reserve_seconds):
                    break
            if self.interruption_requested():
                self.checkpoint(f"ppo-r{r:03d}-drain-u{self.loop['cursor']:03d}")
                self.record.update(status="interrupted_checkpointed",
                                   native_rl_executed=proof_done,
                                   finished_at=datetime.now(UTC).isoformat())
                self.status("interrupted_checkpointed")
                return
            if self.loop["cursor"] != len(self.pending):
                # Retain the unfinished on-policy batch and its exact cursor.
                self.checkpoint(f"ppo-r{r:03d}-partial-u{self.loop['cursor']:03d}")
                break
            self.pending = None
            self.loop.update(round=r + 1, cursor=0)
            self.checkpoint(f"ppo-r{r:03d}-complete")
            if self.should_stop(reserve=self.args.final_reserve_seconds):
                break
        ppo_seconds = time.monotonic() - arm_start
        self.record["ppo_arm"] = {
            "wall_seconds_including_rollouts_reference_and_replay": ppo_seconds,
            "actor_updates": self.updates, "critic_updates": self.critic_steps,
            "local_real_transitions": self.transitions, "replay_proof_passed": proof_done,
            "unique_global_bandit_components": self.target_cursor,
            "samples_per_question": self.args.samples_per_question,
            "rl_mechanism": self.args.rl_mechanism,
        }
        self.evaluation("ppo-final", expanded=True)
        self.checkpoint("ppo-final")
        # Keep a standalone PPO result even if the remaining paid time cannot
        # support its separate control; report unmatched time rather than hide it.
        if not self.should_stop(reserve=self.args.final_reserve_seconds + 300):
            restore_local_parameters(self.parameters, self.reference)
            self.critic.load_state_dict(self.critic_initial)
            self.new_optimizers()
            self.sampling.manual_seed(self.args.seed + self.rank)
            self.lane = self.new_lane()
            self.replay_cursor = self.target_cursor = 0
            self.updates = self.transitions = self.critic_steps = 0
            self.loop = {"arm": "supervised", "round": 0, "cursor": 0}
            control_start = time.monotonic()
            for i in range(self.args.max_supervised_updates):
                if (self.global_flag(time.monotonic() - control_start >= ppo_seconds)
                        or self.should_stop(reserve=self.args.final_reserve_seconds)):
                    break
                begin = time.monotonic()
                stats = self.supervised_update()
                self.loop["cursor"] = i + 1
                self.log_update(stats, time.monotonic() - begin)
                if (i + 1) % 4 == 0:
                    self.checkpoint(f"supervised-u{i + 1:04d}")
            if self.interruption_requested():
                self.checkpoint(f"supervised-drain-u{self.loop['cursor']:04d}")
                self.record.update(status="interrupted_checkpointed", native_rl_executed=proof_done,
                                   finished_at=datetime.now(UTC).isoformat())
                self.status("interrupted_checkpointed")
                return
            self.record["supervised_arm"] = {
                "wall_seconds": time.monotonic() - control_start,
                "target_wall_seconds": ppo_seconds, "actor_updates": self.updates,
                "local_teacher_transitions": self.transitions,
                "same_initial_actor_and_world_schedule": True,
                "global_supervised_target_components": self.target_cursor,
                "objective": "direct_brier_kl_and_retention" if
                    self.args.rl_mechanism == "calibrated_bandit" else "visible_workflow_oracle",
            }
            self.evaluation("supervised-final", expanded=True)
            self.checkpoint("supervised-final")
        self.status("verifying_frozen_original_state")
        after = {n: tensor_digest(p) for n, p in self.model.state_dict().items()
                 if "lora_" not in n}
        preserved = after == self.frozen
        if self.global_flag(not preserved):
            raise ValueError("The original pretrained body or vocabulary head changed.")
        self.record.update(status="completed", frozen_original_state_preserved=True,
                           native_rl_executed=proof_done,
                           finished_at=datetime.now(UTC).isoformat())
        self.status("completed")


def main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "source", "out", "curriculum", "suite", "parent-adapter",
                 "parent-complete", "source-root", "git-tree", "source-model-set",
                 "source-verification", "stop-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("parent-adapter-sha256", "parent-complete-sha256", "job-sha256", "recipe-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expert-backend", choices=("torch", "torch_mm"), default="torch")
    parser.add_argument("--rl-mechanism",
                        choices=("proper_score_reinforce", "calibrated_bandit", "workflow"),
                        default="proper_score_reinforce")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--primary-arm-seconds", type=int, default=1200)
    parser.add_argument("--max-bandit-updates", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--samples-per-question", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--max-supervised-updates", type=int, default=160)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-seconds", type=int, default=10800)
    parser.add_argument("--final-reserve-seconds", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-3)
    parser.add_argument("--reference-kl-coefficient", type=float, default=0.)
    parser.add_argument("--replay-coefficient", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=202609232201)
    parser.add_argument("--train-world-start", type=int, default=1000000)
    parser.add_argument("--train-worlds", type=int, default=1024)
    parser.add_argument("--eval-world-start", type=int, default=2000000)
    parser.add_argument("--eval-worlds", type=int, default=32)
    args = parser.parse_args()
    if (not 1 <= args.rounds <= 16 or not 4 <= args.rollout_steps <= 64
            or not 2 <= args.samples_per_question <= 32
            or args.eval_worlds % 4 or not 4 <= args.eval_worlds <= 256
            or not 1800 < args.max_seconds <= 25000
            or not 600 <= args.final_reserve_seconds < args.max_seconds / 2
            or not 0 < args.learning_rate <= 3e-5
            or not 0 < args.critic_learning_rate <= .01
            or not 300 <= args.primary_arm_seconds <= args.max_seconds / 3
            or not 1 <= args.max_bandit_updates <= 512
            or not 4 <= args.checkpoint_every <= 32
            or not 0 <= args.replay_coefficient <= 1
            or (args.rl_mechanism == "proper_score_reinforce"
                and args.reference_kl_coefficient != 0)
            or (args.resume_from and args.rl_mechanism != "proper_score_reinforce")):
        parser.error("Use a finite, explicitly bounded original-model RL pilot.")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / f"rank-{os.environ['RANK']}.json").exists():
        parser.error("Use a fresh output directory for this immutable execution.")
    verify_source(args.source_root, args.git_tree)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    record = {"schema": "bobcat-native-glm-rl-execution-v1",
              "rank": dist.get_rank(), "status": "initializing",
              "started_at": datetime.now(UTC).isoformat(),
              "job_sha256": args.job_sha256, "native_rl_executed": False}
    experiment = None
    try:
        experiment = NativeExperiment(args, record)

        def request_stop(_signal, _frame):
            experiment.stop_requested = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, request_stop)
        experiment.run()
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__,
                      error=str(error)[:3000], finished_at=datetime.now(UTC).isoformat())
        atomic_json(args.out / f"rank-{dist.get_rank()}.json", record)
        raise
    finally:
        if experiment is not None:
            experiment.hook.remove()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
