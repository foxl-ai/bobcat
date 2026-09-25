import argparse
import copy
import json
import os
import signal
import socket
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from bobcat.checkpoints import verify_checkpoint, write_checkpoint
from bobcat.corpus import atomic_json
from bobcat.decision_train import (
    FORMAT,
    context_weights,
    load_language_parent,
    load_partitions,
    parameter_groups,
    run,
)
from bobcat.model import DecisionModel, ModelConfig
from bobcat.pretrain import mask_tokens, trainable_language_parameters
from bobcat.schema import Choice, Example, file_hash, read_examples, write_examples
from bobcat.serve import StudentScorer
from bobcat.tokenization import BOS, EOS


@pytest.fixture
def decision_run(corpus, tmp_path, request):
    tokenizer = corpus[2]
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2,
        encoder_layers=2, schema_layers=1, reader_blocks=1,
        max_context_tokens=256, max_schema_tokens=getattr(
            request, "param", {},
        ).get("max_schema_tokens", 128), dropout=0.2, mlm_transform=True,
    )
    torch.manual_seed(923)
    parent = DecisionModel(config)
    optimizer = torch.optim.AdamW(trainable_language_parameters(parent), lr=0.001)
    ids = torch.tensor([[BOS, *tokenizer.encode("A customer asks about a payment."), EOS]])
    corrupted, selected, labels = mask_tokens(ids, tokenizer.vocab_size, 31)
    F.cross_entropy(parent.mlm_logits(corrupted, selected), labels).backward()
    optimizer.step()
    checkpoint = tmp_path / "language" / "last.pt"
    write_checkpoint({
        "format": "bobcat-real-mlm-v1", "config": config.to_dict(),
        "model": parent.state_dict(), "optimizer": optimizer.state_dict(),
        "step": 1, "counters": {"tokens": ids.numel()},
        "provenance": {
            "initialization": "random", "tokenizer_sha256": tokenizer.digest,
            "tokenizer_encoding_profile": tokenizer.encoding_profile,
            "test_fixture": True,
        },
    }, checkpoint)
    data = tmp_path / "data"
    data.mkdir()
    files = {}
    for split, start, count in (("train", 0, 8), ("dev_train", 20, 3)):
        examples = []
        for index in range(start, start + count):
            for view in range(3 if index % 2 == 0 else 1):
                examples.append(Example(
                    id=f"fixture-{index}-{view}", group_id=f"context-{index}",
                    family="fixture", split=split,
                    context=f"Customer message {index}: a payment needs to be checked.",
                    instruction=f"Using criterion {view}, select the matching category.",
                    choices=[Choice("billing", "billing support"), Choice("other", "other")],
                    target="billing", metadata={"annotation_origin": "unit_fixture"},
                ))
        path = data / f"{split}.jsonl"
        write_examples(path, examples)
        files[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    atomic_json(data / "manifest.json", {
        "schema": "bobcat-korean-decisions-v1", "files": files,
        "dataset_license": "project-authored software-test fixture",
    })
    return argparse.Namespace(
        initial_mlm=checkpoint, tokenizer=tokenizer.path, data=data, out=tmp_path / "run",
        resume=None, device="cpu", minimum_parent_tokens=1, contexts_per_gpu=2,
        accumulation=2, freeze_backbone=False, backbone_lr=0.0002, reader_lr=0.001,
        weight_decay=0.01, schedule_steps=4, max_steps=4, warmup_steps=1,
        max_seconds=600, checkpoint_reserve_seconds=30, save_every=2,
        eval_every=2, eval_batch_size=4, log_every=1, seed=311, cpu_threads=1,
        checkpoint_s3=None, s3_region=None, stop_file=None,
    )


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_equal(x, y)
    else:
        assert a == b


def test_parent_transfer_preserves_language_weights_and_freezes_unused_mlm_head(decision_run):
    from bobcat.tokenization import ScratchTokenizer
    tokenizer = ScratchTokenizer(decision_run.tokenizer)
    model, identity = load_language_parent(decision_run.initial_mlm, tokenizer)
    saved = torch.load(decision_run.initial_mlm, weights_only=True)
    assert_equal(model.state_dict(), saved["model"])
    assert identity["checkpoint_sha256"] == file_hash(decision_run.initial_mlm)
    parameter_groups(model, backbone_lr=1e-5, reader_lr=1e-4, freeze_backbone=False)
    assert model.embedding.weight.requires_grad
    assert not model.mlm_bias.requires_grad
    assert all(not p.requires_grad for p in model.mlm_transform.parameters())
    assert not model.score.bias.requires_grad
    assert not model.decision_norm.bias.requires_grad
    # Changing either common offset must leave the candidate distribution intact.
    model.eval()
    train, _, _ = load_partitions(decision_run.data, tokenizer, model.config)
    inputs = train.collate([0, 1, 2, 3]).model_inputs()
    with torch.no_grad():
        before = model(inputs).softmax(-1)
        model.score.bias.add_(3.0)
        model.decision_norm.bias.add_(torch.linspace(-1.0, 1.0, model.config.d_model))
        torch.testing.assert_close(before, model(inputs).softmax(-1), rtol=1e-5, atol=1e-6)
    tokenizer.digest = "0" * 64
    with pytest.raises(ValueError):
        load_language_parent(decision_run.initial_mlm, tokenizer)


def test_all_views_share_one_context_weight_and_no_extra_candidates(decision_run):
    from bobcat.tokenization import ScratchTokenizer
    tokenizer = ScratchTokenizer(decision_run.tokenizer)
    model, _ = load_language_parent(decision_run.initial_mlm, tokenizer)
    train, _, _ = load_partitions(decision_run.data, tokenizer, model.config)
    batch = train.collate([0, 1, 2, 3])
    weights = context_weights(batch, train)
    assert torch.allclose(weights, torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0]).double())
    assert weights.sum() == 2
    assert all(ids == ["billing", "other"] for ids in batch.candidate_ids)
    assert batch.unique_contexts == 2


def test_decision_resume_matches_uninterrupted_weights_optimizer_and_rng(decision_run, monkeypatch):
    original = DecisionModel.forward
    seen = []

    def forward(model, inputs):
        assert set(inputs) == {
            "context_ids", "context_index", "schema_ids", "schema_candidate_mask",
            "candidate_keep", "joint_ids", "candidate_positions", "joint_candidate_mask",
        }
        seen.append(True)
        return original(model, inputs)

    monkeypatch.setattr(DecisionModel, "forward", forward)
    full = copy.copy(decision_run)
    full.out = decision_run.out.parent / "full"
    run(full)
    first = copy.copy(decision_run)
    first.max_steps = 2
    run(first)
    resumed = copy.copy(decision_run)
    resumed.out = decision_run.out.parent / "resumed"
    resumed.resume = first.out / "last.pt"
    run(resumed)
    complete = torch.load(full.out / "last.pt", weights_only=True)
    recovered = torch.load(resumed.out / "last.pt", weights_only=True)
    assert complete["step"] == recovered["step"] == 4
    for field in ("model", "optimizer", "rank_rng_states"):
        assert_equal(complete[field], recovered[field])
    for field in ("tokens", "questions", "context_draws"):
        assert complete["counters"][field] == recovered["counters"][field]
    verify_checkpoint(resumed.out / "last.pt", expected_format=FORMAT)
    scorer = StudentScorer(resumed.out / "last.pt", decision_run.tokenizer,
                           device="cpu", allow_unvalidated=True)
    assert scorer.release_gate_passed is False
    assert scorer.model.config.to_dict() == complete["config"]
    assert seen


def test_changed_partition_is_rejected_before_training(decision_run):
    path = decision_run.data / "train.jsonl"
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        run(decision_run)
    assert not decision_run.out.exists()


def install_mean_fixture(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    for split in ("train", "dev_train"):
        path = args.data / f"{split}.jsonl"
        examples = read_examples(path)
        for e in examples:
            if int(e.id.split("-")[1]) % 2:
                e.kind, e.target = "ordinal", None
                e.supervision, e.score_target = "score_mean", 1 / 3
                e.instruction = "Rate the urgency on the ordered scale."
                e.choices = [Choice(str(i), text) for i, text in enumerate(
                    ("can wait", "today", "immediately"),
                )]
        write_examples(path, examples)
        manifest["files"][path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    atomic_json(args.data / "manifest.json", manifest)


@pytest.mark.parametrize("with_means", [False, True])
def test_two_rank_context_weighting_and_resume(decision_run, with_means):
    """Different question counts must still give each source context weight one."""
    if with_means:
        install_mean_fixture(decision_run)
    payload = torch.load(decision_run.initial_mlm, weights_only=True)
    payload["config"]["dropout"] = 0.0
    parent_path = decision_run.initial_mlm.parent.parent / "ddp-parent" / "last.pt"
    write_checkpoint(payload, parent_path)
    decision_run.initial_mlm = parent_path
    serial = copy.copy(decision_run)
    serial.accumulation = 4  # The same indices as 2 ranks x 2 microbatches.
    serial.out = decision_run.out.parent / "serial"
    run(serial)

    def distributed(destination, max_steps=4, resume=None):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--master-addr=127.0.0.1", f"--master-port={port}",
            "--nproc_per_node=2", "-m", "bobcat.decision_train",
            "--device", "cpu", "--initial-mlm", str(decision_run.initial_mlm),
            "--tokenizer", str(decision_run.tokenizer), "--data", str(decision_run.data),
            "--out", str(destination), "--max-steps", str(max_steps),
            "--minimum-parent-tokens", "1", "--max-seconds", "600",
            "--checkpoint-reserve-seconds", "30", "--schedule-steps", "4",
            "--warmup-steps", "1", "--contexts-per-gpu", "2", "--accumulation", "2",
            "--backbone-lr", "0.0002", "--reader-lr", "0.001", "--weight-decay", "0.01",
            "--seed", "311", "--eval-every", "2", "--eval-batch-size", "4",
            "--save-every", "2", "--log-every", "1", "--cpu-threads", "1",
        ]
        if resume:
            command.extend(["--resume", str(resume)])
        environment = {
            key: value for key, value in os.environ.items()
            if key not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"}
        }
        environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=environment, start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                pytest.fail("Distributed test deadline exceeded:\n" + stdout + stderr)
            assert process.returncode == 0, stdout + stderr
        return torch.load(destination / "last.pt", weights_only=True)

    full = distributed(decision_run.out.parent / "distributed")
    first = decision_run.out.parent / "first"
    partial = distributed(first, max_steps=2)
    assert partial["step"] == 2
    recovered = distributed(decision_run.out.parent / "resumed", resume=first / "last.pt")
    for key in ("model", "optimizer", "rank_rng_states"):
        assert_equal(full[key], recovered[key])
    serial_weights = torch.load(serial.out / "last.pt", weights_only=True)
    assert full["counters"]["context_draws"] == serial_weights["counters"]["context_draws"] == 32
    assert full["counters"]["questions"] == serial_weights["counters"]["questions"]
    for name, tensor in full["model"].items():
        torch.testing.assert_close(
            tensor, serial_weights["model"][name], rtol=1e-4, atol=2e-6,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )
