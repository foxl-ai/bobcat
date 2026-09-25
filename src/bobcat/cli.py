from __future__ import annotations

import argparse
import json
from pathlib import Path

from bobcat.batching import EncodedDataset
from bobcat.benchmark import benchmark_model
from bobcat.data import generate_dataset
from bobcat.metrics import evaluate_rows, fit_referral_policy, fit_temperature, scored_row
from bobcat.model import ModelConfig, parameter_count
from bobcat.schema import SENTINELS, Choice, Example, file_hash, read_examples
from bobcat.tokenization import ScratchTokenizer
from bobcat.training import TrainConfig, load_model, pick_device, predict, save_json, train


def load_configuration(path: Path, tokenizer: ScratchTokenizer | None = None) -> ModelConfig:
    values = json.loads(path.read_text())
    if tokenizer is not None:
        values["vocab_size"] = tokenizer.vocab_size
    return ModelConfig(**values)


def model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")


def checked_model(args):
    device = pick_device(args.device)
    tokenizer = ScratchTokenizer(args.tokenizer)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint["provenance"]["tokenizer_sha256"] != tokenizer.digest:
        raise ValueError("Checkpoint and tokenizer do not match.")
    return model, checkpoint, tokenizer, device


def main() -> None:
    parser = argparse.ArgumentParser(description="Bobcat scratch decision model research tools")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--train-worlds", type=int, default=1600)
    generate.add_argument("--eval-worlds", type=int, default=240)
    generate.add_argument("--seed", type=int, default=41)
    tokenize = sub.add_parser("tokenizer")
    tokenize.add_argument("--train", type=Path, required=True)
    tokenize.add_argument("--out", type=Path, required=True)
    tokenize.add_argument("--vocab-size", type=int, default=2048)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--config", type=Path, required=True)
    training = sub.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--train", type=Path, required=True)
    training.add_argument("--tokenizer", type=Path, required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--dev", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--init-from", type=Path)
    training.add_argument("--steps", type=int, default=600)
    training.add_argument("--groups-per-batch", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=0.0005)
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--device", default="auto")
    training.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    training.add_argument("--max-train-seconds", type=float)
    training.add_argument("--objective", choices=["decision", "mlm"], default="decision")
    training.add_argument("--group-strategy", choices=["context", "world"], default="context")
    training.add_argument("--counterfactual-weight", type=float, default=0)
    evaluation = sub.add_parser("evaluate")
    model_arguments(evaluation)
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--calibration", type=Path)
    evaluation.add_argument("--release", type=Path)
    calibrate = sub.add_parser("calibrate")
    model_arguments(calibrate)
    calibrate.add_argument("--temperature-data", type=Path, required=True)
    calibrate.add_argument("--policy-data", type=Path, required=True)
    calibrate.add_argument("--target-risk", type=float, default=0.05)
    calibrate.add_argument("--out", type=Path, required=True)
    benchmark = sub.add_parser("benchmark")
    model_arguments(benchmark)
    benchmark.add_argument("--data", type=Path, required=True)
    benchmark.add_argument("--out", type=Path, required=True)
    benchmark.add_argument("--repetitions", type=int, default=50)
    freeze = sub.add_parser("freeze")
    model_arguments(freeze)
    freeze.add_argument("--calibration", type=Path, required=True)
    freeze.add_argument("--decision", required=True, help="Reason this final experiment is frozen.")
    freeze.add_argument("--out", type=Path, required=True)
    decide = sub.add_parser("decide")
    model_arguments(decide)
    decide.add_argument("--input", type=Path, required=True)
    decide.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        report = generate_dataset(args.out, args.train_worlds, args.eval_worlds, args.seed)
        print(json.dumps(report, indent=2))
        return
    if args.command == "tokenizer":
        tokenizer = ScratchTokenizer.train(read_examples(args.train), args.out, args.vocab_size)
        print(json.dumps({"vocab_size": tokenizer.vocab_size, "sha256": tokenizer.digest}))
        return
    if args.command == "inspect":
        config = load_configuration(args.config)
        print(
            json.dumps(
                {"config": config.to_dict(), "parameters": parameter_count(config)}, indent=2
            )
        )
        return
    if args.command == "train":
        tokenizer = ScratchTokenizer(args.tokenizer)
        config = load_configuration(args.config, tokenizer)
        settings = TrainConfig(
            steps=args.steps,
            groups_per_batch=args.groups_per_batch,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            precision=args.precision,
            max_train_seconds=args.max_train_seconds,
            objective=args.objective,
            group_strategy=args.group_strategy,
            counterfactual_weight=args.counterfactual_weight,
        )
        result = train(
            config,
            settings,
            args.train,
            args.tokenizer,
            args.out,
            dev_path=args.dev,
            resume=args.resume,
            init_from=args.init_from,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    model, checkpoint, tokenizer, device = checked_model(args)
    calibration = None
    if getattr(args, "calibration", None):
        calibration = json.loads(args.calibration.read_text())
        if calibration["checkpoint_sha256"] != file_hash(args.checkpoint):
            raise ValueError("Calibration was fitted to another checkpoint.")
        if calibration["precision"] != args.precision:
            raise ValueError("Calibration and inference precision must match.")
    if args.command == "freeze":
        if args.out.exists():
            raise ValueError("A release record is immutable; choose a new name.")
        save_json(
            args.out,
            {
                "checkpoint_sha256": file_hash(args.checkpoint),
                "tokenizer_sha256": tokenizer.digest,
                "calibration_sha256": file_hash(args.calibration),
                "decision": args.decision,
                "precision": args.precision,
                "scope": "Frozen research evaluation; not a product-readiness claim.",
            },
        )
        print(args.out)
        return
    if args.command == "calibrate":
        temperature_examples = read_examples(args.temperature_data)
        policy_examples = read_examples(args.policy_data)
        if {e.group_id for e in temperature_examples} & {e.group_id for e in policy_examples}:
            raise ValueError("Temperature and threshold partitions overlap.")
        temperature_rows = predict(
            model,
            EncodedDataset(temperature_examples, tokenizer, model.config),
            device,
            args.precision,
        )
        policy_rows = predict(
            model, EncodedDataset(policy_examples, tokenizer, model.config), device, args.precision
        )
        temperature = fit_temperature(temperature_rows)
        policy = fit_referral_policy(policy_rows, temperature, args.target_risk)
        result = {
            "checkpoint_sha256": file_hash(args.checkpoint),
            "tokenizer_sha256": tokenizer.digest,
            "precision": args.precision,
            "temperature_data_sha256": file_hash(args.temperature_data),
            "policy_data_sha256": file_hash(args.policy_data),
            "temperature": temperature,
            "policy": policy,
            "raw_temperature_partition": evaluate_rows(temperature_rows),
            "fitted_temperature_partition": evaluate_rows(temperature_rows, temperature),
        }
        save_json(args.out, result)
        print(json.dumps({"temperature": temperature, "policy": policy}, indent=2))
        return
    if args.command == "decide":
        request = json.loads(args.input.read_text())
        requests = request if isinstance(request, list) else [request]
        examples = [
            Example(
                id=f"inference-{i}",
                group_id=f"inference-{i}",
                family="user",
                split="inference",
                context=item["context"],
                instruction=item["instruction"],
                choices=[Choice(**c) for c in item["choices"]],
                target=item["choices"][0]["id"],
                kind=item.get("kind", "choice"),
            )
            for i, item in enumerate(requests)
        ]
        for example in examples:
            example.validate()
        rows = predict(
            model, EncodedDataset(examples, tokenizer, model.config), device, args.precision
        )
        outputs = []
        for row in rows:
            scored = scored_row(row, calibration["temperature"] if calibration else 1.0)
            threshold = calibration["policy"]["threshold"] if calibration else None
            execute = (
                threshold is not None
                and scored["prediction"] not in SENTINELS
                and scored["top_probability"] >= threshold
            )
            outputs.append(
                {
                    "decision": scored["prediction"],
                    "probabilities": dict(
                        zip(scored["candidate_ids"], scored["probabilities"], strict=True)
                    ),
                    "operational_action": "accept" if execute else "refer",
                    "calibrated": calibration is not None,
                }
            )
        print(json.dumps(outputs if isinstance(request, list) else outputs[0], indent=2))
        return
    examples = read_examples(args.data)
    if any(example.split.startswith("test_") for example in examples):
        if args.command != "evaluate" or args.release is None or calibration is None:
            raise ValueError("Final test is sealed. Supply a frozen release and calibration.")
        release = json.loads(args.release.read_text())
        if (
            release["checkpoint_sha256"] != file_hash(args.checkpoint)
            or release["tokenizer_sha256"] != tokenizer.digest
            or release["calibration_sha256"] != file_hash(args.calibration)
            or release["precision"] != args.precision
        ):
            raise ValueError("Frozen release does not match the evaluated artifacts.")
    dataset = EncodedDataset(examples, tokenizer, model.config)
    if args.command == "benchmark":
        result = benchmark_model(model, dataset, device, args.precision, args.repetitions)
        save_json(args.out, result)
    else:
        rows = predict(model, dataset, device, args.precision)
        result = evaluate_rows(
            rows,
            calibration["temperature"] if calibration else 1.0,
            calibration["policy"] if calibration else None,
        )
        result["checkpoint_sha256"] = file_hash(args.checkpoint)
        result["data_sha256"] = file_hash(args.data)
        result["split"] = sorted({example.split for example in examples})
        save_json(args.out, result)
        save_json(args.out.with_suffix(".predictions.json"), {"rows": rows})
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
