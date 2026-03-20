#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


MODEL_PRESETS: Dict[str, Dict[str, int]] = {
    "m256_l8_h8": {"d_model": 256, "n_layers": 8, "n_heads": 8},
    "m320_l8_h8": {"d_model": 320, "n_layers": 8, "n_heads": 8},
    "m384_l8_h8": {"d_model": 384, "n_layers": 8, "n_heads": 8},
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def resolve_bundle_path(bundle_root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (bundle_root / path).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive hyperparameter search runner for the IAM bundle.")
    parser.add_argument("--tuning-config", type=str, required=True)
    parser.add_argument("--train-npz", type=str, required=True)
    parser.add_argument("--val-npz", type=str, required=True)
    parser.add_argument("--test-npz", type=str, default=None)
    parser.add_argument("--tokenizer-config", type=str, required=True)
    parser.add_argument("--eval-word-panel-config", type=str, required=True)
    parser.add_argument("--decoding-modes-config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--sample-dir", type=str, required=True)
    parser.add_argument("--num-trials", type=int, default=0)
    parser.add_argument("--search-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-final-retrain", action="store_true")
    return parser


def canonical_signature(overrides: Dict[str, Any]) -> str:
    return json.dumps(overrides, sort_keys=True, separators=(",", ":"))


def sample_from_space(
    *,
    rng: random.Random,
    search_space: Dict[str, Dict[str, Any]],
    completed_trials: List[dict],
    initial_random_trials: int,
) -> Dict[str, Any]:
    use_adaptive = len(completed_trials) >= max(1, initial_random_trials)
    parent: Optional[dict] = None
    if use_adaptive:
        ranked = [trial for trial in completed_trials if trial.get("status") == "completed"]
        ranked.sort(key=lambda item: float(item["objective"]))
        if ranked:
            parent = rng.choice(ranked[: max(1, min(3, len(ranked)))])

    overrides: Dict[str, Any] = {}
    parent_overrides = parent.get("overrides", {}) if parent else {}
    for name, spec in search_space.items():
        kind = str(spec["type"])
        if kind == "categorical":
            values = list(spec["values"])
            if parent is not None and rng.random() < 0.65 and name in parent_overrides:
                overrides[name] = parent_overrides[name]
            else:
                overrides[name] = rng.choice(values)
            continue

        low = spec.get("low")
        high = spec.get("high")
        log_scale = bool(spec.get("log", False))
        step = spec.get("step")
        parent_value = parent_overrides.get(name) if parent is not None else None

        if parent_value is None or rng.random() < 0.35:
            value = sample_numeric_global(rng=rng, kind=kind, low=low, high=high, log_scale=log_scale, step=step)
        else:
            value = sample_numeric_local(
                rng=rng,
                kind=kind,
                low=low,
                high=high,
                log_scale=log_scale,
                step=step,
                parent_value=parent_value,
            )
        overrides[name] = value

    return postprocess_overrides(overrides)


def sample_numeric_global(
    *,
    rng: random.Random,
    kind: str,
    low: Any,
    high: Any,
    log_scale: bool,
    step: Any,
) -> Any:
    if kind == "int":
        low_i = int(low)
        high_i = int(high)
        if step is not None:
            step_i = int(step)
            values = list(range(low_i, high_i + 1, step_i))
            return int(rng.choice(values))
        return int(rng.randint(low_i, high_i))

    low_f = float(low)
    high_f = float(high)
    if log_scale:
        value = math.exp(rng.uniform(math.log(low_f), math.log(high_f)))
    else:
        value = rng.uniform(low_f, high_f)
    if step is not None:
        step_f = float(step)
        value = round((value - low_f) / step_f) * step_f + low_f
    return float(min(max(value, low_f), high_f))


def sample_numeric_local(
    *,
    rng: random.Random,
    kind: str,
    low: Any,
    high: Any,
    log_scale: bool,
    step: Any,
    parent_value: Any,
) -> Any:
    if kind == "int":
        low_i = int(low)
        high_i = int(high)
        span = max(1, high_i - low_i)
        sigma = max(1.0, span * 0.15)
        value = int(round(rng.gauss(float(parent_value), sigma)))
        if step is not None:
            step_i = int(step)
            value = int(round((value - low_i) / step_i) * step_i + low_i)
        return int(min(max(value, low_i), high_i))

    low_f = float(low)
    high_f = float(high)
    if log_scale:
        log_low = math.log(low_f)
        log_high = math.log(high_f)
        log_parent = math.log(float(parent_value))
        sigma = max(0.05, (log_high - log_low) * 0.18)
        value = math.exp(rng.gauss(log_parent, sigma))
    else:
        sigma = max(1e-6, (high_f - low_f) * 0.15)
        value = rng.gauss(float(parent_value), sigma)
    if step is not None:
        step_f = float(step)
        value = round((value - low_f) / step_f) * step_f + low_f
    return float(min(max(value, low_f), high_f))


def postprocess_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(overrides)
    if "model_preset" in result:
        preset_name = str(result.pop("model_preset"))
        preset = MODEL_PRESETS[preset_name]
        result.update(preset)

    keep_min = result.get("train_downsample_keep_min")
    keep_max = result.get("train_downsample_keep_max")
    if keep_min is not None and keep_max is not None:
        keep_min = float(keep_min)
        keep_max = float(keep_max)
        if keep_max <= keep_min:
            keep_max = min(0.99, keep_min + 0.05)
        result["train_downsample_keep_min"] = float(min(max(keep_min, 0.0), 0.99))
        result["train_downsample_keep_max"] = float(min(max(keep_max, result["train_downsample_keep_min"] + 0.01), 0.995))

    if "dropout" in result:
        result["dropout"] = float(min(max(float(result["dropout"]), 0.0), 0.5))
    if "train_label_smoothing" in result:
        result["train_label_smoothing"] = float(min(max(float(result["train_label_smoothing"]), 0.0), 0.2))
    return result


def build_trial_config(
    *,
    base_config: Dict[str, Any],
    tuning_config: Dict[str, Any],
    overrides: Dict[str, Any],
    trial_index: int,
) -> Dict[str, Any]:
    config = dict(base_config)
    config.update(tuning_config.get("search_overrides", {}))
    config.update(overrides)
    search_epochs = int(tuning_config.get("search_epochs", config.get("epochs", 10)))
    search_max_train_samples = int(tuning_config.get("search_max_train_samples", 0))
    search_max_val_samples = int(tuning_config.get("search_max_val_samples", 0))
    search_max_test_samples = int(tuning_config.get("search_max_test_samples", 0))
    config["epochs"] = search_epochs
    config["max_train_samples"] = search_max_train_samples
    config["max_val_samples"] = search_max_val_samples
    config["max_test_samples"] = search_max_test_samples
    config["experiment_name"] = f"{base_config.get('experiment_name', 'iam_tune')}_trial_{trial_index:03d}"
    return config


def build_final_config(
    *,
    base_config: Dict[str, Any],
    tuning_config: Dict[str, Any],
    best_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    config = dict(base_config)
    config.update(best_overrides)
    config.update(tuning_config.get("final_overrides", {}))
    config["experiment_name"] = f"{base_config.get('experiment_name', 'iam_tune')}_best_final"
    return config


def run_training(
    *,
    bundle_root: Path,
    config_path: Path,
    train_npz: Path,
    val_npz: Path,
    tokenizer_config: Path,
    eval_word_panel_config: Path,
    decoding_modes_config: Path,
    out_dir: Path,
    sample_dir: Path,
    log_path: Path,
    test_npz: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "train.py",
        "--config",
        config_path.as_posix(),
        "--train-npz",
        train_npz.as_posix(),
        "--val-npz",
        val_npz.as_posix(),
        "--tokenizer-config",
        tokenizer_config.as_posix(),
        "--eval-word-panel-config",
        eval_word_panel_config.as_posix(),
        "--decoding-modes-config",
        decoding_modes_config.as_posix(),
        "--out-dir",
        out_dir.as_posix(),
        "--sample-dir",
        sample_dir.as_posix(),
    ]
    if test_npz is not None:
        cmd.extend(["--test-npz", test_npz.as_posix()])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.run(
            cmd,
            cwd=bundle_root.as_posix(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process


def extract_trial_result(
    *,
    trial_index: int,
    overrides: Dict[str, Any],
    out_dir: Path,
    sample_dir: Path,
    process: subprocess.CompletedProcess[str],
    started_at: float,
) -> Dict[str, Any]:
    run_summary_path = out_dir / "run_summary.json"
    duration_s = time.time() - started_at
    result = {
        "trial_index": int(trial_index),
        "status": "failed",
        "overrides": overrides,
        "objective": None,
        "duration_s": float(duration_s),
        "returncode": int(process.returncode),
        "out_dir": out_dir.as_posix(),
        "sample_dir": sample_dir.as_posix(),
        "run_summary_path": run_summary_path.as_posix() if run_summary_path.exists() else None,
    }
    if process.returncode != 0 or not run_summary_path.exists():
        return result

    summary = load_json(run_summary_path)
    result.update(
        {
            "status": "completed",
            "objective": float(summary["best_val_nll"]),
            "best_val_nll": float(summary["best_val_nll"]),
            "best_val_ppl": float(summary["best_val_ppl"]) if summary.get("best_val_ppl") is not None else None,
            "best_epoch": int(summary["best_epoch"]) if summary.get("best_epoch") is not None else None,
            "best_checkpoint_path": summary.get("best_checkpoint_path"),
            "training_regime": summary.get("training_regime"),
            "eos_bottleneck_hint": (summary.get("eos_summary") or {}).get("eos_bottleneck_hint"),
        }
    )
    return result


def load_completed_trials(results_path: Path) -> List[dict]:
    if not results_path.exists():
        return []
    trials: List[dict] = []
    with results_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            trials.append(json.loads(line))
    return trials


def append_trial_result(results_path: Path, result: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as handle:
        handle.write(json.dumps(result) + "\n")


def choose_best_trial(trials: List[dict]) -> Optional[dict]:
    completed = [trial for trial in trials if trial.get("status") == "completed" and trial.get("objective") is not None]
    if not completed:
        return None
    return min(completed, key=lambda item: float(item["objective"]))


def write_tuning_summary(
    *,
    out_dir: Path,
    tuning_config: Dict[str, Any],
    completed_trials: List[dict],
    best_trial: Optional[dict],
    final_result: Optional[dict],
) -> None:
    ranking = sorted(
        [trial for trial in completed_trials if trial.get("status") == "completed" and trial.get("objective") is not None],
        key=lambda item: float(item["objective"]),
    )
    summary = {
        "search_name": tuning_config.get("search_name"),
        "num_trials_requested": int(tuning_config.get("num_trials", 0)),
        "num_trials_completed": int(len(completed_trials)),
        "num_successful_trials": int(len(ranking)),
        "best_trial": best_trial,
        "top_trials": ranking[:5],
        "final_retrain": final_result,
    }
    dump_json(out_dir / "tuning_summary.json", summary)

    lines = [
        f"# {tuning_config.get('search_name', 'IAM Hyperparameter Search')}",
        "",
        f"- Requested trials: `{summary['num_trials_requested']}`",
        f"- Completed trials: `{summary['num_trials_completed']}`",
        f"- Successful trials: `{summary['num_successful_trials']}`",
        "",
    ]
    if best_trial is None:
        lines.append("- No successful trials were completed.")
    else:
        lines.extend(
            [
                "## Best Trial",
                "",
                f"- Trial index: `{best_trial['trial_index']}`",
                f"- Best validation NLL: `{best_trial['best_val_nll']}`",
                f"- Best validation PPL: `{best_trial.get('best_val_ppl')}`",
                f"- Best epoch: `{best_trial.get('best_epoch')}`",
                f"- Best checkpoint path: `{best_trial.get('best_checkpoint_path')}`",
                f"- Overrides: `{json.dumps(best_trial.get('overrides', {}), sort_keys=True)}`",
                "",
                "## Top Trials",
                "",
            ]
        )
        for trial in ranking[:5]:
            lines.append(
                f"- Trial `{trial['trial_index']}`: val_nll={trial['best_val_nll']} "
                f"epoch={trial.get('best_epoch')} overrides=`{json.dumps(trial.get('overrides', {}), sort_keys=True)}`"
            )
    if final_result is not None:
        lines.extend(
            [
                "",
                "## Final Retrain",
                "",
                f"- Status: `{final_result.get('status')}`",
                f"- Out dir: `{final_result.get('out_dir')}`",
                f"- Best validation NLL: `{final_result.get('best_val_nll')}`",
                f"- Best checkpoint path: `{final_result.get('best_checkpoint_path')}`",
            ]
        )
    (out_dir / "tuning_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle_root = Path(__file__).resolve().parent

    tuning_config_path = resolve_bundle_path(bundle_root, args.tuning_config)
    train_npz = resolve_bundle_path(bundle_root, args.train_npz)
    val_npz = resolve_bundle_path(bundle_root, args.val_npz)
    test_npz = resolve_bundle_path(bundle_root, args.test_npz) if args.test_npz else None
    tokenizer_config = resolve_bundle_path(bundle_root, args.tokenizer_config)
    eval_word_panel_config = resolve_bundle_path(bundle_root, args.eval_word_panel_config)
    decoding_modes_config = resolve_bundle_path(bundle_root, args.decoding_modes_config)
    out_dir = resolve_bundle_path(bundle_root, args.out_dir)
    sample_dir = resolve_bundle_path(bundle_root, args.sample_dir)
    assert (
        tuning_config_path is not None
        and train_npz is not None
        and val_npz is not None
        and tokenizer_config is not None
        and eval_word_panel_config is not None
        and decoding_modes_config is not None
        and out_dir is not None
        and sample_dir is not None
    )

    tuning_config = load_json(tuning_config_path)
    if args.num_trials > 0:
        tuning_config["num_trials"] = int(args.num_trials)
    if args.search_epochs > 0:
        tuning_config["search_epochs"] = int(args.search_epochs)
    if args.seed > 0:
        tuning_config["seed"] = int(args.seed)

    base_config_path = resolve_bundle_path(bundle_root, tuning_config["base_experiment_config"])
    assert base_config_path is not None
    base_config = load_json(base_config_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "resolved_tuning_config.json", tuning_config)

    results_path = out_dir / "trial_results.jsonl"
    completed_trials = load_completed_trials(results_path)
    seen_signatures = {
        canonical_signature(trial.get("overrides", {}))
        for trial in completed_trials
        if trial.get("overrides") is not None
    }

    rng = random.Random(int(tuning_config.get("seed", 3407)))
    for _ in range(len(completed_trials)):
        rng.random()

    num_trials = int(tuning_config.get("num_trials", 0))
    initial_random_trials = int(tuning_config.get("initial_random_trials", 3))
    search_space = dict(tuning_config.get("search_space", {}))
    bundle_train_script = bundle_root / "train.py"

    while len(completed_trials) < num_trials:
        trial_index = len(completed_trials)
        trial_overrides: Optional[Dict[str, Any]] = None
        for _attempt in range(128):
            candidate = sample_from_space(
                rng=rng,
                search_space=search_space,
                completed_trials=completed_trials,
                initial_random_trials=initial_random_trials,
            )
            signature = canonical_signature(candidate)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                trial_overrides = candidate
                break
        if trial_overrides is None:
            raise RuntimeError("Unable to sample a new hyperparameter candidate without duplication.")

        trial_dir = out_dir / "trials" / f"trial_{trial_index:03d}"
        trial_sample_dir = sample_dir / "trials" / f"trial_{trial_index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_sample_dir.mkdir(parents=True, exist_ok=True)
        trial_config = build_trial_config(
            base_config=base_config,
            tuning_config=tuning_config,
            overrides=trial_overrides,
            trial_index=trial_index,
        )
        trial_config_path = trial_dir / "trial_config.json"
        dump_json(trial_config_path, trial_config)

        print(
            f"[tune] starting trial {trial_index + 1}/{num_trials} "
            f"overrides={json.dumps(trial_overrides, sort_keys=True)}"
        )
        started_at = time.time()
        process = run_training(
            bundle_root=bundle_root,
            config_path=trial_config_path,
            train_npz=train_npz,
            val_npz=val_npz,
            tokenizer_config=tokenizer_config,
            eval_word_panel_config=eval_word_panel_config,
            decoding_modes_config=decoding_modes_config,
            out_dir=trial_dir / "checkpoints",
            sample_dir=trial_sample_dir,
            log_path=trial_dir / "train.log",
            test_npz=None,
        )
        result = extract_trial_result(
            trial_index=trial_index,
            overrides=trial_overrides,
            out_dir=trial_dir / "checkpoints",
            sample_dir=trial_sample_dir,
            process=process,
            started_at=started_at,
        )
        dump_json(trial_dir / "trial_result.json", result)
        append_trial_result(results_path, result)
        completed_trials.append(result)
        if result["status"] == "completed":
            print(
                f"[tune] trial {trial_index:03d} completed "
                f"best_val_nll={result['best_val_nll']:.6f} duration_s={result['duration_s']:.1f}"
            )
        else:
            print(
                f"[tune] trial {trial_index:03d} failed returncode={result['returncode']} "
                f"duration_s={result['duration_s']:.1f}"
            )

    best_trial = choose_best_trial(completed_trials)
    final_result = None
    if best_trial is not None and not args.skip_final_retrain and bool(tuning_config.get("final_retrain_best", True)):
        final_dir = out_dir / "final_best"
        final_sample_dir = sample_dir / "final_best"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_sample_dir.mkdir(parents=True, exist_ok=True)
        final_config = build_final_config(
            base_config=base_config,
            tuning_config=tuning_config,
            best_overrides=best_trial["overrides"],
        )
        final_config_path = final_dir / "best_config.json"
        dump_json(final_config_path, final_config)
        print(
            f"[tune] launching final retrain with best trial {best_trial['trial_index']} "
            f"overrides={json.dumps(best_trial['overrides'], sort_keys=True)}"
        )
        started_at = time.time()
        process = run_training(
            bundle_root=bundle_root,
            config_path=final_config_path,
            train_npz=train_npz,
            val_npz=val_npz,
            test_npz=test_npz,
            tokenizer_config=tokenizer_config,
            eval_word_panel_config=eval_word_panel_config,
            decoding_modes_config=decoding_modes_config,
            out_dir=final_dir / "checkpoints",
            sample_dir=final_sample_dir,
            log_path=final_dir / "train.log",
        )
        final_result = extract_trial_result(
            trial_index=-1,
            overrides=best_trial["overrides"],
            out_dir=final_dir / "checkpoints",
            sample_dir=final_sample_dir,
            process=process,
            started_at=started_at,
        )
        dump_json(final_dir / "final_result.json", final_result)
        print(
            f"[tune] final retrain status={final_result['status']} "
            f"best_val_nll={final_result.get('best_val_nll')}"
        )

    write_tuning_summary(
        out_dir=out_dir,
        tuning_config=tuning_config,
        completed_trials=completed_trials,
        best_trial=best_trial,
        final_result=final_result,
    )
    if best_trial is None:
        raise SystemExit("No successful tuning trials completed.")


if __name__ == "__main__":
    main()
