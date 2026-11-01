"""One-command orchestration for the real 8k Qwen-3B experiment.

Nothing runs merely by importing this module. To inspect the planned downloads:

    python -m src.run_8k --stage dry-run

To execute the complete cached/resumable pipeline later:

    python -m src.run_8k --with-opus

Stages are idempotent and can be invoked independently with ``--stage``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from src.progress import format_duration

BASE = "Qwen/Qwen2.5-3B-Instruct"

PROFILES = {
    "8k": {
        "name": "brazil8k",
        "years": "2004-2019,2021-2023",
        "enem_years": "2009-2023",
        "skip_enem": False,
        "train": 8000, "val": 200, "test": 400,
        "epochs": 3, "max_len": 1024,
        "estimates": {
            "download_and_build": "18-48 hours first run; 20-60 minutes from cache",
            "qwen3b_qlora_3_epochs": "12-24 hours on RTX 4070 Laptop 8 GB",
            "opus_400_items": "0.5-1.5 hours, gateway latency dependent",
            "local_eval_and_temperature": "0.5-2 hours",
            "complete_first_run": "32-76 hours",
            "cached_rerun_without_opus": "13-26 hours",
        },
        "stage_hours": {
            "data": (18, 48), "train": (12, 24),
            "frontier": (0.5, 1.5), "evaluate": (0.5, 2),
        },
    },
    "overnight": {
        "name": "brazil2000",
        "years": "2016-2019,2021-2023",
        "enem_years": "",
        "skip_enem": True,
        "train": 2000, "val": 100, "test": 250,
        "epochs": 2, "max_len": 768,
        "estimates": {
            "download_and_build": "2-6 hours first run; 10-30 minutes from cache",
            "qwen3b_qlora_2_epochs": "2.5-4.5 hours on RTX 4070 Laptop 8 GB",
            "opus_250_items": "0.25-0.8 hours, gateway latency dependent",
            "local_eval_and_temperature": "0.4-1 hour",
            "complete_first_run": "5.5-12.5 hours (typically about 7-9)",
            "cached_rerun_without_opus": "3-6 hours",
        },
        "stage_hours": {
            "data": (2, 6), "train": (2.5, 4.5),
            "frontier": (0.25, 0.8), "evaluate": (0.4, 1),
        },
    },
    "staged1k": {
        "name": "brazil1p5b_1k",
        "data_name": "brazil1000",
        "base": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_name": "qwen1p5b_1k",
        "years": "2016-2019,2021-2023",
        "enem_years": "",
        "skip_enem": True,
        "allow_val_year_train": True,
        "train": 1000, "val": 100, "test": 250,
        "epochs": 2, "max_len": 1024, "load_4bit": False,
        "estimates": {
            "download_and_build": "1-4 hours first run; cached work is reused later",
            "qwen1p5b_2_epochs": "1-2 hours on RTX 4070 Laptop 8 GB",
            "opus_250_items": "0.25-0.8 hours, gateway latency dependent",
            "local_eval_and_temperature": "0.3-0.8 hours",
            "complete_first_run": "about 3.5-8 hours (typically 5-6)",
        },
        "stage_hours": {
            "data": (1, 4), "train": (1, 2),
            "frontier": (0.25, 0.8), "evaluate": (0.3, 0.8),
        },
    },
    "staged2k": {
        "name": "brazil1p5b_2k",
        "data_name": "brazil2000",
        "base": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_name": "qwen1p5b_2k",
        "years": "2016-2019,2021-2023",
        "enem_years": "",
        "skip_enem": True,
        "train": 2000, "val": 100, "test": 250,
        "epochs": 2, "max_len": 1024, "load_4bit": False,
        "incremental_from": "brazil1p5b_1k",
        "init_adapter": "out/brazil1p5b_1k/qwen1p5b_1k",
        "estimates": {
            "additional_data_build": "1-3 hours; reuses stage-1 downloads and caches",
            "incremental_qwen1p5b": "1.2-2.5 hours for 1,000 new + 250 replay items",
            "local_eval_and_temperature": "0.3-0.8 hours",
            "additional_stage_total": "about 2.5-7 hours",
        },
        "stage_hours": {
            "data": (1, 3), "train": (1.2, 2.5),
            "frontier": (0, 0.1), "evaluate": (0.3, 0.8),
        },
    },
}

RUN_DIR = Path()
LOG_DIR = Path()
STATE_PATH = Path()
DATA_PREFIX = ""
SWEEP_DIR = ""
ADAPTER = ""
RESULTS = ""
ESTIMATES = {}
DATASET_NAME = ""
STAGE_HOURS = {}
BASE = "Qwen/Qwen2.5-3B-Instruct"
CONTROL_PATH = Path()
TRAIN_FILE = ""
EVAL_TRAIN_FILE = ""
VAL_FILE = ""
TEST_FILE = ""


def _configure(profile):
    global RUN_DIR, LOG_DIR, STATE_PATH, DATA_PREFIX  # noqa: PLW0603
    global SWEEP_DIR, ADAPTER, RESULTS, ESTIMATES, DATASET_NAME  # noqa: PLW0603
    global STAGE_HOURS  # noqa: PLW0603
    global BASE, CONTROL_PATH, TRAIN_FILE, EVAL_TRAIN_FILE  # noqa: PLW0603
    global VAL_FILE, TEST_FILE  # noqa: PLW0603
    name = profile["name"]
    data_name = profile.get("data_name", name)
    RUN_DIR = Path("runs") / name
    LOG_DIR = RUN_DIR / "logs"
    STATE_PATH = RUN_DIR / "state.json"
    DATA_PREFIX = f"data/{data_name}"
    SWEEP_DIR = f"out/{name}"
    ADAPTER = f"{SWEEP_DIR}/{profile.get('adapter_name', 'qwen3b')}"
    RESULTS = f"results/{name}"
    ESTIMATES = profile["estimates"]
    STAGE_HOURS = profile["stage_hours"]
    BASE = profile.get("base", "Qwen/Qwen2.5-3B-Instruct")
    CONTROL_PATH = RUN_DIR / "control.json"
    TRAIN_FILE = f"{DATA_PREFIX}_train.jsonl"
    EVAL_TRAIN_FILE = TRAIN_FILE
    VAL_FILE = f"{DATA_PREFIX}_val.jsonl"
    TEST_FILE = f"{DATA_PREFIX}_test.jsonl"
    if profile.get("incremental_from"):
        TRAIN_FILE = f"{DATA_PREFIX}_incremental_train.jsonl"
        EVAL_TRAIN_FILE = f"{DATA_PREFIX}_seen_train.jsonl"
        VAL_FILE = "data/brazil1000_val.jsonl"
        TEST_FILE = "data/brazil1000_test.jsonl"
    DATASET_NAME = (
        f"Brazil real distributions ({profile['train']} cumulative train items)")


def _load_state():
    if STATE_PATH.is_file():
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"stages": {}}


def _save_state(state):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def _done(stage):
    checks = {
        "data": [
            Path(f"{DATA_PREFIX}_train.jsonl"),
            Path(f"{DATA_PREFIX}_val.jsonl"),
            Path(f"{DATA_PREFIX}_test.jsonl"),
            Path(f"{DATA_PREFIX}_manifest.json"),
            Path(f"{DATA_PREFIX}_qc.json"),
        ],
        "train": [Path(ADAPTER) / "adapter_model.safetensors"],
        "frontier": [Path(RESULTS) / "preds_opus.json"],
        "evaluate": [
            Path(RESULTS) / "results_tuned.json",
            Path(RESULTS) / "report_tuned.md",
        ],
    }
    if stage == "qc":
        path = Path(f"{DATA_PREFIX}_qc.json")
        if not path.is_file():
            return False
        try:
            with open(path, encoding="utf-8") as f:
                return bool(json.load(f).get("qc_passed"))
        except (OSError, json.JSONDecodeError):
            return False
    return stage in checks and all(p.is_file() for p in checks[stage])


def _record_interruption(stage, started, command, log_path, elapsed, code=130):
    state = _load_state()
    prior = state["stages"].get(stage, {})
    state["stages"][stage] = {
        **prior,
        "status": "interrupted",
        "started": started,
        "ended": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": code,
        "command": command,
        "log": str(log_path),
    }
    _save_state(state)


def _run(
        stage, command, force=False, part_index=1, part_total=1,
        mark_part_complete=True):
    print(
        f"[pipeline] {100*(part_index-1)/part_total:.1f}% parts complete | "
        f"part {part_index}/{part_total}: {stage}", flush=True)
    if _done(stage) and not force:
        print(f"[{stage}] already complete; use --force to rerun")
        if mark_part_complete:
            print(
                f"[pipeline] {100*part_index/part_total:.1f}% parts complete",
                flush=True)
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{stage}.log"
    state = _load_state()
    started = datetime.now(timezone.utc).isoformat()
    state["stages"][stage] = {
        "status": "running", "started": started, "command": command}
    _save_state(state)
    print(f"[{stage}] {' '.join(command)}", flush=True)
    t0 = time.time()
    proc = None
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n\n[{started}] {' '.join(command)}\n")
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
            state = _load_state()
            state["stages"][stage]["pid"] = proc.pid
            state["stages"][stage]["log"] = str(log_path)
            _save_state(state)
            assert proc.stdout is not None
            last_state_save = 0.0
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if "[progress]" in line and time.time() - last_state_save >= 5:
                    state = _load_state()
                    current = state["stages"].get(stage, {})
                    current["last_progress"] = line.strip()
                    match = re.search(r"(\d+(?:\.\d+)?)%", line)
                    if match:
                        current["progress_percent"] = float(match.group(1))
                    state["stages"][stage] = current
                    _save_state(state)
                    last_state_save = time.time()
            code = proc.wait()
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        elapsed = time.time() - t0
        _record_interruption(
            stage, started, command, log_path, elapsed,
            proc.returncode if proc is not None and proc.returncode else 130)
        print(
            f"\n[{stage}] interrupted after {format_duration(elapsed)}. "
            "Run the same command to resume.", flush=True)
        raise SystemExit(130) from None
    elapsed = time.time() - t0
    state = _load_state()
    state["stages"][stage] = {
        "status": (
            "complete" if code == 0 else "paused" if code == 75 else "failed"),
        "started": started,
        "ended": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": code,
        "command": command,
        "log": str(log_path),
    }
    _save_state(state)
    if code == 75:
        print(
            f"[{stage}] paused safely after {format_duration(elapsed)}. "
            "Run the same command to resume.", flush=True)
        raise SystemExit(0)
    if code:
        raise SystemExit(f"[{stage}] failed ({code}); see {log_path}")
    if mark_part_complete:
        print(
            f"[pipeline] {100*part_index/part_total:.1f}% parts complete | "
            f"{stage} took {format_duration(elapsed)}", flush=True)


def _python(*args):
    return [sys.executable, *args]


def _request_pause(after_minutes: float):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    not_before = time.time() + max(0.0, after_minutes) * 60
    with open(CONTROL_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "action": "pause_at_checkpoint",
            "requested_unix": time.time(),
            "not_before_unix": not_before,
            "after_minutes": max(0.0, after_minutes),
        }, f, indent=2)
    when = (
        "the next checkpoint" if after_minutes <= 0 else
        f"the first checkpoint after {after_minutes:g} minutes")
    print(f"Pause requested at {when}. Training will stop only after a safe save.")


def _rollback_checkpoint():
    state = _load_state()
    if state.get("stages", {}).get("train", {}).get("status") == "running":
        raise SystemExit(
            "Training is marked running. Request pause-next and wait for the "
            "safe checkpoint before rolling back.")
    if (Path(ADAPTER) / "adapter_model.safetensors").is_file():
        raise SystemExit(
            "Training is already finalized. Rollback is only enabled for a "
            "paused, incomplete run; no files were changed.")
    checkpoint_root = Path(ADAPTER) / "checkpoints"
    checkpoints = sorted(
        checkpoint_root.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]))
    if len(checkpoints) < 2:
        raise SystemExit(
            "Rollback needs at least two retained checkpoints; none were changed.")
    latest = checkpoints[-1]
    archive = Path(ADAPTER) / "archived-checkpoints"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / (
        f"{latest.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    shutil.move(str(latest), str(destination))
    state["stages"]["train"] = {
        **state.get("stages", {}).get("train", {}),
        "status": "rolled_back",
        "active_checkpoint": str(checkpoints[-2]),
        "archived_checkpoint": str(destination),
    }
    _save_state(state)
    print(f"Archived {latest}; next resume starts from {checkpoints[-2]}.")


def _prepare_incremental(profile):
    if not profile.get("incremental_from"):
        return
    stage1_train = "data/brazil1000_train.jsonl"
    expanded_train = f"{DATA_PREFIX}_train.jsonl"
    command = _python(
        "-m", "src.prepare_incremental",
        "--stage1-train", stage1_train,
        "--expanded-train", expanded_train,
        "--out", TRAIN_FILE,
        "--seen-out", EVAL_TRAIN_FILE,
        "--new-items", "1000",
        "--replay-items", "250")
    subprocess.run(command, check=True)
    prior_opus = Path("results/brazil1p5b_1k/preds_opus.json")
    current_opus = Path(RESULTS) / "preds_opus.json"
    if prior_opus.is_file() and not current_opus.is_file():
        current_opus.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prior_opus, current_opus)
        print(f"reused fixed-test Opus predictions from {prior_opus}")


def print_status():
    print("Pre-run time estimates (RTX 4070 Laptop 8 GB):")
    for name, estimate in ESTIMATES.items():
        print(f"  {name.replace('_', ' ')}: {estimate}")
    state = _load_state()
    print("\nStages:")
    for stage in ("data", "train", "frontier", "evaluate"):
        record = state.get("stages", {}).get(stage, {})
        status = "complete" if _done(stage) else record.get("status", "pending")
        elapsed = record.get("elapsed_seconds")
        if status == "running" and record.get("started"):
            try:
                elapsed = (
                    datetime.now(timezone.utc) -
                    datetime.fromisoformat(record["started"])).total_seconds()
            except ValueError:
                pass
        timing = f" ({format_duration(elapsed)})" if elapsed is not None else ""
        print(f"  {stage:10s} {status}{timing}")
        if record.get("last_progress") and status != "complete":
            print(f"             {record['last_progress']}")
    checkpoints = sorted((Path(ADAPTER) / "checkpoints").glob("checkpoint-*"))
    if checkpoints:
        print(f"\nRetained checkpoints ({len(checkpoints)}):")
        for checkpoint in checkpoints:
            print(f"  {checkpoint}")
    if CONTROL_PATH.is_file():
        with open(CONTROL_PATH, encoding="utf-8") as f:
            control = json.load(f)
        print(f"\nCheckpoint control: {control.get('action', 'unknown')}")
    if STATE_PATH.is_file():
        print(f"\nDetailed state: {STATE_PATH}")


def main():
    # PowerShell may launch Python with a legacy cp1252 console even though
    # subprocess output and logs are UTF-8. Never let display-only characters
    # terminate a multi-hour run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        choices=("all", "dry-run", "data", "train", "frontier",
                 "evaluate", "status", "pause-next", "rollback"),
        default="all")
    ap.add_argument("--profile", choices=tuple(PROFILES), default="8k")
    ap.add_argument("--with-opus", action="store_true",
                    help="run the profile's Opus baseline before final evaluation")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument(
        "--after-minutes", type=float, default=0,
        help="with pause-next, wait this long before pausing at a checkpoint")
    args = ap.parse_args()
    profile = dict(PROFILES[args.profile])
    _configure(profile)
    years = args.years or profile["years"]
    max_len = args.max_len or profile["max_len"]

    if args.stage == "status":
        print_status()
        return
    if args.stage == "pause-next":
        _request_pause(args.after_minutes)
        return
    if args.stage == "rollback":
        _rollback_checkpoint()
        return
    if args.stage == "dry-run":
        command = _python(
            "-m", "src.data_prep_brazil", "--dry-run",
            "--years", years,
            "--enem-years", profile["enem_years"])
        if profile["skip_enem"]:
            command.append("--skip-enem")
        subprocess.run(command, check=True)
        print_status()
        return

    stages = (
        ["data", "train"] +
        (["frontier"] if args.with_opus else []) +
        ["evaluate"]
        if args.stage == "all" else [args.stage]
    )
    part_total = len(stages)
    for part_index, stage in enumerate(stages, start=1):
        if stage != "data":
            _prepare_incremental(profile)
        if stage == "data":
            command = _python(
                "-m", "src.data_prep_brazil",
                "--years", years,
                "--enem-years", profile["enem_years"],
                "--train-items", str(profile["train"]),
                "--val-items", str(profile["val"]),
                "--test-items", str(profile["test"]),
                "--min-responses", "100",
                "--max-len", str(max_len),
                "--out-prefix", DATA_PREFIX)
            if profile["skip_enem"]:
                command.append("--skip-enem")
            if profile.get("allow_val_year_train"):
                command.append("--allow-val-year-train")
            if args.force:
                command.append("--rebuild")
            _run("data", command, args.force, part_index, part_total)
            qc_command = _python(
                "-m", "src.qc_brazil",
                "--prefix", DATA_PREFIX,
                "--tokenizer", BASE,
                "--max-len", str(max_len),
                "--min-responses", "100")
            if profile.get("allow_val_year_train"):
                qc_command.append("--allow-val-year-train")
            _run("qc", qc_command,
                args.force, part_index, part_total,
                mark_part_complete=False)
            _prepare_incremental(profile)
        elif stage == "train":
            if not _done("data"):
                raise SystemExit(
                    "prepared data has not passed deterministic QC; "
                    "run the data stage first")
            command = _python(
                "-m", "src.train",
                "--train", TRAIN_FILE,
                "--base", BASE,
                "--out", ADAPTER,
                "--epochs", str(profile["epochs"]),
                "--batch", "1",
                "--grad-accum", "16",
                "--lr", "1e-4",
                "--lora-r", "16",
                "--max-len", str(max_len),
                "--options", "5",
                "--save-steps", "25",
                "--save-total-limit", "2",
                "--control-file", str(CONTROL_PATH))
            if profile.get("init_adapter"):
                command.extend(["--init-adapter", profile["init_adapter"]])
            if profile.get("load_4bit", True):
                command.append("--load-4bit")
            if (Path(ADAPTER) / "checkpoints").is_dir() and not args.force:
                command.append("--resume")
            _run("train", command, args.force, part_index, part_total)
        elif stage == "frontier":
            _run("frontier", _python(
                "-m", "src.predict_frontier",
                "--model", "claude-opus-4-8",
                "--tag", "opus",
                "--test", TEST_FILE,
                "--train", EVAL_TRAIN_FILE,
                "--out-dir", RESULTS), args.force, part_index, part_total)
        elif stage == "evaluate":
            frontier = "opus" if (Path(RESULTS) / "preds_opus.json").is_file() else ""
            # First compute source-aware mean/uniform and load frontier predictions.
            _run("evaluate-baselines", _python(
                "-m", "src.run_experiment",
                "--test", TEST_FILE,
                "--train", EVAL_TRAIN_FILE,
                "--frontier", frontier,
                "--adapter", "",
                "--base", BASE,
                "--out-dir", RESULTS,
                "--dataset-name", DATASET_NAME), args.force,
                 part_index, part_total, mark_part_complete=False)
            # Then calibrate the adapter on val and produce the final leaderboard.
            _run("evaluate", _python(
                "-m", "src.tune",
                "--val", VAL_FILE,
                "--test", TEST_FILE,
                "--sweep-dir", SWEEP_DIR,
                "--frontier-json", f"{RESULTS}/results.json",
                "--out-dir", RESULTS,
                "--dataset-name", DATASET_NAME,
                "--max-len", str(max_len)) +
                (["--load-4bit"] if profile.get("load_4bit", True) else []),
                args.force,
                 part_index, part_total)
    print_status()


if __name__ == "__main__":
    main()
