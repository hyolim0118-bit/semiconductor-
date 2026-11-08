"""Independent STEP 9-5 CPU continuations from the epoch-200 checkpoint."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch import nn

from .model import MLPRegressor
from .train import ROOT, SETTINGS, MappedBatches, make_loader, memory_usage, plot_history, run_epoch, save_json


RATES = (0.001, 0.0003, 0.0001)
RNG_NOTE = ("Original RNG/shuffle states unavailable. Each candidate resets Python, NumPy, "
            "Torch and the loader generator to seed 42, sharing new shuffle sequences. "
            "This is a single-seed comparison, not proof of a general LR-only effect.")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(path, learning_rate):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != 200:
        raise ValueError("Expected epoch-200 checkpoint")
    for key, value in SETTINGS.items():
        if key != "max_epochs" and checkpoint["config"][key] != value:
            raise ValueError("Checkpoint setting mismatch: " + key)
    model = MLPRegressor()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError("Incomplete Adam state")
    for parameter, state in optimizer.state.items():
        if state["step"].item() <= 0:
            raise ValueError("Invalid Adam step")
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].shape != parameter.shape or not torch.isfinite(state[key]).all():
                raise ValueError("Invalid Adam moments")
    # load_state_dict restores the old LR, so override only AFTER restoring.
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return model, optimizer, checkpoint


def validation_metrics(model, loader):
    overall, _ = run_epoch(model, loader, nn.L1Loss())
    model.eval()
    errors = torch.zeros(4, dtype=torch.float64)
    rows = 0
    with torch.no_grad():
        for x, y in loader:
            errors += (model(x) - y).abs().double().sum(dim=0)
            rows += len(y)
    layers = (errors / rows).tolist()
    if not math.isclose(overall, sum(layers) / 4, abs_tol=1e-5):
        raise RuntimeError("Layer/overall MAE mismatch")
    return dict(validation_mae=overall, **{"layer_{}_mae".format(i + 1): v for i, v in enumerate(layers)})


def compare(checkpoint_path, full=False):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    checkpoint_path = checkpoint_path.resolve()
    source = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    # Check source identities using metadata only. Never open hold-out arrays.
    tracked = {ROOT / p: metadata for p, metadata in source["config"]["source_artifacts"].items()}
    for path, expected in tracked.items():
        if [path.stat().st_size, path.stat().st_mtime_ns] != expected:
            raise ValueError("Source artifact changed: " + str(path))
    tracked.update({p: [p.stat().st_size, p.stat().st_mtime_ns]
                    for p in checkpoint_path.parent.rglob("*") if p.is_file()})
    original_hash = digest(checkpoint_path)
    run_dir = ROOT / "outputs" / "lr_comparison" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + ("_full_" if full else "_smoke_") + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    config = dict(SETTINGS, max_epochs=250, additional_epochs=50 if full else 1,
                  learning_rates=list(RATES), checkpoint=str(checkpoint_path), checkpoint_sha256=original_hash,
                  threads=4, rng_note=RNG_NOTE, holdout_used=False,
                  torch_version=str(torch.__version__), python=__import__("platform").python_version())
    save_json(run_dir / "config.json", config)
    train = MappedBatches(ROOT / "outputs/preprocessed", "train", 648000)
    validation = MappedBatches(ROOT / "outputs/preprocessed", "validation", 81000)
    val_loader = make_loader(validation, False, torch.Generator().manual_seed(42))
    baseline_model, baseline_optimizer, _ = restore(checkpoint_path, 0.001)
    baseline = validation_metrics(baseline_model, val_loader)
    if not math.isclose(baseline["validation_mae"], source["validation_mae"], rel_tol=0, abs_tol=1e-7):
        raise RuntimeError("Baseline checkpoint MAE not reproduced")
    save_json(run_dir / "baseline.json", dict(baseline, epoch=200, checkpoint=str(checkpoint_path),
              adam_states=len(baseline_optimizer.state),
              adam_steps=sorted({float(s["step"].item()) for s in baseline_optimizer.state.values()})))
    print("Comparison directory: " + run_dir.relative_to(ROOT).as_posix(), flush=True)
    print("Baseline: " + json.dumps(baseline), flush=True)
    results = [dict(name="baseline", learning_rate=0.001, best_epoch=200,
                    stopped_early=False, epochs_completed=0, training_seconds=0,
                    improvement_percent=0, checkpoint=str(checkpoint_path), **baseline)]
    for rate in RATES:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model, optimizer, checkpoint = restore(checkpoint_path, rate)
        loader = make_loader(train, True, torch.Generator().manual_seed(42))
        candidate_dir = run_dir / ("lr_" + str(rate))
        candidate_dir.mkdir()
        candidate_config = dict(config, learning_rate=rate)
        save_json(candidate_dir / "config.json", candidate_config)
        initial, _ = run_epoch(model, val_loader, nn.L1Loss())
        if initial != baseline["validation_mae"]:
            raise RuntimeError("Candidate did not start from baseline")
        save_json(candidate_dir / "restore_verification.json", dict(initial_validation_mae=initial,
                  learning_rate=optimizer.param_groups[0]["lr"], adam_states=len(optimizer.state),
                  checkpoint_sha256=original_hash))
        best, best_epoch, stale = initial, 200, 0
        # Candidate config records the explicit LR even if baseline remains best.
        checkpoint["config"] = candidate_config
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, str(candidate_dir / "best.pt"))
        history = []
        started = time.perf_counter()
        fields = ["epoch", "training_mae", "validation_mae", "train_seconds", "validation_seconds", "epoch_seconds"]
        with (candidate_dir / "history.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(201, 201 + config["additional_epochs"]):
                epoch_start = time.perf_counter()
                training_mae, train_seconds = run_epoch(model, loader, nn.L1Loss(), optimizer)
                val_mae, val_seconds = run_epoch(model, val_loader, nn.L1Loss())
                if val_mae < best:
                    best, best_epoch, stale = val_mae, epoch, 0
                    torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                                    optimizer_state_dict=optimizer.state_dict(), validation_mae=best,
                                    config=candidate_config), str(candidate_dir / "best.pt"))
                else:
                    stale += 1
                row = dict(epoch=epoch, training_mae=training_mae, validation_mae=val_mae,
                           train_seconds=train_seconds, validation_seconds=val_seconds,
                           epoch_seconds=time.perf_counter() - epoch_start)
                history.append(row)
                writer.writerow(row)
                stream.flush()
                print(json.dumps(dict(learning_rate=rate, **row)), flush=True)
                if stale >= 10:
                    break
        elapsed = time.perf_counter() - started
        best_checkpoint = torch.load(str(candidate_dir / "best.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        metrics = validation_metrics(model, val_loader)
        if not math.isclose(metrics["validation_mae"], best, rel_tol=0, abs_tol=1e-7):
            raise RuntimeError("Best checkpoint reload mismatch")
        plot_history([dict(epoch=200, training_mae=float("nan"), validation_mae=initial)] + history,
                     candidate_dir / "learning_curve.png", 200)
        result = dict(name=candidate_dir.name, learning_rate=rate, best_epoch=best_epoch,
                      stopped_early=stale >= 10, epochs_completed=len(history), training_seconds=elapsed,
                      improvement_percent=(initial - best) / initial * 100,
                      checkpoint=str(candidate_dir / "best.pt"), memory=memory_usage(), **metrics)
        save_json(candidate_dir / "summary.json", result)
        results.append(result)
    for path, expected in tracked.items():
        if [path.stat().st_size, path.stat().st_mtime_ns] != expected:
            raise RuntimeError("Protected source modified: " + str(path))
    if digest(checkpoint_path) != original_hash:
        raise RuntimeError("Original checkpoint modified")
    winner = min(results, key=lambda row: row["validation_mae"])
    save_json(run_dir / "summary.json", dict(results=results, winner=winner,
              protected_sources_unchanged=True, holdout_used=False, rng_note=RNG_NOTE))
    fields = [key for key in results[0]]
    with (run_dir / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for result in results[1:]:
        with (Path(result["checkpoint"]).parent / "history.csv").open(encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        ax.plot([200] + [int(r["epoch"]) for r in history],
                [baseline["validation_mae"]] + [float(r["validation_mae"]) for r in history],
                label="LR " + str(result["learning_rate"]))
    ax.axhline(baseline["validation_mae"], linestyle="--", color="gray", label="Epoch 200 baseline")
    ax.set(xlabel="Epoch", ylabel="Validation MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(run_dir / "comparison.png"), dpi=150)
    plt.close(fig)
    print("Completed: " + run_dir.relative_to(ROOT).as_posix(), flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="50 additional epochs per LR; default: 1")
    args = parser.parse_args()
    compare(args.checkpoint, args.full)
