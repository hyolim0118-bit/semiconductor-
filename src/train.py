"""CPU-only STEP 9 training. Default execution runs exactly one epoch."""

import argparse
import csv
import ctypes
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from .config import HIDDEN_DIMS, INPUT_DIM, OUTPUT_DIM
from .model import MLPRegressor, count_trainable_parameters


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "loss": "L1Loss", "optimizer": "Adam", "learning_rate": 0.001,
    "batch_size": 1024, "max_epochs": 100, "patience": 10,
    "seed": 42, "target_scaling": False, "device": "cpu",
    "input_dim": INPUT_DIM, "hidden_dims": list(HIDDEN_DIMS),
    "output_dim": OUTPUT_DIM,
}


class MappedBatches(Dataset):
    """Read saved arrays without loading the full dataset into Python memory.

    The sampler supplies lists of row indices; each access copies one batch.
    No worker processes are used, avoiding Windows process/mapping duplication.
    """

    def __init__(self, directory, split, expected_rows):
        self.x = np.load(str(directory / ("X_{}_scaled.npy".format(split))),
                         mmap_mode="r", allow_pickle=False)
        self.y = np.load(str(directory / ("y_{}.npy".format(split))),
                         mmap_mode="r", allow_pickle=False)
        if self.x.shape != (expected_rows, INPUT_DIM) or self.y.shape != (expected_rows, OUTPUT_DIM):
            raise ValueError("{}: unexpected X/y shapes: {}, {}".format(
                split, self.x.shape, self.y.shape))
        if self.x.dtype != np.float32 or self.y.dtype != np.float32:
            raise ValueError("{}: X and y must be float32".format(split))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indices):
        return (torch.from_numpy(self.x[indices].copy()),
                torch.from_numpy(self.y[indices].copy()))


def make_loader(dataset, shuffle, generator):
    sampler = (RandomSampler(dataset, generator=generator) if shuffle
               else SequentialSampler(dataset))
    batches = BatchSampler(sampler, SETTINGS["batch_size"], drop_last=False)
    return DataLoader(dataset, batch_size=None, sampler=batches, num_workers=0,
                      pin_memory=False, generator=generator)


def memory_usage():
    """Windows working set, including native tensors and mapped resident pages."""
    if os.name != "nt":
        return {"measurement": "unavailable (Windows API only)"}

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"working_set_mib": counters.WorkingSetSize / 2**20,
            "peak_working_set_mib": counters.PeakWorkingSetSize / 2**20,
            "private_commit_mib": counters.PagefileUsage / 2**20}


def smoke_update(dataset):
    """Use a disposable model so the real run starts with fresh Adam state."""
    model = MLPRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
    x, y = dataset[list(range(32))]
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss = nn.L1Loss()(prediction, y)
    if prediction.shape != y.shape or not torch.isfinite(loss):
        raise RuntimeError("Small-batch forward/loss check failed")
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Small-batch gradients are missing or nonfinite")
    optimizer.step()
    changed = any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    if not changed or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Adam update failed")
    return {"rows": 32, "loss": loss.item(), "finite_gradients": True,
            "parameters_changed": changed}


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    absolute_error, elements, rows = 0.0, 0, 0
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for x, y in loader:
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError("Training/validation batch contains NaN or Inf")
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            if training:
                loss.backward()
                optimizer.step()
            absolute_error += loss.item() * y.numel()
            elements += y.numel()
            rows += len(y)
    if rows != len(loader.dataset):
        raise RuntimeError("Epoch did not cover every row exactly once")
    return absolute_error / elements, time.perf_counter() - started


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def plot_history(history, path, resume_epoch=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for key, label in (("training_mae", "Training (during updates)"),
                       ("validation_mae", "Validation")):
        ax.plot([row["epoch"] for row in history], [row[key] for row in history],
                marker="o", label=label)
    if resume_epoch is not None:
        ax.axvline(resume_epoch, color="gray", linestyle="--", label="Resume boundary")
    ax.set(xlabel="Epoch", ylabel="MAE (original target units)", title="MLP learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def run_training(full=False, threads=None, resume=None):
    threads = min(4, os.cpu_count() or 1) if threads is None else threads
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    random.seed(SETTINGS["seed"])
    np.random.seed(SETTINGS["seed"])
    torch.manual_seed(SETTINGS["seed"])
    directory = ROOT / "outputs" / "preprocessed"
    with (ROOT / "outputs" / "preprocessors" / "preprocessing_config.json").open(
            encoding="utf-8") as stream:
        preprocessing = json.load(stream)
    if preprocessing["target_scaling"] is not False:
        raise ValueError("STEP 9 requires unscaled targets")
    if preprocessing["feature_columns"] != [str(i) for i in range(INPUT_DIM)] or \
            preprocessing["target_columns"] != ["layer_{}".format(i) for i in range(1, OUTPUT_DIM + 1)]:
        raise ValueError("Preprocessing column order does not match the model")
    train = MappedBatches(directory, "train", 648000)
    validation = MappedBatches(directory, "validation", 81000)
    # Track source artifacts without hashing or copying the large source CSVs.
    source_paths = list(ROOT.glob("*.csv"))
    for subdir in ("preprocessed", "preprocessors", "splits"):
        source_paths.extend(p for p in (ROOT / "outputs" / subdir).rglob("*") if p.is_file())
    def snapshot():
        return {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns]
                for p in source_paths}
    before = snapshot()
    parent_history = []
    checkpoint = None
    start_epoch = 0
    if resume is not None:
        resume = Path(resume).resolve()
        checkpoint = torch.load(str(resume), map_location="cpu", weights_only=True)
        previous_config = checkpoint["config"]
        for key in SETTINGS:
            if key != "max_epochs" and previous_config[key] != SETTINGS[key]:
                raise ValueError("Resume setting mismatch: " + key)
        if previous_config["source_artifacts"] != before:
            raise ValueError("Source artifacts differ from the original run")
        with (resume.parent / "history.csv").open(encoding="utf-8", newline="") as stream:
            parent_history = [{key: float(value) for key, value in row.items()}
                              for row in csv.DictReader(stream)]
        start_epoch = int(checkpoint["epoch"])
        if start_epoch != 100 or int(parent_history[-1]["epoch"]) != start_epoch:
            raise ValueError("STEP 9-4 requires the best checkpoint at the completed epoch 100")
        source_paths.extend(p for p in resume.parent.rglob("*") if p.is_file())
        before = snapshot()
    mode = "full" if full else "smoke_1epoch"
    if checkpoint is not None:
        mode = "resume_" + mode
    run_dir = ROOT / "outputs" / "training" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + mode + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    config = dict(SETTINGS, mode=mode, epochs_to_run=100 if full else 1,
                  threads=threads, num_workers=0, python=__import__("platform").python_version(),
                  torch_version=str(torch.__version__), train_rows=len(train), validation_rows=len(validation),
                  trainable_parameters=count_trainable_parameters(MLPRegressor()),
                  training_mae_definition="element-weighted MAE during optimizer updates",
                  source_artifacts=before)
    if checkpoint is not None:
        config.update(max_epochs=200, resume_checkpoint=str(resume), start_epoch=start_epoch,
                      rng_note="Original RNG and DataLoader shuffle states were not saved; "
                               "seeded restart does not reproduce uninterrupted training.")
    save_json(run_dir / "config.json", config)
    print("Run directory:", run_dir.relative_to(ROOT).as_posix(), flush=True)
    smoke = smoke_update(train)
    save_json(run_dir / "smoke_check.json", smoke)
    print("Forward/backward/Adam check:", smoke, flush=True)
    # Reset seed after the disposable smoke model; full and one-epoch runs start alike.
    torch.manual_seed(SETTINGS["seed"])
    model = MLPRegressor().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
    criterion = nn.L1Loss()
    generator = torch.Generator().manual_seed(SETTINGS["seed"])
    train_loader = make_loader(train, True, generator)
    validation_loader = make_loader(validation, False, torch.Generator().manual_seed(42))
    history, best, best_epoch, stale = [], math.inf, 0, 0
    baseline_best = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if len(optimizer.state) != len(list(model.parameters())):
            raise RuntimeError("Adam state missing for some parameters")
        for parameter, state in optimizer.state.items():
            if state["step"].item() <= 0:
                raise RuntimeError("Invalid Adam step")
            for name in ("exp_avg", "exp_avg_sq"):
                if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                    raise RuntimeError("Invalid Adam moment: " + name)
        if any(group["lr"] != SETTINGS["learning_rate"] for group in optimizer.param_groups):
            raise ValueError("Restored Adam learning rate differs")
        best = baseline_best = float(checkpoint["validation_mae"])
        best_epoch = start_epoch
        initial_mae, initial_seconds = run_epoch(model, validation_loader, criterion)
        if not math.isclose(initial_mae, best, rel_tol=0, abs_tol=1e-7):
            raise RuntimeError("Resume validation MAE differs from saved checkpoint")
        verification = dict(model_loaded=True, optimizer_loaded=True,
                            optimizer_parameter_states=len(optimizer.state),
                            adam_steps=sorted({float(s["step"].item()) for s in optimizer.state.values()}),
                            learning_rate=optimizer.param_groups[0]["lr"],
                            saved_validation_mae=best, restored_validation_mae=initial_mae,
                            absolute_difference=abs(initial_mae - best),
                            validation_seconds=initial_seconds, rng_note=config["rng_note"])
        save_json(run_dir / "resume_verification.json", verification)
        print(json.dumps(verification), flush=True)
        # Preserve the inherited best even if no additional epoch improves it.
        shutil.copyfile(str(resume), str(run_dir / "best.pt"))
    memory_before = memory_usage()
    started = time.perf_counter()
    fields = ["epoch", "training_mae", "validation_mae", "train_seconds",
              "validation_seconds", "epoch_seconds", "working_set_mib", "peak_working_set_mib"]
    with (run_dir / "history.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(start_epoch + 1, start_epoch + config["epochs_to_run"] + 1):
            epoch_started = time.perf_counter()
            train_mae, train_seconds = run_epoch(model, train_loader, criterion, optimizer)
            val_mae, val_seconds = run_epoch(model, validation_loader, criterion)
            improved = val_mae < best
            if improved:
                best, best_epoch, stale = val_mae, epoch, 0
                # Only this fresh run's best checkpoint is replaced on improvement.
                temporary = run_dir / "best.tmp"
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "validation_mae": best, "config": config}, str(temporary))
                temporary.replace(run_dir / "best.pt")
            else:
                stale += 1
            memory = memory_usage()
            row = dict(epoch=epoch, training_mae=train_mae, validation_mae=val_mae,
                       train_seconds=train_seconds, validation_seconds=val_seconds,
                       epoch_seconds=time.perf_counter() - epoch_started,
                       working_set_mib=memory.get("working_set_mib"),
                       peak_working_set_mib=memory.get("peak_working_set_mib"))
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            if full and stale >= SETTINGS["patience"]:
                break
    elapsed = time.perf_counter() - started
    # Verify that the saved best model reloads and reproduces its validation MAE.
    checkpoint = torch.load(str(run_dir / "best.pt"), map_location="cpu", weights_only=True)
    restored = MLPRegressor()
    restored.load_state_dict(checkpoint["model_state_dict"])
    reloaded_mae, reload_seconds = run_epoch(restored, validation_loader, criterion)
    if not math.isclose(reloaded_mae, best, rel_tol=1e-7, abs_tol=1e-7):
        raise RuntimeError("Reloaded checkpoint validation MAE differs")
    if snapshot() != before:
        raise RuntimeError("Source artifact size/mtime changed")
    plot_history(parent_history + history, run_dir / "learning_curve.png", start_epoch or None)
    if parent_history:
        with (run_dir / "combined_history.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(parent_history + history)
    summary = dict(mode=mode, epochs_completed=len(history), best_epoch=best_epoch,
                   best_validation_mae=best, stopped_early=full and stale >= SETTINGS["patience"],
                   training_loop_seconds=elapsed, first_epoch=history[0],
                   estimated_100_epoch_seconds=history[0]["epoch_seconds"] * 100,
                   estimate_note="Linear estimate; CPU load/cache and early stopping affect actual time",
                   memory_before=memory_before, memory_after=memory_usage(),
                   checkpoint_reload_validation_mae=reloaded_mae,
                   checkpoint_reload_seconds=reload_seconds,
                   source_sizes_and_mtimes_unchanged=True,
                   holdout_used=False)
    if baseline_best is not None:
        summary.update(resume_epoch=start_epoch, final_epoch=history[-1]["epoch"],
                       baseline_best_validation_mae=baseline_best,
                       improvement_percent=(baseline_best - best) / baseline_best * 100,
                       rng_note=config["rng_note"])
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Train up to 100 epochs with early stopping")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default: up to 4)")
    parser.add_argument("--resume", type=Path, help="Resume epoch-100 best.pt; default: one additional epoch; --full: up to epoch 200")
    args = parser.parse_args()
    run_training(full=args.full, threads=args.threads, resume=args.resume)
