"""STEP 9-6: check six architectures, smoke B/E only, or explicitly train A-F."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import uuid
import traceback

import numpy as np
import torch
from torch import nn

from .architecture_models import ResidualBlock, build_model
from .model import count_trainable_parameters
from .train import ROOT, MappedBatches, make_loader, memory_usage, plot_history, run_epoch, save_json


DEFAULT_CONFIG = ROOT / "configs" / "architecture_comparison.json"
PAPER = "https://arxiv.org/html/2106.11959v3#S3"
OFFICIAL = "https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    training = config["training"]
    for key, value in dict(device="cpu", loss="L1Loss", optimizer="Adam", batch_size=1024,
                           target_scaling=False).items():
        if training[key] != value:
            raise ValueError("Unsupported training setting: " + key)
    if (config["input_dim"], config["output_dim"]) != (226, 4):
        raise ValueError("Saved preprocessing requires 226 inputs and 4 targets")
    if [spec["id"] for spec in config["models"]] != list("ABCDEF"):
        raise ValueError("Expected candidates A-F in order")
    for key in ("max_epochs", "patience", "threads"):
        if training[key] < 1:
            raise ValueError(key + " must be positive")
    if not math.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite")
    return config


def protected_manifest():
    # Explicit paths: no listing, stat, mapping or opening of hold-out arrays.
    paths = [ROOT / "train.csv", ROOT / "src/model.py", ROOT / "src/train.py",
             ROOT / "src/config.py", ROOT / "src/lr_compare.py",
             ROOT / "outputs/splits/data_split_indices.npz"]
    paths += [ROOT / "outputs/preprocessed" / name for name in (
        "X_train_scaled.npy", "y_train.npy", "X_validation_scaled.npy", "y_validation.npy")]
    paths += list((ROOT / "outputs/preprocessors").glob("*"))
    for directory in ("training", "lr_comparison"):
        paths += [p for p in (ROOT / "outputs" / directory).rglob("*") if p.is_file()]
    manifest = {}
    for path in paths:
        if path.is_file():
            stat = path.stat()
            record = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if stat.st_size < 10 * 2**20:
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[path.relative_to(ROOT).as_posix()] = record
    return manifest


def check_model(spec, config):
    """Disposable synthetic check/benchmark; never reused for actual training."""
    seed_all(config["training"]["seed"])
    model = build_model(spec, config["input_dim"], config["output_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    x, y = torch.randn(32, 226), torch.randn(32, 4)
    norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm1d)]
    before_counts = [module.num_batches_tracked.item() for module in norms]
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.train()
    prediction = model(x)
    assert prediction.shape == (32, 4) and torch.isfinite(prediction).all()
    loss = nn.L1Loss()(prediction, y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(module.num_batches_tracked.item() == count + 1
               for module, count in zip(norms, before_counts))
    dropout_active = any(isinstance(module, nn.Dropout) and module.p > 0 for module in model.modules())
    if dropout_active:
        assert not torch.equal(model(x), model(x))
    model.eval()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    with torch.no_grad():
        assert torch.equal(model(x), model(x))
        for size in (1, 32, 1024):
            output = model(torch.randn(size, 226))
            assert output.shape == (size, 4) and torch.isfinite(output).all()
    assert all(torch.equal(buffers[name], value) for name, value in model.named_buffers())
    skip_verified = None
    if spec["kind"] == "residual":
        block = ResidualBlock(spec["width"], spec["hidden_factor"],
                              spec["hidden_dropout"], spec["residual_dropout"]).eval()
        with torch.no_grad():
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    module.bias.zero_()
        identity_input = torch.randn(8, spec["width"], requires_grad=True)
        identity_output = block(identity_input)
        assert torch.equal(identity_output, identity_input)
        identity_output.sum().backward()
        assert torch.equal(identity_input.grad, torch.ones_like(identity_input))
        skip_verified = True
    # Full-size synthetic batches estimate relative compute, not accuracy or real epochs.
    model.train()
    x, y = torch.randn(1024, 226), torch.randn(1024, 4)
    times = []
    for index in range(7):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        nn.L1Loss()(model(x), y).backward()
        optimizer.step()
        if index >= 2:
            times.append(time.perf_counter() - started)
    parameters = count_trainable_parameters(model)
    return dict(id=spec["id"], name=spec["name"], parameters=parameters,
                forward_backward_passed=True, eval_shapes=[1, 32, 1024],
                finite_gradients=True, optimizer_update_verified=True,
                batchnorm_modules=len(norms), batchnorm_training_updates_verified=bool(norms),
                eval_buffers_unchanged=True, deterministic_eval=True,
                dropout_training_stochastic=dropout_active, skip_identity_and_gradient=skip_verified,
                synthetic_step_median_seconds=statistics.median(times),
                parameter_gradient_adam_mib=parameters * 16 / 2**20,
                memory=memory_usage(), structure=str(model))


def evaluate(model, loader, prediction_path=None):
    """Element-weighted original-unit MAE and MAPE (%) over every validation row."""
    model.eval()
    absolute = torch.zeros(4, dtype=torch.float64)
    relative = torch.zeros(4, dtype=torch.float64)
    rows, near_zero = 0, [0] * 4
    predictions = np.empty((len(loader.dataset), 4), dtype=np.float32) if prediction_path else None
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x)
            if not torch.isfinite(prediction).all() or not torch.isfinite(y).all():
                raise RuntimeError("Nonfinite validation values")
            if predictions is not None:
                predictions[rows:rows + len(y)] = prediction.numpy()
            target = y.double()
            errors = (prediction.double() - target).abs()
            absolute += errors.sum(dim=0)
            relative += (errors / target.abs().clamp_min(1e-8)).sum(dim=0)
            near_zero = [a + int(b) for a, b in zip(near_zero, (target.abs() < 1e-8).sum(dim=0))]
            rows += len(y)
    if rows != len(loader.dataset):
        raise RuntimeError("Incomplete validation coverage")
    if predictions is not None:
        np.save(str(prediction_path), predictions, allow_pickle=False)
    maes, mapes = (absolute / rows).tolist(), (relative / rows * 100).tolist()
    result = dict(validation_mae=sum(maes) / 4, validation_mape_percent=sum(mapes) / 4,
                  mape_denominator_epsilon=1e-8, mape_near_zero_counts=near_zero)
    for index in range(4):
        result["layer_{}_mae".format(index + 1)] = maes[index]
        result["layer_{}_mape_percent".format(index + 1)] = mapes[index]
    return result


def train_candidate(spec, config, directory, train, validation, epochs):
    training = config["training"]
    seed_all(training["seed"])
    model = build_model(spec, config["input_dim"], config["output_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    generator = torch.Generator().manual_seed(training["seed"])
    train_loader = make_loader(train, True, generator)
    val_loader = make_loader(validation, False, torch.Generator().manual_seed(training["seed"]))
    memory_before = memory_usage()
    best, best_epoch, stale = math.inf, 0, 0
    history = []
    started = time.perf_counter()
    fields = ["epoch", "training_mae", "validation_mae", "train_seconds", "validation_seconds",
              "epoch_seconds", "working_set_mib", "peak_working_set_mib"]
    with (directory / "history.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_mae, train_seconds = run_epoch(model, train_loader, nn.L1Loss(), optimizer)
            val_mae, val_seconds = run_epoch(model, val_loader, nn.L1Loss())
            if val_mae < best:
                best, best_epoch, stale = val_mae, epoch, 0
                temporary = directory / "best.tmp"
                torch.save(dict(epoch=epoch, model_spec=spec, config=config,
                                model_state_dict=model.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(), validation_mae=best,
                                torch_rng_state=torch.get_rng_state(), loader_rng_state=generator.get_state()),
                           str(temporary))
                temporary.replace(directory / "best.pt")
            else:
                stale += 1
            memory = memory_usage()
            row = dict(epoch=epoch, training_mae=train_mae, validation_mae=val_mae,
                       train_seconds=train_seconds, validation_seconds=val_seconds,
                       epoch_seconds=time.perf_counter() - epoch_start,
                       working_set_mib=memory.get("working_set_mib"),
                       peak_working_set_mib=memory.get("peak_working_set_mib"))
            writer.writerow(row)
            stream.flush()
            history.append(row)
            print(json.dumps(dict(model=spec["id"], **row)), flush=True)
            if stale >= training["patience"]:
                break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(str(directory / "best.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = evaluate(model, val_loader, directory / "validation_predictions.npy")
    # Float64 per-element aggregation can differ slightly from training's float32 batch means.
    if not math.isclose(metrics["validation_mae"], best, rel_tol=0, abs_tol=1e-5):
        raise RuntimeError("Checkpoint reload validation mismatch")
    reloaded_mae, _ = run_epoch(model, val_loader, nn.L1Loss())
    if not math.isclose(reloaded_mae, best, rel_tol=0, abs_tol=1e-7):
        raise RuntimeError("Checkpoint reload training-MAE aggregation mismatch")
    plot_history(history, directory / "learning_curve.png")
    result = dict(id=spec["id"], name=spec["name"], parameters=count_trainable_parameters(model),
                  epochs_completed=len(history), best_epoch=best_epoch, stopped_early=stale >= training["patience"],
                  best_selection_validation_mae=best, checkpoint_reload_validation_mae=reloaded_mae,
                  training_loop_seconds=elapsed, first_epoch_seconds=history[0]["epoch_seconds"],
                  memory_before=memory_before, memory_after=memory_usage(),
                  validation_predictions=(directory / "validation_predictions.npy").relative_to(ROOT).as_posix(),
                  checkpoint=(directory / "best.pt").relative_to(ROOT).as_posix(), **metrics)
    save_json(directory / "summary.json", result)
    return result


def write_comparison(run_dir, results):
    if not results:
        return
    scalar_fields = [key for key, value in results[0].items() if not isinstance(value, (dict, list))]
    with (run_dir / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    with (run_dir / "memory_comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = ["id", "before_working_set_mib", "after_working_set_mib",
                  "process_peak_working_set_mib", "after_private_commit_mib"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(dict(id=result["id"],
                before_working_set_mib=result["memory_before"].get("working_set_mib"),
                after_working_set_mib=result["memory_after"].get("working_set_mib"),
                process_peak_working_set_mib=result["memory_after"].get("peak_working_set_mib"),
                after_private_commit_mib=result["memory_after"].get("private_commit_mib")))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for result in results:
        with (ROOT / result["checkpoint"]).with_name("history.csv").open(encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        ax.plot([int(r["epoch"]) for r in history], [float(r["validation_mae"]) for r in history],
                marker="o" if len(history) == 1 else None, label=result["name"])
    ax.set(xlabel="Epoch (fresh initialization)", ylabel="Validation MAE",
           title="Architecture comparison: completed runs only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(run_dir / "comparison.png"), dpi=150)
    plt.close(fig)


def write_report(run_dir, config, catalog, results, mode, cost):
    lines = ["# STEP 9-6 Architecture comparison", "", "Mode: **{}**. Full-budget mode executed: **{}**.".format(mode, mode == "full"),
             "", "## Sources and application", "",
             "[Paper Sections 3.1 and 3.2]({}) and [official MLP]({}mlp.py) / [ResNet]({}resnet.py) were inspected.".format(PAPER, OFFICIAL, OFFICIAL),
             "The paper's MLP uses Linear-ReLU-Dropout blocks. Its residual branch is BN-Linear-ReLU-Dropout-Linear-Dropout, added to the block input, without a post-add activation. A BN-ReLU-Linear prediction head follows the blocks.",
             "BatchNorm learns affine normalization and running statistics; training updates them, evaluation uses saved statistics. ReLU supplies nonlinearity. Dropout masks activations during training and is disabled during evaluation.",
             "Our widths, block counts, hidden factor 1.0, dropout probabilities 0.1/0.1, four-output regression, L1Loss and training protocol are project choices, not reproduced paper results. A-D keep zero dropout. E/F add BN, dropout and skips together: this is not a skip-only ablation.",
             "", "## Fair comparison", "",
             "Full mode trains the selected models from fresh initialization with Adam LR 0.0003, batch 1024, maximum 200 epochs, patience 15, seed 42, the saved train/validation arrays and unchanged train-fitted StandardScaler. No target scaling. A is retrained using the original MLP class. Equal epoch caps do not mean equal CPU time or parameter budgets; early stopping may use fewer epochs.",
             "The historical MAE 14.4135215038 at epoch 244 used a different LR schedule and more epochs. It is a reference only, excluded from the equal-budget ranking; its checkpoint is never loaded into new models.",
             "Model initialization and Dropout consume different RNG amounts across architectures; the same seed is not proof that performance differences isolate one architectural feature. The independent sampler generator gives the same new shuffle sequence to all candidates.",
             "", "## Architecture checks (synthetic data, not trained validation results)", "",
             "| ID | Model | Parameters | BN layers | Forward/backward | Synthetic step seconds |", "|---|---|---:|---:|---|---:|"]
    for row in catalog:
        lines.append("| {id} | {name} | {parameters:,} | {batchnorm_modules} | passed | {synthetic_step_median_seconds:.6f} |".format(**row))
    lines += ["", "All models passed output-shape checks at evaluation batch sizes 1, 32 and 1024, finite gradients, Adam parameter updates, repeated deterministic evaluation and frozen evaluation buffers. E/F also passed BN training-counter updates, stochastic training Dropout, exact zero-branch residual identity and identity-gradient checks.",
              "BatchNorm training requires more than one row per batch. Saved training size 648000 leaves a last batch of 832 rows, so no row is dropped. A singleton is supported in eval mode.",
              "", "## Completed training results", ""]
    if results:
        lines += ["| ID | Epochs | Best epoch | Validation MAE | MAPE (%) | Loop seconds | Working set MiB (after) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in results:
            lines.append("| {} | {} | {} | {:.8f} | {:.5f} | {:.3f} | {:.2f} |".format(
                row["id"], row["epochs_completed"], row["best_epoch"], row["validation_mae"],
                row["validation_mape_percent"], row["training_loop_seconds"],
                row["memory_after"].get("working_set_mib", 0)))
        lines += ["", "| ID | Layer 1 MAE / MAPE% | Layer 2 MAE / MAPE% | Layer 3 MAE / MAPE% | Layer 4 MAE / MAPE% |", "|---|---|---|---|---|"]
        for row in results:
            lines.append("| {} | {} |".format(row["id"], " | ".join(
                "{:.6f} / {:.4f}".format(row["layer_{}_mae".format(i)], row["layer_{}_mape_percent".format(i)]) for i in range(1, 5))))
    lines += ["", "MAPE is mean(abs(pred-target)/max(abs(target),1e-8))*100 over all elements; per-layer values average over validation rows. Near-zero denominator counts are saved in model summaries. Overall metrics use float64 accumulation; the early-stopping MAE uses the existing float32 batch-loss aggregation and may differ by rounding.",
              "", "## CPU cost and limitations", "",
              "Only B/E receive real 1-epoch training in smoke mode. Other models have synthetic compute estimates, not validation results. All real runs use all 648000 training rows and 81000 validation rows. Models are trained sequentially; num_workers=0 and CPU threads=4 by default.",
              "Memory snapshots include mapped resident pages and Python/PyTorch native memory. Peak working set is process-wide and cumulative, not an isolated per-model peak. The 16 bytes/parameter estimate covers FP32 weights, gradients and Adam moments only, excluding activations, temporary tensors, BN buffers and dataset mappings.",
              "A 200-epoch linear estimate ignores early stopping, CPU load, cache, initialization and sustained thermal effects. Synthetic estimates for untrained models exclude validation and disk overhead; calibrated estimates use observed smoke data but remain uncertain. See cost_estimates.json.",
              "", "## Next candidates", "",
              "One epoch cannot rank converged architectures or establish ensemble gains. Keep the preserved historical MLP as an anchor. Evaluate B and E under the full equal-budget protocol first, together with freshly trained A; E is a possible source of architectural diversity, not a confirmed accuracy winner. Select any ensemble only after checking validation residual correlations and measured validation performance; no ensemble is executed here.",
              "", "## Safety and commands", "",
              "Protected source/checkpoint metadata and hashes were checked before/after. Hold-out arrays are not listed, stat'ed or opened. No Gaussian noise, FT-Transformer, CNN, stacking or environment change.",
              "```bash", "python -m src.architecture_compare --check-only", "python -m src.architecture_compare", "python -m src.architecture_compare --full", "```",
              "Full mode starts selected models fresh; smoke checkpoints are not resumed."]
    lines += ["", "## Estimated CPU budget", "",
              "| ID | Estimated seconds/epoch | Estimated minutes at epoch cap | Method |",
              "|---|---:|---:|---|"]
    for row in cost:
        lines.append("| {} | {:.2f} | {:.1f} | {} |".format(
            row["id"], row["seconds_per_epoch_estimate"], row["seconds_200_epochs_estimate"] / 60, row["method"]))
    lines += ["", "Estimated sequential total at epoch cap: {:.1f} minutes. These are cost estimates, not completed training results.".format(
        sum(row["seconds_200_epochs_estimate"] for row in cost) / 60)]
    (run_dir / "STEP_9_6_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_models(config, models=None, full=False, check_only=False):
    ids = list(models) if models is not None else (list("ABCDEF") if full or check_only else list("BE"))
    if not ids or len(ids) != len(set(ids)) or any(i not in "ABCDEF" or len(i) != 1 for i in ids):
        raise ValueError("models must be unique IDs from A-F")
    return [spec for spec in config["models"] if spec["id"] in ids]


def compare(config_path=DEFAULT_CONFIG, full=False, check_only=False, models=None):
    config = load_config(config_path)
    selected = select_models(config, models, full, check_only)
    torch.set_num_threads(config["training"]["threads"])
    torch.use_deterministic_algorithms(True)
    before = protected_manifest()
    # Preserve all previous architecture runs, including smoke checkpoints.
    previous = [p for p in (ROOT / "outputs/architecture_comparison").rglob("*") if p.is_file()]
    previous_state = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in previous}
    mode = "check_only" if check_only else ("full" if full else "smoke_1epoch_BE")
    run_dir = ROOT / "outputs/architecture_comparison" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + mode + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", dict(config, selected_models=[s["id"] for s in selected], mode=mode, python=__import__("platform").python_version(),
              torch_version=str(torch.__version__), train_rows=648000, validation_rows=81000))
    save_json(run_dir / "source_manifest.json", before)
    print("Run directory: " + run_dir.relative_to(ROOT).as_posix(), flush=True)
    catalog, failures = [], []
    for spec in selected:
        directory = run_dir / (spec["id"] + "_" + spec["name"])
        directory.mkdir()
        try:
            check = check_model(spec, config)
        except Exception:
            failure = dict(id=spec["id"], stage="synthetic_check", traceback=traceback.format_exc())
            failures.append(failure)
            save_json(directory / "failure.json", failure)
            print(json.dumps(failure), flush=True)
            continue
        save_json(directory / "model_config.json", dict(spec, input_dim=226, output_dim=4,
                  training=config["training"], parameters=check["parameters"]))
        save_json(directory / "checks.json", check)
        catalog.append(check)
        print(json.dumps({key: check[key] for key in ("id", "parameters", "synthetic_step_median_seconds")}), flush=True)
    save_json(run_dir / "model_catalog.json", catalog)
    results = []
    if not check_only:
        preprocessing = json.loads((ROOT / "outputs/preprocessors/preprocessing_config.json").read_text(encoding="utf-8"))
        if preprocessing["target_scaling"] is not False or preprocessing["feature_columns"] != [str(i) for i in range(226)] or preprocessing["target_columns"] != ["layer_{}".format(i) for i in range(1, 5)]:
            raise ValueError("Preprocessing order/scaling mismatch")
        # Validate the exact same training/validation sources against the original run manifest.
        baseline_config = json.loads((ROOT / "outputs/training/20261107_112339_full_1f33cc10/config.json").read_text(encoding="utf-8"))
        for path, record in before.items():
            expected = baseline_config["source_artifacts"].get(str(Path(path)))
            if expected is not None and [record["size"], record["mtime_ns"]] != expected:
                raise ValueError("Original source artifact changed: " + path)
        train = MappedBatches(ROOT / "outputs/preprocessed", "train", 648000)
        validation = MappedBatches(ROOT / "outputs/preprocessed", "validation", 81000)
        with np.load(str(ROOT / "outputs/splits/data_split_indices.npz"), allow_pickle=False) as splits:
            validation_key = next(key for key in splits.files if "val" in key.lower())
            indices = splits[validation_key]
            if indices.shape != (len(validation),):
                raise ValueError("Validation indices shape mismatch")
            np.save(str(run_dir / "validation_indices.npy"), indices, allow_pickle=False)
        for spec in selected:
            if spec["id"] in {f["id"] for f in failures}:
                continue
            directory = run_dir / (spec["id"] + "_" + spec["name"])
            try:
                result = train_candidate(spec, config, directory, train, validation,
                                         config["training"]["max_epochs"] if full else 1)
                results.append(result)
            except Exception:
                failure = dict(id=spec["id"], traceback=traceback.format_exc())
                failures.append(failure)
                save_json(directory / "failure.json", failure)
                print(json.dumps(failure), flush=True)
            save_json(run_dir / ("progress_" + spec["id"] + ".json"),
                      dict(results=results, failures=failures))
    synthetic_steps = {row["id"]: row["synthetic_step_median_seconds"] for row in catalog}
    observed = {row["id"]: row["first_epoch_seconds"] for row in results}
    # Use B for MLPs and E for residuals when real smoke timings are available.
    cost = []
    for spec in selected:
        if spec["id"] not in synthetic_steps:
            continue
        anchor = "E" if spec["kind"] == "residual" else "B"
        estimated = observed.get(spec["id"])
        method = "observed first epoch"
        if estimated is None:
            if anchor in observed:
                estimated = observed[anchor] * synthetic_steps[spec["id"]] / synthetic_steps[anchor]
                method = "synthetic step ratio calibrated to " + anchor
            else:
                estimated = synthetic_steps[spec["id"]] * math.ceil(648000 / 1024)
                method = "synthetic train steps only; validation and IO excluded"
        cost.append(dict(id=spec["id"], seconds_per_epoch_estimate=estimated,
                         seconds_200_epochs_estimate=estimated * config["training"]["max_epochs"], method=method))
    save_json(run_dir / "cost_estimates.json", dict(models=cost,
              total_epoch_cap_seconds_estimate=sum(row["seconds_200_epochs_estimate"] for row in cost),
              caveat="Linear estimates, not measured full runs; load/cache/thermal behavior and early stopping vary."))
    write_comparison(run_dir, results)
    unchanged = before == protected_manifest()
    unchanged = unchanged and previous_state == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in previous}
    if not unchanged:
        raise RuntimeError("Protected artifacts changed")
    summary = dict(selected_models=[s["id"] for s in selected], failures=failures, mode=mode, full_comparison_executed=full, results=results,
                   untrained_ids=[spec["id"] for spec in config["models"] if spec["id"] not in {r["id"] for r in results}],
                   protected_sources_unchanged=True, holdout_accessed=False,
                   ranking_scope="fresh equal-budget models only" if full else "smoke only; not a final ranking")
    if full and results:
        summary["winner_by_validation_mae"] = min(results, key=lambda row: row["best_selection_validation_mae"])["id"]
    save_json(run_dir / "summary.json", summary)
    write_report(run_dir, config, catalog, results, mode, cost)
    if full and results:
        from .architecture_analysis import analyze
        analyze(run_dir)
    print("Completed: " + run_dir.relative_to(ROOT).as_posix(), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--full", action="store_true", help="Explicitly train fresh A-F, max 200 epochs each")
    modes.add_argument("--check-only", action="store_true", help="Synthetic checks/compute estimates only")
    parser.add_argument("--models", nargs="+", choices=list("ABCDEF"), help="Model IDs; default full A-F, smoke B/E")
    args = parser.parse_args()
    compare(args.config, args.full, args.check_only, args.models)
