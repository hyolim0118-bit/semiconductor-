"""Prepare, verify and restore immutable migration artifacts; never train."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / "artifacts"
RUN = "20261107_112339_full_1f33cc10"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def inspect_checkpoint(device="cpu", checkpoint_path=None):
    import torch
    from src.model import MLPRegressor, count_trainable_parameters

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check GPU driver and PyTorch CUDA build")
    path = checkpoint_path or BUNDLE / "baseline_100epoch/best.pt"
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    required = {"model_state_dict", "optimizer_state_dict", "epoch", "validation_mae", "config"}
    if not required.issubset(checkpoint):
        raise ValueError("Missing checkpoint keys: {}".format(required - set(checkpoint)))
    model = MLPRegressor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError("Adam state does not cover all parameters")
    steps = []
    for parameter, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].shape != parameter.shape or not torch.isfinite(state[key]).all():
                raise ValueError("Invalid Adam {}".format(key))
        steps.append(int(state["step"].item()))
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite model parameters")
    return {"checkpoint_bytes": path.stat().st_size, "keys": sorted(checkpoint),
            "epoch": checkpoint["epoch"], "validation_mae": checkpoint["validation_mae"],
            "trainable_parameters": count_trainable_parameters(model),
            "model_strict_load": True, "adam_state_load": True,
            "adam_parameter_states": len(optimizer.state), "adam_steps": sorted(set(steps)),
            "optimizer_learning_rates": [group["lr"] for group in optimizer.param_groups],
            "device_checked": device, "torch_version": str(torch.__version__),
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "rng_state_saved": False, "shuffle_generator_state_saved": False,
            "scheduler_state_saved": False, "resume_cli_implemented": False,
            "training_or_validation_run": False}


def prepare():
    """Local-only packaging, streaming hashes of large files without copying them."""
    if BUNDLE.exists():
        raise FileExistsError("artifacts already exists; refusing to replace it")
    mapping = {
        "splits/data_split_indices.npz": "outputs/splits/data_split_indices.npz",
        "preprocessors/preprocessing_config.json": "outputs/preprocessors/preprocessing_config.json",
        "preprocessors/x_scaler.joblib": "outputs/preprocessors/x_scaler.joblib",
        "preprocessors/training_channel_scaling_statistics.csv": "outputs/preprocessors/training_channel_scaling_statistics.csv",
        "statistics/split_summary.json": "outputs/metrics/split_summary.json",
        "statistics/split_target_distributions.csv": "outputs/metrics/split_target_distributions.csv",
        "statistics/train_target_statistics.csv": "outputs/metrics/train_target_statistics.csv",
    }
    for name in ("best.pt", "history.csv", "config.json", "summary.json",
                 "completion_verification.json", "learning_curve.png"):
        mapping["baseline_100epoch/" + name] = "outputs/training/" + RUN + "/" + name
    external_paths = [ROOT / name for name in ("train.csv", "test.csv", "sample_submission.csv")]
    external_paths.extend(sorted((ROOT / "outputs/preprocessed").glob("*.npy")))
    protected = [ROOT / path for path in mapping.values()] + external_paths
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}
    inspection = inspect_checkpoint(checkpoint_path=ROOT / mapping["baseline_100epoch/best.pt"])
    environment = {name: version(name) for name in (
        "torch", "numpy", "pandas", "scikit-learn", "matplotlib", "joblib",
        "scipy", "threadpoolctl")}
    BUNDLE.mkdir()
    entries = []
    for destination, source in mapping.items():
        original, copied = ROOT / source, BUNDLE / destination
        copied.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as reader, copied.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        info = file_info(original)
        if file_info(copied) != info:
            raise RuntimeError("Copy verification failed: " + destination)
        entries.append(dict(path="artifacts/" + destination, restore_to=source, **info))
    write_json(BUNDLE / "checkpoint_inspection.json", inspection)
    write_json(BUNDLE / "environment_cpu.json", environment)
    with (BUNDLE / "requirements-data.txt").open("x", encoding="utf-8") as stream:
        stream.write("# Install torch separately using a compatible CUDA wheel.\n")
        for name, installed in environment.items():
            if name != "torch":
                stream.write("{}=={}\n".format(name, installed))
    for name in ("checkpoint_inspection.json", "environment_cpu.json", "requirements-data.txt"):
        entries.append(dict(path="artifacts/" + name, **file_info(BUNDLE / name)))
    external = []
    required = {"X_train_scaled.npy", "y_train.npy", "X_validation_scaled.npy", "y_validation.npy"}
    for path in external_paths:
        print("Hashing external file:", path.relative_to(ROOT).as_posix(), flush=True)
        external.append(dict(path=path.relative_to(ROOT).as_posix(),
                             required_for_training=path.name in required, **file_info(path)))
    if before != {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}:
        raise RuntimeError("Original file size/mtime changed")
    write_json(BUNDLE / "manifest.json", {
        "algorithm": "SHA-256", "source_run": RUN,
        "files": entries, "external_files": external,
        "note": "External data are excluded from Git. Never regenerate the split for migration.",
        "source_sizes_and_mtimes_unchanged": True})
    print("Prepared artifacts; original files preserved.")


def load_manifest():
    return json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))


def verified_path(entry):
    path = (ROOT / entry["path"]).resolve()
    path.relative_to(ROOT)
    if not path.is_file() or file_info(path) != {key: entry[key] for key in ("bytes", "sha256")}:
        raise ValueError("Missing or mismatched file: " + entry["path"])
    return path


def verify(external=False):
    manifest = load_manifest()
    for entry in manifest["files"]:
        verified_path(entry)
    count = 0
    if external:
        for entry in manifest["external_files"]:
            if entry["required_for_training"] or (ROOT / entry["path"]).exists():
                verified_path(entry)
                count += 1
    print("SHA-256 verified: {} bundled, {} external files".format(len(manifest["files"]), count))


def restore():
    """Restore existing code's paths; preflight all conflicts before any copy."""
    verify()
    planned = []
    for entry in load_manifest()["files"]:
        if "restore_to" not in entry:
            continue
        destination = (ROOT / entry["restore_to"]).resolve()
        destination.relative_to(ROOT / "outputs")
        if destination.exists():
            if file_info(destination) != {key: entry[key] for key in ("bytes", "sha256")}:
                raise FileExistsError("Different existing file; refusing overwrite: " + entry["restore_to"])
        else:
            planned.append((verified_path(entry), destination))
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if sha256(source) != sha256(destination):
            raise RuntimeError("Restored copy differs: " + destination.name)
    print("Restored {} missing files; identical existing files left untouched.".format(len(planned)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "restore", "inspect"))
    parser.add_argument("--external", action="store_true", help="Also verify transferred data arrays/CSVs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        prepare()
    elif arguments.action == "verify":
        verify(arguments.external)
    elif arguments.action == "restore":
        restore()
    else:
        verify()
        print(json.dumps(inspect_checkpoint(arguments.device), indent=2))
