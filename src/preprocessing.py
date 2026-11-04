"""Leakage-safe, chunked preprocessing for the fixed STEP 6 splits."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import CHANNELS, PROJECT_ROOT, TARGETS, validate_csv_files


SPLIT_NAMES = ("train", "validation", "holdout_test")


def _load_labeled_chunks(path, chunksize):
    """Read only the labeled table's numeric columns as float32."""
    dtype = {column: np.float32 for column in CHANNELS + TARGETS}
    return pd.read_csv(
        str(path), usecols=TARGETS + CHANNELS, dtype=dtype,
        chunksize=chunksize, encoding="utf-8-sig",
    )


def _load_split_indices(path):
    """Load and validate the saved STEP 6 row positions without rewriting them."""
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as saved:
        if set(saved.files) != set(SPLIT_NAMES):
            raise ValueError("Unexpected split names in {}".format(path))
        splits = {name: np.asarray(saved[name], dtype=np.int64) for name in SPLIT_NAMES}
    for name, indices in splits.items():
        if indices.ndim != 1 or np.unique(indices).size != len(indices):
            raise ValueError("{} indices must be a unique one-dimensional array".format(name))
        if len(indices) == 0:
            raise ValueError("{} split is empty".format(name))
    total = sum(len(indices) for indices in splits.values())
    if any(indices.min() < 0 or indices.max() >= total for indices in splits.values()):
        raise ValueError("Split indices do not form a contiguous zero-based partition")
    combined = np.concatenate(list(splits.values()))
    if len(np.unique(combined)) != total or not np.array_equal(np.sort(combined), np.arange(total)):
        raise ValueError("Saved splits overlap or do not cover train.csv exactly once")
    return splits, total


def _read_rows_in_index_order(path, indices, chunksize):
    """Read feature rows in the exact order requested by row-position indices."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0):
        raise ValueError("Probe indices must be a nonempty, nonnegative vector")
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    sorted_values = np.empty((len(indices), len(CHANNELS)), dtype=np.float32)
    row_start = 0
    found = 0
    for frame in _load_labeled_chunks(path, chunksize):
        row_end = row_start + len(frame)
        offsets = np.flatnonzero((sorted_indices >= row_start) & (sorted_indices < row_end))
        if len(offsets):
            local_rows = sorted_indices[offsets] - row_start
            sorted_values[offsets] = frame.iloc[local_rows][CHANNELS].to_numpy(dtype=np.float32)
            found += len(offsets)
        row_start = row_end
        if row_start > sorted_indices[-1]:
            break
    if found != len(indices):
        raise ValueError("Probe indices exceed the available train.csv rows")
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def transform_features(raw_reflectance, scaler=None, scaler_path=None):
    """Transform original-scale reflectance using the saved training scaler.

    For a noise experiment, add Gaussian noise to raw_reflectance before calling
    this function. Input must already be in CHANNELS order and have shape (n, 226).
    """
    if scaler is None:
        path = Path(scaler_path) if scaler_path else PROJECT_ROOT / "outputs" / "preprocessors" / "x_scaler.joblib"
        scaler = joblib.load(str(path))
    values = np.asarray(raw_reflectance, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(CHANNELS):
        raise ValueError("Expected raw reflectance with shape (n, 226)")
    if not np.isfinite(values).all():
        raise ValueError("Input reflectance contains NaN or Inf")
    return scaler.transform(values).astype(np.float32, copy=False)


def inverse_transform_targets(values, scaler=None, scaler_path=None):
    """Return scaled targets to original units when target scaling was enabled."""
    if scaler is None:
        path = Path(scaler_path) if scaler_path else PROJECT_ROOT / "outputs" / "preprocessors" / "y_scaler.joblib"
        if not path.is_file():
            raise ValueError("Target scaling is disabled; no target scaler was saved")
        scaler = joblib.load(str(path))
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(TARGETS):
        raise ValueError("Expected target values with shape (n, 4)")
    return scaler.inverse_transform(values).astype(np.float32, copy=False)


def run_preprocessing(root=PROJECT_ROOT, output=None, scale_targets=False, chunksize=20000):
    """Fit on train only, transform all fixed splits, and save float32 arrays."""
    root = Path(root)
    output = root / "outputs" if output is None else Path(output)
    paths = validate_csv_files(root)
    index_path = output / "splits" / "data_split_indices.npz"
    index_stat_before = (index_path.stat().st_size, index_path.stat().st_mtime_ns)
    splits, n_rows = _load_split_indices(index_path)
    split_id = np.full(n_rows, -1, dtype=np.int8)
    split_positions = np.empty(n_rows, dtype=np.int64)
    for split_number, name in enumerate(SPLIT_NAMES):
        split_id[splits[name]] = split_number
        split_positions[splits[name]] = np.arange(len(splits[name]), dtype=np.int64)
    if np.any(split_id < 0):
        raise ValueError("Some train rows are not assigned to a split")

    preprocessors = output / "preprocessors"
    arrays_dir = output / "preprocessed"
    existing = list(arrays_dir.glob("*.npy")) if arrays_dir.exists() else []
    if preprocessors.exists():
        existing.extend(preprocessors.glob("*.joblib"))
        for name in ("preprocessing_config.json", "training_channel_scaling_statistics.csv"):
            path = preprocessors / name
            if path.exists():
                existing.append(path)
    if existing:
        names = ", ".join(str(path) for path in sorted(existing))
        raise FileExistsError(
            "Preprocessing outputs already exist and will not be overwritten: " + names
        )
    preprocessors.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler() if scale_targets else None

    # First pass: update moments only with training rows; no validation or hold-out fit.
    row_start = 0
    for frame in _load_labeled_chunks(paths["train"], chunksize):
        row_end = row_start + len(frame)
        train_mask = split_id[row_start:row_end] == 0
        if train_mask.any():
            x = frame[CHANNELS].to_numpy(dtype=np.float32, copy=True)[train_mask]
            y = frame[TARGETS].to_numpy(dtype=np.float32, copy=True)[train_mask]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Training data contains NaN or Inf")
            x_scaler.partial_fit(x)
            if y_scaler is not None:
                y_scaler.partial_fit(y)
        row_start = row_end
    if row_start != n_rows or x_scaler.n_samples_seen_ != len(splits["train"]):
        raise RuntimeError("Training scaler was not fit on exactly the saved training rows")

    x_scaler_path = preprocessors / "x_scaler.joblib"
    joblib.dump(x_scaler, str(x_scaler_path))
    if y_scaler is not None:
        joblib.dump(y_scaler, str(preprocessors / "y_scaler.joblib"))
    else:
        stale_y_scaler = preprocessors / "y_scaler.joblib"
        if stale_y_scaler.exists():
            stale_y_scaler.unlink()

    # Disk-backed arrays bound peak memory. Values are written in saved-index order.
    mapped = {}
    targets_written = {name: 0 for name in SPLIT_NAMES}
    input_quality = {name: {"X_nan_count": 0, "X_inf_count": 0,
                           "y_nan_count": 0, "y_inf_count": 0} for name in SPLIT_NAMES}
    x_paths, y_paths = {}, {}
    for name in SPLIT_NAMES:
        shape_x = (len(splits[name]), len(CHANNELS))
        shape_y = (len(splits[name]), len(TARGETS))
        x_paths[name] = arrays_dir / ("X_{}_scaled.npy".format(name))
        y_paths[name] = arrays_dir / ("y_{}.npy".format(name))
        mapped[(name, "X")] = np.lib.format.open_memmap(str(x_paths[name]), mode="w+", dtype=np.float32, shape=shape_x)
        mapped[(name, "y")] = np.lib.format.open_memmap(str(y_paths[name]), mode="w+", dtype=np.float32, shape=shape_y)

    row_start = 0
    for frame in _load_labeled_chunks(paths["train"], chunksize):
        row_end = row_start + len(frame)
        x_raw = frame[CHANNELS].to_numpy(dtype=np.float32, copy=True)
        y_raw = frame[TARGETS].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(x_raw).all() or not np.isfinite(y_raw).all():
            raise ValueError("Input data contains NaN or Inf")
        for split_number, name in enumerate(SPLIT_NAMES):
            mask = split_id[row_start:row_end] == split_number
            if not mask.any():
                continue
            positions = split_positions[row_start:row_end][mask]
            input_quality[name]["X_nan_count"] += int(np.isnan(x_raw[mask]).sum())
            input_quality[name]["X_inf_count"] += int(np.isinf(x_raw[mask]).sum())
            input_quality[name]["y_nan_count"] += int(np.isnan(y_raw[mask]).sum())
            input_quality[name]["y_inf_count"] += int(np.isinf(y_raw[mask]).sum())
            mapped[(name, "X")][positions] = transform_features(x_raw[mask], scaler=x_scaler)
            y_values = y_scaler.transform(y_raw[mask]) if y_scaler is not None else y_raw[mask]
            mapped[(name, "y")][positions] = y_values.astype(np.float32, copy=False)
            targets_written[name] += len(positions)
        row_start = row_end
    for array in mapped.values():
        array.flush()
    del mapped
    if row_start != n_rows or any(targets_written[name] != len(splits[name]) for name in SPLIT_NAMES):
        raise RuntimeError("Not all split rows were transformed and saved")

    # Reload the serialized scaler and compare its outputs to saved float32 values.
    loaded_scaler = joblib.load(str(x_scaler_path))
    probe_rows = min(1024, len(splits["train"]))
    probe_indices = splits["train"][:probe_rows]
    raw_probe = np.empty((probe_rows, len(CHANNELS)), dtype=np.float32)
    # The helper restores the original split-index order after reading CSV rows.
    raw_probe[:] = _read_rows_in_index_order(paths["train"], probe_indices, chunksize)
    saved_probe = np.load(str(x_paths["train"]), mmap_mode="r")[:probe_rows]
    reloaded_probe = transform_features(raw_probe, scaler=loaded_scaler)
    if not np.array_equal(reloaded_probe, saved_probe):
        raise RuntimeError("Reloaded scaler does not reproduce saved transformed values")

    stats_rows = []
    split_stats = {}
    for name in SPLIT_NAMES:
        x_values = np.load(str(x_paths[name]), mmap_mode="r")
        y_values = np.load(str(y_paths[name]), mmap_mode="r")
        if x_values.dtype != np.float32 or y_values.dtype != np.float32:
            raise RuntimeError("Saved arrays must use float32")
        if x_values.shape != (len(splits[name]), len(CHANNELS)) or y_values.shape != (len(splits[name]), len(TARGETS)):
            raise RuntimeError("Saved array shape does not match split indices")
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise RuntimeError("Transformed data contains NaN or Inf")
        split_stats[name] = {
            "X_shape": list(x_values.shape), "X_dtype": str(x_values.dtype),
            "y_shape": list(y_values.shape), "y_dtype": str(y_values.dtype),
            "input_X_dtype": "float32", "input_y_dtype": "float32",
            "input_quality_counts": input_quality[name],
            "X_nan_count": int(np.isnan(x_values).sum()), "X_inf_count": int(np.isinf(x_values).sum()),
            "y_nan_count": int(np.isnan(y_values).sum()), "y_inf_count": int(np.isinf(y_values).sum()),
        }
        if name == "train":
            means = np.asarray(x_scaler.mean_)
            scales = np.asarray(x_scaler.scale_)
            for i, channel in enumerate(CHANNELS):
                stats_rows.append({
                    "channel": channel, "mean_after_scaling": float(np.mean(x_values[:, i])),
                    "std_after_scaling_ddof0": float(np.std(x_values[:, i], dtype=np.float64)),
                    "scaler_mean_before_scaling": float(means[i]),
                    "scaler_scale_before_scaling": float(scales[i]),
                })
    training_stats = pd.DataFrame(stats_rows)
    max_abs_mean = float(training_stats["mean_after_scaling"].abs().max())
    max_std_error = float((training_stats["std_after_scaling_ddof0"] - 1).abs().max())
    if max_abs_mean > 1e-4 or max_std_error > 1e-4:
        raise RuntimeError("Training scaled channels are not approximately zero-mean/unit-variance")
    training_stats.to_csv(preprocessors / "training_channel_scaling_statistics.csv", index=False)

    config = {
        "method": "sklearn.preprocessing.StandardScaler",
        "feature_columns": CHANNELS,
        "target_columns": TARGETS,
        "feature_scaler_fit_split": "train only",
        "target_scaling": bool(scale_targets),
        "float_dtype": "float32",
        "split_indices": "outputs/splits/data_split_indices.npz (read-only)",
        "input_order_for_noise_experiments": "original-scale reflectance -> optional Gaussian noise -> transform_features()",
        "arrays": {name: {"X": str(x_paths[name].relative_to(output)), "y": str(y_paths[name].relative_to(output))} for name in SPLIT_NAMES},
        "split_shapes_and_quality": split_stats,
        "training_scaled_max_abs_channel_mean": max_abs_mean,
        "training_scaled_max_channel_std_error_from_one": max_std_error,
        "reloaded_scaler_probe_rows": int(probe_rows),
        "reloaded_scaler_matches_saved_output": True,
    }
    with (preprocessors / "preprocessing_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
    index_stat_after = (index_path.stat().st_size, index_path.stat().st_mtime_ns)
    if index_stat_after != index_stat_before:
        raise RuntimeError("Saved STEP 6 split indices were modified")
    return config


if __name__ == "__main__":
    result = run_preprocessing()
    print(json.dumps(result, indent=2, ensure_ascii=False))
