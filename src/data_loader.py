"""Read-only, chunked CSV loading and schema validation."""

import csv
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANNELS = [str(i) for i in range(226)]
TARGETS = ["layer_{}".format(i) for i in range(1, 5)]
SCHEMAS = {
    "train": TARGETS + CHANNELS,
    "test": ["id"] + CHANNELS,
    "sample_submission": ["id"] + TARGETS,
}


def validate_csv_files(root=PROJECT_ROOT):
    """Reject missing files, duplicate headers and unexpected columns."""
    paths = {}
    for name, expected in SCHEMAS.items():
        path = Path(root) / (name + ".csv")
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)) or set(header) != set(expected):
            raise ValueError("{}: invalid columns: {}".format(path.name, header))
        paths[name] = path
    return paths


def load_csv_chunks(path, chunksize=20000):
    """Keep float64 precision; retain missing, infinite and negative values.

    IDs remain strings so their original formatting and ordering are preserved.
    Nonnumeric feature/target values raise an error rather than being coerced.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    path = Path(path)
    columns = SCHEMAS[path.stem]
    dtype = {c: ("string" if c == "id" else "float64") for c in columns}
    return pd.read_csv(path, dtype=dtype, chunksize=chunksize, encoding="utf-8-sig")


def validate_chunk(frame, name):
    if set(frame.columns) != set(SCHEMAS[name]):
        raise ValueError("{}: inconsistent chunk schema".format(name))
    if "id" in frame and frame["id"].isna().any():
        raise ValueError("{}: missing id".format(name))


def create_data_splits(paths=None, output=None, random_state=42, chunksize=20000):
    """Make reproducible 80/10/10 row splits from labeled train.csv only.

    Saved indices are zero-based data-row positions (the CSV header is excluded).
    Validation and hold-out indices are saved for reuse in later experiments.
    No preprocessing is fitted here, so no statistics can leak across splits.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    paths = validate_csv_files(PROJECT_ROOT) if paths is None else paths
    train_path = Path(paths["train"])
    output = PROJECT_ROOT / "outputs" if output is None else Path(output)
    split_dir = output / "splits"
    metrics_dir = output / "metrics"
    split_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_chunks = []
    n_rows = 0
    for frame in load_csv_chunks(train_path, chunksize=chunksize):
        validate_chunk(frame, "train")
        target_chunks.append(frame[TARGETS].to_numpy(copy=True))
        n_rows += len(frame)
    if n_rows == 0:
        raise ValueError("train.csv contains no data rows")
    targets = pd.DataFrame(np.concatenate(target_chunks), columns=TARGETS)

    row_indices = np.arange(n_rows, dtype=np.int64)
    train_indices, remainder = train_test_split(
        row_indices, test_size=0.2, random_state=random_state, shuffle=True
    )
    validation_indices, holdout_indices = train_test_split(
        remainder, test_size=0.5, random_state=random_state, shuffle=True
    )
    splits = {
        "train": np.sort(train_indices),
        "validation": np.sort(validation_indices),
        "holdout_test": np.sort(holdout_indices),
    }

    names = list(splits)
    for i, first in enumerate(names):
        if np.unique(splits[first]).size != len(splits[first]):
            raise RuntimeError("{} split contains duplicate indices".format(first))
        for second in names[i + 1:]:
            if np.intersect1d(splits[first], splits[second]).size:
                raise RuntimeError("{} and {} splits overlap".format(first, second))
    combined = np.concatenate(list(splits.values()))
    if len(combined) != n_rows or not np.array_equal(np.sort(combined), row_indices):
        raise RuntimeError("Splits do not cover each train row exactly once")

    index_path = split_dir / "data_split_indices.npz"
    np.savez(index_path, **splits)
    distribution_rows = []
    shapes = {}
    for name, indices in splits.items():
        selected_targets = targets.iloc[indices]
        shapes[name] = {
            "rows": int(len(indices)),
            "X_shape": [int(len(indices)), len(CHANNELS)],
            "y_shape": [int(len(indices)), len(TARGETS)],
        }
        described = selected_targets.describe().T
        for target in TARGETS:
            for statistic, value in described.loc[target].items():
                distribution_rows.append({
                    "split": name,
                    "target": target,
                    "statistic": statistic,
                    "value": float(value),
                })

    pd.DataFrame(distribution_rows).to_csv(
        metrics_dir / "split_target_distributions.csv", index=False
    )
    summary = {
        "source": "train.csv",
        "index_semantics": "zero-based data row position; excludes CSV header",
        "random_state": random_state,
        "fractions": {name: len(indices) / n_rows for name, indices in splits.items()},
        "shapes": shapes,
        "pairwise_overlap": {"train_validation": 0, "train_holdout_test": 0,
                             "validation_holdout_test": 0},
        "covers_each_train_row_once": True,
        "public_test_used_for_evaluation": False,
        "holdout_reuse": "Keep holdout_test fixed for final evaluation and week-2 noise experiments.",
        "indices_file": str(index_path.relative_to(output)),
    }
    with (metrics_dir / "split_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return splits, summary
