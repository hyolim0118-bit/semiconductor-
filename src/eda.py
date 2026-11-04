"""Read-only, bounded-memory EDA; no data cleaning or modelling."""

import json
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import CHANNELS, TARGETS, load_csv_chunks, validate_chunk


def _hashes(frame):
    normalized = frame.copy()
    numeric = normalized.select_dtypes(include="number").columns
    normalized[numeric] = normalized[numeric].mask(normalized[numeric] == 0, 0.0).fillna(np.nan)
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy()


def _exact_duplicates(path, scopes, db, chunksize):
    """Hash collisions are verified against full rows, never counted blindly."""
    candidates = {s: {r[0] for r in db.execute(
        "SELECT key FROM fingerprints WHERE scope=? AND n>1", (s,))} for s in scopes}
    results = {s: 0 for s in scopes}
    if not any(candidates.values()):
        return results
    db.execute("CREATE TABLE exact (scope TEXT, value BLOB, PRIMARY KEY(scope,value)) WITHOUT ROWID")
    for frame in load_csv_chunks(path, chunksize):
        for scope, columns in scopes.items():
            if not candidates[scope]:
                continue
            subset = frame[columns]
            mask = [int(h).to_bytes(8, "little") in candidates[scope] for h in _hashes(subset)]
            for row in subset.loc[mask].itertuples(index=False, name=None):
                values = [None if pd.isna(v) else (0.0 if isinstance(v, float) and v == 0 else v) for v in row]
                key = json.dumps(values, separators=(",", ":"), allow_nan=True).encode()
                cursor = db.execute("INSERT OR IGNORE INTO exact VALUES (?,?)", (scope, key))
                results[scope] += int(cursor.rowcount == 0)
    db.commit()
    db.execute("DROP TABLE exact")
    return results


def _analyze_table(name, path, output, chunksize, db):
    rows, targets, ids = 0, [], []
    rng = np.random.default_rng(42)
    sampled = np.empty((0, len(CHANNELS)))
    indices, priorities = np.empty(0, dtype=np.int64), np.empty(0)
    scopes = None
    db.execute("DELETE FROM fingerprints")
    for frame in load_csv_chunks(path, chunksize):
        validate_chunk(frame, name)
        if scopes is None:
            columns = frame.columns.tolist()
            numeric = [c for c in columns if c != "id"]
            scopes = {"rows": columns}
            if set(CHANNELS).issubset(columns):
                scopes["X"] = CHANNELS
            count = np.zeros(len(numeric), dtype=np.int64)
            missing, posinf, neginf, negative = (count.copy() for _ in range(4))
            mean = np.zeros(len(numeric))
            m2 = mean.copy()
            minimum, maximum = np.full(len(numeric), np.inf), np.full(len(numeric), -np.inf)
            dtypes = {c: str(t) for c, t in frame.dtypes.items()}
        values = frame[numeric].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        n = finite.sum(axis=0)
        batch_mean = np.divide(np.where(finite, values, 0).sum(axis=0), n,
                               out=np.zeros_like(mean), where=n > 0)
        centered = np.where(finite, values - batch_mean, 0)
        batch_m2 = np.einsum("ij,ij->j", centered, centered)
        total, delta = count + n, batch_mean - mean
        mean += np.divide(delta * n, total, out=np.zeros_like(mean), where=total > 0)
        m2 += batch_m2 + np.divide(delta ** 2 * count * n, total,
                                   out=np.zeros_like(mean), where=total > 0)
        count = total
        missing += np.isnan(values).sum(axis=0)
        posinf += np.isposinf(values).sum(axis=0)
        neginf += np.isneginf(values).sum(axis=0)
        negative += (finite & (values < 0)).sum(axis=0)
        minimum = np.minimum(minimum, np.where(finite, values, np.inf).min(axis=0))
        maximum = np.maximum(maximum, np.where(finite, values, -np.inf).max(axis=0))
        for scope, scope_columns in scopes.items():
            db.executemany(
                "INSERT INTO fingerprints VALUES (?,?,1) ON CONFLICT(scope,key) DO UPDATE SET n=n+1",
                ((scope, int(h).to_bytes(8, "little")) for h in _hashes(frame[scope_columns])),
            )
        db.commit()
        if "id" in frame:
            ids.extend(frame["id"].tolist())
        if name == "train":
            targets.append(frame[TARGETS].to_numpy(copy=True))
        if "X" in scopes:
            sampled = np.concatenate((sampled, frame[CHANNELS].to_numpy()))
            priorities = np.concatenate((priorities, rng.random(len(frame))))
            indices = np.concatenate((indices, np.arange(rows, rows + len(frame))))
            selected = np.argsort(priorities)[:8]
            sampled, priorities, indices = sampled[selected], priorities[selected], indices[selected]
        rows += len(frame)
        print("{}: {:,} rows analyzed".format(name, rows), flush=True)
    if rows == 0:
        raise ValueError("{} is empty".format(path))
    duplicates = _exact_duplicates(path, scopes, db, chunksize)
    std = np.sqrt(np.divide(m2, count - 1, out=np.full_like(mean, np.nan), where=count > 1))
    stats = pd.DataFrame({
        "column": numeric, "finite_count": count, "missing_count": missing,
        "positive_infinity_count": posinf, "negative_infinity_count": neginf,
        "negative_finite_count": negative,
        "min": np.where(count > 0, minimum, np.nan),
        "max": np.where(count > 0, maximum, np.nan),
        "mean": np.where(count > 0, mean, np.nan), "std": std,
    }).set_index("column")
    stats.to_csv(output / "metrics" / (name + "_column_statistics.csv"))
    info = {
        "shape": [rows, len(columns)], "columns": columns, "dtypes": dtypes,
        "missing_cells": int(missing.sum()),
        "infinite_cells": int(posinf.sum() + neginf.sum()),
        "duplicate_rows": duplicates["rows"], "duplicate_X_rows": duplicates.get("X"),
    }
    if ids:
        info["duplicate_ids"] = len(ids) - len(set(ids))
        if info["duplicate_ids"]:
            raise ValueError("{} has duplicate IDs".format(name))
    if "X" in scopes:
        channels = stats.loc[CHANNELS]
        channels.to_csv(output / "metrics" / (name + "_channel_statistics.csv"))
        info["reflectance"] = {
            "min": float(channels["min"].min()), "max": float(channels["max"].max()),
            "negative_cells": int(channels["negative_finite_count"].sum()),
        }
        _plot_reflectance(name, channels, sampled, indices, output / "figures")
    if name == "train":
        target_frame = pd.DataFrame(np.concatenate(targets), columns=TARGETS)
        finite_targets = target_frame.replace([np.inf, -np.inf], np.nan)
        finite_targets.describe().T.to_csv(output / "metrics" / "train_target_statistics.csv")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for column, ax in zip(TARGETS, axes.flat):
            ax.hist(finite_targets[column].dropna(), bins=60, color="steelblue", edgecolor="white")
            ax.set(title=column, xlabel="Thickness (original CSV units)", ylabel="Count")
        fig.tight_layout()
        fig.savefig(output / "figures" / "train_target_distributions.png", dpi=150)
        plt.close(fig)
    return info, ids


def _plot_reflectance(name, stats, samples, indices, directory):
    x = np.arange(len(CHANNELS))
    mean, std = stats["mean"].to_numpy(), stats["std"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, mean, label="Mean")
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, label="Mean +/- 1 std")
    ax.set(xlabel="Channel index", ylabel="Reflectance", title=name + " channel statistics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / (name + "_channel_statistics.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for values, index in zip(samples, indices):
        ax.plot(x, np.where(np.isfinite(values), values, np.nan), label="row {}".format(index), linewidth=0.9)
    ax.set(xlabel="Channel index", ylabel="Reflectance", title=name + " random spectra (seed=42)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(directory / (name + "_sample_spectra.png"), dpi=150)
    plt.close(fig)


def run_eda(paths, output, chunksize=20000):
    """Analyze every row, with disk-backed exact duplicate validation."""
    output = Path(output)
    for directory in (output / "figures", output / "metrics"):
        directory.mkdir(parents=True, exist_ok=True)
    before = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in paths.items()}
    summary = {
        "chunk_size": chunksize, "random_seed": 42,
        "statistics_policy": "All rows; finite-only numeric statistics, sample std (ddof=1). No rows removed. Nonfinite counts reported separately.",
        "duplicates_policy": "Within-file excess occurrences; hash candidates verified against full values on disk.",
        "channel_axis": "Channel index 0..225, not wavelength.", "tables": {},
    }
    ids = {}
    with TemporaryDirectory(prefix="thinfilm_eda_") as temporary:
        with closing(sqlite3.connect(str(Path(temporary) / "duplicates.sqlite"))) as db:
            db.execute("PRAGMA cache_size=-16000")
            db.execute("CREATE TABLE fingerprints (scope TEXT, key BLOB, n INTEGER, PRIMARY KEY(scope,key)) WITHOUT ROWID")
            for name, path in paths.items():
                summary["tables"][name], ids[name] = _analyze_table(name, path, output, chunksize, db)
    if ids["test"] != ids["sample_submission"]:
        raise ValueError("test and sample_submission IDs or their order differ")
    summary["test_submission_ids_match"] = True
    after = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("Source CSV size or modification time changed during EDA")
    summary["source_metadata_unchanged"] = True
    clean = json.loads(json.dumps(summary), parse_constant=lambda value: None)
    with (output / "metrics" / "eda_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(clean, stream, indent=2, ensure_ascii=False, allow_nan=False)
    for name, info in clean["tables"].items():
        print(name, {key: value for key, value in info.items() if key not in ("columns", "dtypes")}, flush=True)
    print("EDA complete: outputs/figures and outputs/metrics", flush=True)
    return clean
