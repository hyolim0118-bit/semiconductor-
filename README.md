# Semiconductor thin-film thickness prediction

## Scope

STEP 5 implements read-only CSV loading, validation and EDA. STEP 6 creates
reproducible train, validation and hold-out row splits. STEP 7 scales input
features using training data only. Model training is not implemented. Source
CSVs remain at the project root and are never written.

## Environment

Created with Python 3.9.12 on Windows 11. Dependencies are listed in
requirements.txt without version pins; pip selects compatible releases.

## Git Bash

Open a terminal in this project folder, then run:

```bash
./.venv/Scripts/python.exe main.py
```

Activation is optional:

```bash
source .venv/Scripts/activate
```

In VS Code, run Python: Select Interpreter and select
`.venv/Scripts/python.exe`. The explicit Python path needs no activation.

## Paths and directories

Paths use pathlib.Path and derive the project root from __file__, supporting
Korean folder names independently of the terminal working directory.

- src/data_loader.py: schema checks and chunked CSV reading
- src/eda.py: full-dataset validation, statistics and figures
- main.py: run loading, EDA and data splitting
- outputs/figures/: plots
- outputs/metrics/: evaluation results
- outputs/checkpoints/: model weights
- outputs/preprocessors/: saved preprocessing objects

.venv/ and outputs/ are excluded by .gitignore.

## EDA details

Reads use 20,000-row float64 chunks to preserve CSV precision. Column moments
are merged across chunks. Only four train target columns are retained for
exact quantiles and histograms. SQLite stores duplicate fingerprints on disk;
repeated fingerprints are verified against full rows in a second pass, so hash
collisions do not count as duplicates. Temporary databases are removed on exit.
Counts describe excess occurrences within each CSV, both full rows and X-only
vectors. Test/submission IDs must be unique and match in order.

Missing/infinite measurements are counted separately. Statistics and plots use
finite values; all finite negative values and outliers are retained. Standard
deviation uses ddof=1. Eight sample spectra are uniformly selected with seed 42.
Channel indices 0..225 are not wavelengths. Thickness units remain as in the CSV.
Invalid schemas and nonnumeric measurements stop execution.

Generated artifacts (refreshed on each run):

- outputs/figures/train_target_distributions.png
- outputs/figures/{train,test}_channel_statistics.png
- outputs/figures/{train,test}_sample_spectra.png
- outputs/metrics/eda_summary.json: shapes, columns, dtypes and validation counts
- outputs/metrics/*_column_statistics.csv: per-column quality counts and statistics
- outputs/metrics/{train,test}_channel_statistics.csv: all 226 reflectance channels
- outputs/metrics/train_target_statistics.csv: thickness statistics and quantiles

## STEP 7: preprocessing

Run the preprocessing step from Git Bash with:

```bash
./.venv/Scripts/python.exe -m src.preprocessing
```

It reads (and verifies without rewriting) the saved STEP 6 indices. A
`StandardScaler` is incrementally fitted using training rows only, then applied
to all three splits. Inputs are read and saved as float32. Targets stay in their
original units by default; `run_preprocessing(scale_targets=True)` enables an
optional target scaler and `inverse_transform_targets` restores original units.
For a future noise experiment, add Gaussian noise to the original reflectance
values first, then call `transform_features` with the saved scaler.

Saved artifacts are under `outputs/preprocessors/` (scaler, configuration and
training channel statistics) and `outputs/preprocessed/` (float32 `.npy` arrays
for X and y in each split). The validation report records shapes, dtypes,
nonfinite counts and scaled training channel statistics.

## STEP 6: data splits

`create_data_splits` uses only labeled rows in `train.csv`. With `random_state=42`
and shuffled `train_test_split` calls, it makes 80% training, 10% validation and
10% hold-out test partitions. Indices are zero-based row positions after the
header; the sorted arrays are saved to `outputs/splits/data_split_indices.npz`.
`outputs/metrics/split_summary.json` records row counts, shapes, seed, fractions
and overlap/coverage checks. `outputs/metrics/split_target_distributions.csv`
compares each layer's count, mean, standard deviation, min, quartiles and max
between partitions. The hold-out indices are fixed for final evaluation and
week-2 noise experiments. Public `test.csv` is not used for any split or score.
No scalers or other data-dependent transforms are fitted during splitting.
