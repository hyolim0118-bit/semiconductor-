# Semiconductor thin-film thickness prediction

## Scope

STEP 5 implements read-only CSV loading, validation and EDA. STEP 6 creates
reproducible train, validation and hold-out row splits. STEP 7 scales input
features using training data only. STEP 8 implements and verifies the PyTorch
MLP baseline. STEP 9 implements CPU training with a one-epoch default. Source
CSVs remain at the project root and are never written.

## Environment

GPU migration instructions and upload contents are in the **GPU migration**
section below. Keep the existing local CPU environment unchanged.

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

## STEP 9: CPU training

From the project root in Git Bash:

```bash
# Disposable 32-row update check, then exactly one real epoch (default).
./.venv/Scripts/python.exe -m src.train
# Separate fresh full run: at most 100 epochs; patience 10.
./.venv/Scripts/python.exe -m src.train --full
# Regression checks.
./.venv/Scripts/python.exe -m unittest tests.test_train tests.test_model -v
```

Both commands use CPU, seed 42, L1Loss, Adam (lr=0.001), batches of 1024,
648,000 training rows and 81,000 validation rows, without target scaling.
CPU intra-op threads default to min(4, CPU count); `--threads N` overrides this.
Read-only NumPy mappings and single-process batch loading bound Python-side
copies to batches; the OS may keep mapped file pages resident. Training rows
are shuffled every epoch and partial final batches are retained. Training MAE
is measured during updates, weighted by target-element count; validation MAE
uses the end-of-epoch model in eval/no-grad mode. Both average all four targets
in original units. Hold-out arrays are never opened.

Each invocation creates a unique `outputs/training/<timestamp>_<mode>_<id>/`
directory with config.json, smoke_check.json, history.csv, best.pt,
learning_curve.png and summary.json. Within this new directory only, best.pt
is replaced when validation MAE strictly improves. Full runs stop after 10
consecutive epochs without improvement. A full command starts fresh, rather
than resuming the one-epoch checkpoint. The best checkpoint includes model
and optimizer state, epoch, validation MAE and configuration; it is reloaded
with `weights_only=True` and its validation MAE checked before completion.
No previous run's logs/checkpoints are overwritten. Failed runs may contain
partial artifacts and are retained for diagnosis.

STEP 9 verified run: `outputs/training/20261107_111633_smoke_1epoch_efee698e/`.
One real epoch completed: training 3.3165 s, validation 0.2044 s, total including
checkpoint save 3.5322 s. Training MAE was 66.23812577 and validation MAE was
55.63218701. Checkpoint reload reproduced the validation MAE exactly. Peak
process working set through plotting was 905.38 MiB (epoch end: 879.51 MiB).
100 epochs linearly extrapolate to 353.22 s (5.89 min); an earlier run took
8.49 s/epoch, giving 14.14 min. Cache and CPU load affect this estimate, and
early stopping can reduce it. Process peak includes native tensors, mapped
resident pages, and plotting, not just Python allocations. Approximately
1 GiB was used in this run; allow around 1.5-2 GiB of available RAM for headroom.
Resident memory is not expected to grow proportionally with epoch count.

The 32-row smoke check confirmed finite gradients and changed parameters.
The first real one-epoch attempt completed training but failed checkpoint
reload because the PyTorch version was stored as a TorchVersion object;
configuration now stores a plain string. The fresh successful run above used
the corrected serialization. A preceding console-print attempt also stopped
before training on the decomposed Korean absolute path; console output now
uses the relative ASCII run path. Neither failed run was overwritten.
Source CSV, split, scaler and preprocessed file sizes/mtimes were unchanged.
Full training was subsequently completed in
`outputs/training/20261107_112339_full_1f33cc10/`: 100 epochs, best validation
MAE 17.99285922617971 at epoch 100, and successful checkpoint reload.
Early stopping did not trigger. Gaussian noise experiments and STEP 10 final
evaluation have not been run.

## STEP 8: MLP baseline and forward checks

`src/config.py` defines dimensions; `src/model.py` provides `MLPRegressor`.
The architecture is 226 -> 256 -> 128 -> 64 -> 4, with ReLU after each hidden
layer and no output activation (99,524 trainable weights and biases).
Output columns follow layer_1, layer_2, layer_3, layer_4 order.

Run from the project root in Git Bash:

```bash
./.venv/Scripts/python.exe -m src.model
./.venv/Scripts/python.exe -m unittest tests.test_model -v
```

Checks cover float32 batches of 1, 32 and 1024, finite outputs, invalid input
shapes, and 32 real training rows loaded with a read-only NumPy memory map.
The real-data test is explicitly skipped if its array is missing. Tests check
that existing CSV/output file sizes and modification times remain unchanged.
No split, scaler fit, training or tuning is performed.

Verified on Python 3.9.12 / PyTorch 2.8.0+cpu: all four tests passed on CPU;
CUDA was unavailable. The real array was (648000, 226), float32, and its sample
produced (32, 4) finite outputs. These checks establish input compatibility,
not predictive accuracy. Current preprocessing leaves targets unscaled.

Before STEP 9, define the DataLoader batch size, loss, optimizer, learning
rate, epochs, seeds, validation/checkpoint rules, and target-scale handling.
Retain the saved splits and reserve hold-out data for final evaluation.

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

## GPU migration

This section prepares an external GPU machine; it does not change the local
Windows/Python 3.9.12/PyTorch 2.8.0+cpu installation. No new training was run.

### Files to commit after review

- Existing `src/`, `tests/`, `main.py`, `requirements.txt`, README and roadmap.
- `.gitignore`, `.gitattributes`, `scripts/gpu_transfer.py` and transfer tests.
- `artifacts/splits/data_split_indices.npz`: byte-identical saved 80/10/10 indices.
- `artifacts/preprocessors/`: original x_scaler.joblib, preprocessing_config.json,
  training_channel_scaling_statistics.csv.
- `artifacts/statistics/`: split_summary.json, split_target_distributions.csv,
  train_target_statistics.csv.
- `artifacts/baseline_100epoch/`: original best.pt, history.csv, config.json,
  summary.json, completion_verification.json and learning_curve.png.
- `artifacts/manifest.json`, checkpoint_inspection.json, environment_cpu.json,
  requirements-data.txt.

`outputs/`, raw CSV files, all `.npy`, `.venv/`, `.venv-gpu/`, Python caches and
scratch scripts remain excluded. Only the curated statistics/log CSV paths in
artifacts are allowed. `.gitattributes` disables newline conversion inside
artifacts so Git checkout preserves SHA-256 across Windows and Linux.
Nothing has been staged, committed or pushed by this migration preparation.
Existing changes to src/config.py, src/model.py, src/train.py and tests from
STEP 8–9 are still uncommitted and must be included in the reviewed commit.

### Recreate the external environment

On the GPU machine, first check its GPU model and driver with `nvidia-smi`.
Choose a PyTorch CUDA wheel compatible with both the GPU architecture and
driver. The CUDA version displayed by nvidia-smi is a driver capability,
not a command to install a particular locally installed toolkit.
Use the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v280)
and [NVIDIA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The example below uses PyTorch 2.8.0/cu126 **only if compatible with that machine**;
2.8.0 also has cu128/cu129 wheels. No GPU/driver has yet been inspected here.

Linux example, after installing Python 3.9.12 and cloning the reviewed repository:

```bash
python3.9 -m venv .venv-gpu
source .venv-gpu/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r artifacts/requirements-data.txt
python -m pip check
```

Windows Git Bash: create the environment with
`python -m venv .venv-gpu` using Python 3.9.12, then
`source .venv-gpu/Scripts/activate`; use the same pip commands afterward.
requirements-data.txt pins the locally observed data/scaler dependencies
(including scikit-learn 1.6.1 and joblib 1.5.3), leaving torch to be installed
separately. The original requirements.txt is preserved and is not a GPU lockfile.
torchvision and torchaudio are not used by this project.

### Verify and restore the same data/splits/scaler

From the cloned repository root, with the new environment active:

```bash
# Stdlib-only SHA-256 verification; no model or data fitting.
python scripts/gpu_transfer.py verify
# Restore bundled files to existing code paths; differing files cause an error.
python scripts/gpu_transfer.py restore
```

restore copies the shared index into `outputs/splits/data_split_indices.npz`,
the scaler/settings into `outputs/preprocessors/`, statistics into
`outputs/metrics/`, and the baseline into its original `outputs/training/` run
path. Existing identical files are untouched; existing different files are
never overwritten. `src/preprocessing.py` already reads the restored split
path, so no loader/path changes or new splitting are needed.

Separately transfer the four existing training/validation arrays to the same
relative locations (these are not in Git):

```text
outputs/preprocessed/X_train_scaled.npy
outputs/preprocessed/y_train.npy
outputs/preprocessed/X_validation_scaled.npy
outputs/preprocessed/y_validation.npy
```

These four arrays total 670,680,512 bytes (about 639.61 MiB). They already use
the saved split order and train-only StandardScaler; y remains in original
units. Do not scale them again. Source CSVs and the two Hold-out arrays may be
transferred separately when needed later; Hold-out is not needed for tuning.
The manifest records hashes of all 3 original CSVs and 6 arrays, without
including their contents in Git. Identical row order matters, not just shape.

```bash
# Require all four training/validation arrays; also check any optional CSVs
# or hold-out arrays present. Absent optional files are not an error.
python scripts/gpu_transfer.py verify --external
python scripts/gpu_transfer.py inspect --device cuda
python -m unittest tests.test_gpu_transfer tests.test_train tests.test_model -v
```

Do not run main.py for migration: it runs EDA and creates/writes splits.
Do not run src.preprocessing to recreate already transferred .npy arrays.
If separately regenerating arrays is ever needed, first design a path that
uses the saved scaler for transform only; the current preprocessing entry
point fits a scaler and is not the migration procedure.

### Checkpoint and resume readiness

best.pt is 1,205,379 bytes (about 1.15 MiB). CPU inspection verified strict model
load, Adam state load for all 8 parameter tensors, 63,300 optimizer steps,
epoch=100, lr=0.001 and stored Validation MAE=17.99285922617971. No optimizer
step or validation pass was run for this inspection. GPU loading is untested
locally; `inspect --device cuda` checks availability and state loading on the
destination machine without training or evaluation.

The inspection script loads with `torch.load(..., map_location="cpu",
weights_only=True)`, then moves the model to the selected device, constructs
Adam using those model parameters and calls optimizer.load_state_dict.
PyTorch casts parameter-associated optimizer state to the parameter device;
the Adam step counter may remain on CPU for the ordinary non-capturable
optimizer. This is expected. Only load the checkpoint from this verified source.
See [PyTorch checkpoint loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

The model/Adam state supports continued optimization, but **the current
src.train CLI is CPU-only and has no resume flag**. `python -m src.train --full`
starts a fresh CPU model, even on a GPU machine; it is not a GPU resume command.
Before additional GPU training, a separate task must implement device/batch
transfer, explicit resume from epoch 100, an increased total epoch cap,
optimizer restoration, new run directories and parent-checkpoint provenance.
No GPU training, tuning or ensemble execution command is claimed to exist yet.

Python/NumPy/Torch RNG and shuffle-generator states were not saved, and no
scheduler state exists (no scheduler was used). CPU→GPU arithmetic and random
states can change the optimization path; an uninterrupted CPU run cannot be
reproduced exactly. Future checkpoints should save RNG, sampler and scheduler
states. Model selection must keep using validation; reserve Hold-out for STEP 10.

### Upload review

Inspect `git status --short --untracked-files=all` and the exact artifact list
above. Confirm whether this repository may share the competition split indices,
fitted scaler/statistics and model checkpoint. The configured origin is
`https://github.com/hyolim0118-bit/semiconductor-.git`; its visibility and remote
contents were not changed or inspected. Confirm destination branch and the
reviewed commit contents before publishing. Do not use `git add -f` on ignored
data. Review commit/push separately; no history rewrite is needed.

Local preparation verified 16 bundled files and 9 external files against the
manifest. Restore left all 13 existing destination files untouched. Original
CSV/array/scaler/split/result sizes and mtimes were unchanged during packaging.
Use the manifest as the reference again on the GPU host; local success alone
does not establish that the destination environment is compatible.
