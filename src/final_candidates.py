"""Frozen STEP 9-7 candidates and validation-only evaluation; never trains."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time
import uuid

import numpy as np
import torch

from .model import MLPRegressor
from .train import ROOT


LR_RUN = ROOT / "outputs/lr_comparison/20261107_132952_full_8457bb4f"
NAMES = ("single_mlp", "lr_mean_ensemble")
RANGES = ((10, 100), (110, 200), (210, 300))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    stat = path.stat()
    return dict(path=path.relative_to(ROOT).as_posix(), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, sha256=sha256(path))


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def make_manifest():
    parent = read_json(LR_RUN / "config.json")
    base = Path(parent["checkpoint"])
    assert sha256(base) == parent["checkpoint_sha256"]
    members = []
    for rate in (0.001, 0.0003, 0.0001):
        directory = LR_RUN / ("lr_" + str(rate))
        summary = read_json(directory / "summary.json")
        checkpoint = torch.load(str(directory / "best.pt"), map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        assert Path(config["checkpoint"]).resolve() == base.resolve()
        assert config["checkpoint_sha256"] == parent["checkpoint_sha256"]
        assert checkpoint["epoch"] == summary["best_epoch"]
        assert checkpoint["validation_mae"] == summary["validation_mae"]
        assert all(group["lr"] == rate for group in checkpoint["optimizer_state_dict"]["param_groups"])
        members.append(dict(id="lr_" + str(rate), learning_rate=rate,
                            best_epoch=checkpoint["epoch"], saved_validation_mae=checkpoint["validation_mae"],
                            architecture={k: config[k] for k in ("input_dim", "hidden_dims", "output_dim")},
                            training_config=config, **identity(directory / "best.pt")))
    artifacts = [identity(ROOT / "outputs/preprocessors" / name) for name in
                 ("x_scaler.joblib", "preprocessing_config.json", "training_channel_scaling_statistics.csv")]
    artifacts += [identity(ROOT / "src/model.py"), identity(ROOT / "src/config.py")]
    return dict(schema_version=1, members=members, common_epoch200_checkpoint=identity(base),
                candidates=dict(single_mlp=dict(members=["lr_0.0003"], weights=[1.0]),
                                lr_mean_ensemble=dict(members=[m["id"] for m in members], weights=[1/3]*3)),
                preprocessing_artifacts=artifacts, input_dtype="float32", input_shape=[None, 226],
                target_order=["layer_" + str(i) for i in range(1, 5)], target_scaling=False,
                ensemble_arithmetic="float64 mean of three float32 model outputs",
                lock_policy="Both candidates fixed for noise stability and STEP 10; do not alter composition or weights after hold-out evaluation.",
                provenance="Three continuations of the SAME epoch-200 MLP, not independent initializations.",
                selection_disclosure="Ensemble combinations were explored on Validation before this fixed two-candidate comparison.")


class FrozenCandidates:
    """Load verified fixed candidates; predict from already standardized inputs."""

    def __init__(self, manifest_path):
        self.manifest = read_json(Path(manifest_path))
        expected = dict(single_mlp=dict(members=["lr_0.0003"], weights=[1.0]),
                        lr_mean_ensemble=dict(members=["lr_0.001", "lr_0.0003", "lr_0.0001"], weights=[1/3]*3))
        if self.manifest["candidates"] != expected:
            raise ValueError("Candidate composition/weights differ from the fixed STEP 9-7 protocol")
        records = self.manifest["members"] + self.manifest["preprocessing_artifacts"] + [self.manifest["common_epoch200_checkpoint"]]
        for record in records:
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise ValueError("Artifact integrity mismatch: " + record["path"])
        self.models = {}
        for member in self.manifest["members"]:
            checkpoint = torch.load(str(ROOT / member["path"]), map_location="cpu", weights_only=True)
            architecture = member["architecture"]
            model = MLPRegressor(architecture["input_dim"], tuple(architecture["hidden_dims"]), architecture["output_dim"])
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite checkpoint parameters")
            self.models[member["id"]] = model.eval().requires_grad_(False)

    def predict(self, candidate, x_scaled, batch_size=1024):
        if candidate not in NAMES:
            raise ValueError("Unknown fixed candidate")
        x_scaled = np.asarray(x_scaled)
        if x_scaled.ndim != 2 or x_scaled.shape[1] != 226 or x_scaled.dtype != np.float32:
            raise ValueError("Expected float32 standardized inputs with shape (n, 226)")
        if batch_size < 1 or not np.isfinite(x_scaled).all():
            raise ValueError("Invalid batch size or nonfinite inputs")
        spec = self.manifest["candidates"][candidate]
        output = np.empty((len(x_scaled), 4), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(x_scaled), batch_size):
                x = torch.from_numpy(x_scaled[start:start+batch_size])
                # Keep each FP32 forward identical; aggregate in FP64 in fixed member order.
                predictions = [self.models[name](x).numpy().astype(np.float64) for name in spec["members"]]
                output[start:start+len(x)] = np.mean(predictions, axis=0)
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite predictions")
        return output


def metrics(target, prediction, mask=None):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Invalid target/prediction arrays")
    if np.any(target == 0):
        raise ValueError("Standard MAPE is undefined at zero targets")
    error = np.abs(prediction-target)
    relative = error / np.abs(target) * 100
    if mask is not None:
        error, relative = error[mask], relative[mask]
    count = int(error.size)
    return dict(target_count=count, mae=float(error.mean()) if count else None,
                mape_percent=float(relative.mean()) if count else None)


def comparison_rows(target, predictions):
    overall, by_range = [], []
    for scope, column in [("overall", None)] + [("layer_"+str(i+1), i) for i in range(4)]:
        y = target if column is None else target[:, column]
        for name in NAMES:
            p = predictions[name] if column is None else predictions[name][:, column]
            overall.append(dict(candidate=name, scope=scope, **metrics(y, p)))
            covered = np.zeros(y.shape, dtype=bool)
            for lower, upper in RANGES:
                mask = (y >= lower) & (y <= upper)
                covered |= mask
                by_range.append(dict(candidate=name, scope=scope, thickness_range=f"{lower}-{upper}",
                                     **metrics(y, p, mask)))
            if not covered.all():
                raise ValueError("Targets outside the three requested closed thickness intervals")
    scopes = ["overall"] + ["layer_"+str(i) for i in range(1,5)]
    range_names = [f"{a}-{b}" for a,b in RANGES]
    by_range.sort(key=lambda row: (scopes.index(row["scope"]), range_names.index(row["thickness_range"]), NAMES.index(row["candidate"])))
    for rows in (overall, by_range):
        for index in range(0, len(rows), 2):
            single, ensemble = rows[index:index+2]
            for key in ("mae", "mape_percent"):
                baseline = single[key]
                improvement = None if baseline is None or baseline == 0 else (baseline-ensemble[key])/baseline*100
                ensemble[key+"_improvement_percent"] = improvement
                single[key+"_improvement_percent"] = 0.0 if baseline is not None else None
    return overall, by_range


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_validation(manifest_path=None, output_root="outputs/final_validation"):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    run = ROOT / output_root / (datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    manifest = make_manifest() if manifest_path is None else read_json(manifest_path)
    save_json(run / "model_manifest.json", manifest)
    predictor = FrozenCandidates(run / "model_manifest.json")
    # Explicit Validation-only paths. Never discover or open hold-out arrays.
    sources = [ROOT / "outputs/preprocessed" / name for name in ("X_validation_scaled.npy", "y_validation.npy")]
    sources += [ROOT / "outputs/splits/data_split_indices.npz"]
    source_records = [identity(path) for path in sources]
    x = np.array(np.load(str(sources[0]), mmap_mode="r", allow_pickle=False), copy=True)
    y = np.array(np.load(str(sources[1]), mmap_mode="r", allow_pickle=False), copy=True)
    if x.shape != (81000, 226) or y.shape != (81000, 4):
        raise ValueError("Unexpected saved Validation shapes")
    zero_counts = (y == 0).sum(axis=0).tolist()
    if any(zero_counts):
        raise ValueError("Zero targets: standard MAPE cannot be computed")
    with np.load(str(sources[2]), allow_pickle=False) as splits:
        keys = [key for key in splits.files if "val" in key.lower()]
        if len(keys) != 1:
            raise ValueError("Ambiguous Validation index key")
        indices = splits[keys[0]]
    assert len(indices) == len(y) and len(np.unique(indices)) == len(y)
    np.save(str(run / "validation_indices.npy"), indices, allow_pickle=False)
    predictions, timings = {}, {name: [] for name in NAMES}
    # One full-set warm-up per candidate; alternate order for five timed repeats.
    for name in NAMES:
        predictions[name] = predictor.predict(name, x)
    for repetition in range(5):
        for name in (NAMES if repetition % 2 == 0 else NAMES[::-1]):
            started = time.perf_counter()
            current = predictor.predict(name, x)
            timings[name].append(time.perf_counter()-started)
            np.testing.assert_array_equal(current, predictions[name])
    for name in NAMES:
        np.save(str(run / (name+"_validation_predictions.npy")), predictions[name], allow_pickle=False)
    overall, ranges = comparison_rows(y, predictions)
    expected = dict(single_mlp=14.413521503778151, lr_mean_ensemble=14.075488109711364)
    for row in overall:
        if row["scope"] == "overall":
            assert abs(row["mae"]-expected[row["candidate"]]) < 1e-6, "Previous Validation MAE not reproduced"
    timing_rows = [dict(candidate=name, repetitions=5, median_seconds=statistics.median(timings[name]),
                        minimum_seconds=min(timings[name]), maximum_seconds=max(timings[name]),
                        median_ms_per_sample=statistics.median(timings[name])*1000/len(y)) for name in NAMES]
    write_csv(run / "overall_layer_comparison.csv", overall)
    write_csv(run / "thickness_range_comparison.csv", ranges)
    write_csv(run / "inference_timing.csv", timing_rows)
    settings = dict(split="validation", rows=len(y), target_order=manifest["target_order"], batch_size=1024,
                    device="cpu", threads=4, torch_version=str(torch.__version__), seed=42,
                    primary_metric="MAE", auxiliary_metric="MAPE percent", mape_definition="mean(abs(pred-y)/abs(y))*100; no epsilon", zero_target_counts=zero_counts,
                    thickness_intervals=RANGES, thickness_assignment="Each true target element independently; pooled overall and per-layer metrics; closed intervals, counts recorded, full coverage asserted.",
                    timing_protocol="Loaded models and resident standardized inputs; 1 full-set warmup each, 5 full-set repeats in alternating order; includes input validation/batch loop/output assembly/FP64 mean; excludes file/model loading, scaler transform and metric computation.",
                    inference_seconds=timings, source_artifacts=source_records,
                    previous_validation_mae=expected, no_training=True, no_new_combinations=True,
                    holdout_accessed=False, gaussian_noise_evaluated=False,
                    model_composition_frozen=True, selection_disclosure=manifest["selection_disclosure"])
    save_json(run / "evaluation_config.json", settings)
    plot_comparison(run, overall, ranges)
    write_report(run, manifest, overall, ranges, timing_rows)
    assert source_records == [identity(path) for path in sources]
    for record in manifest["members"] + manifest["preprocessing_artifacts"] + [manifest["common_epoch200_checkpoint"]]:
        assert identity(ROOT / record["path"]) == {k:record[k] for k in ("path", "size", "mtime_ns", "sha256")}
    save_json(run / "verification.json", dict(checkpoints_and_preprocessing_unchanged=True,
              validation_arrays_and_split_unchanged=True, previous_results_reproduced=True,
              prediction_repeats_identical=True, zero_target_counts=zero_counts, holdout_accessed=False))
    print(run.relative_to(ROOT).as_posix(), flush=True)
    print(json.dumps(dict(overall=overall[:2], timing=timing_rows), ensure_ascii=False, indent=2), flush=True)
    return run


def plot_comparison(run, overall, ranges):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for rows, filename, scopes in ((overall, "mae_mape_comparison.png", ["overall"]+["layer_"+str(i) for i in range(1,5)]),
                                  ([r for r in ranges if r["scope"] == "overall"], "thickness_range_comparison.png", [f"{a}-{b}" for a,b in RANGES])):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric, label in zip(axes, ("mae", "mape_percent"), ("MAE", "MAPE (%)")):
            for offset, name in zip((-0.18, 0.18), NAMES):
                ax.bar(np.arange(len(scopes))+offset, [r[metric] for r in rows if r["candidate"] == name], width=.36, label=name)
            ax.set_xticks(np.arange(len(scopes)))
            ax.set_xticklabels(scopes)
            ax.set(ylabel=label, title="Validation "+label)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(str(run / filename), dpi=160)
        plt.close(fig)


def write_report(run, manifest, overall, ranges, timings):
    lines = ["# STEP 9-7: 확정 단일 MLP와 3개 LR 평균 앙상블", "",
             "MAE를 주 지표, 표준 MAPE(%)를 보조 지표로 사용한다. 저장된 동일 Validation 81,000행·동일 순서만 평가했다. 타깃 0은 네 층 모두 0개다. MAPE에 epsilon이나 클리핑을 사용하지 않았다.", "",
             "단일 MLP는 LR 0.0003 Best Epoch 244, 앙상블은 LR 0.001/0.0003/0.0001의 Best Checkpoint 예측을 각각 1/3로 평균한다. 모델은 226→256→128→64→4, 은닉 ReLU, 출력 선형, 타깃 스케일링 없음이다.", "",
             "세 모델은 독립 초기화가 아니라 동일 epoch 200 MLP에서 분기한 연속 학습이다. 조합은 이전 Validation에서 탐색했으며, 이번에는 두 후보만 재평가하고 새 조합/가중치를 탐색하지 않았다. 이번 결과는 별도 미사용 데이터의 성능 추정이 아니다.", "",
             "## 전체 및 층별 성능", "", "| 범위 | 단일 MAE | 평균 MAE | MAE 개선 % | 단일 MAPE % | 평균 MAPE % | MAPE 개선 % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a,b in zip(overall[::2], overall[1::2]):
        lines.append(f"| {a['scope']} | {a['mae']:.8f} | {b['mae']:.8f} | {b['mae_improvement_percent']:.4f} | {a['mape_percent']:.6f} | {b['mape_percent']:.6f} | {b['mape_percent_improvement_percent']:.4f} |")
    worsened = [b["scope"] for a,b in zip(overall[::2], overall[1::2]) if b["mape_percent_improvement_percent"] < 0]
    lines += ["", "네 층 모두 MAE가 감소했다. MAPE가 악화한 층: "+", ".join(worsened)+". MAE 개선이 모든 층의 상대 오차 개선을 뜻하지 않으므로 두 지표를 구분한다.",
              "", "![MAE/MAPE 비교](mae_mape_comparison.png)", "", "## 두께 구간별 성능", "",
              "각 층의 실제 타깃값으로 닫힌 구간 10–100, 110–200, 210–300에 배정한다. overall은 해당 구간의 모든 타깃 원소를 합친 값이며, 한 샘플이 층마다 다른 구간에 속할 수 있다. 구간 밖 원소가 없는 것을 확인했다. count는 샘플 수가 아닌 타깃 원소 수다.", "",
              "| 범위 | 두께 구간 | Count | 단일 MAE | 평균 MAE | MAE 개선 % | 단일 MAPE % | 평균 MAPE % | MAPE 개선 % |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a,b in zip(ranges[::2], ranges[1::2]):
        lines.append(f"| {a['scope']} | {a['thickness_range']} | {a['target_count']} | {a['mae']:.6f} | {b['mae']:.6f} | {b['mae_improvement_percent']:.4f} | {a['mape_percent']:.5f} | {b['mape_percent']:.5f} | {b['mape_percent_improvement_percent']:.4f} |")
    lines += ["", "개선율 = (단일−앙상블)/단일×100. 음수는 악화다.", "", "![두께 구간 비교](thickness_range_comparison.png)", "", "## 추론 시간", "",
              "CPU 4 threads, batch 1024. 모델/데이터 로딩 및 Scaler 변환은 제외하고 입력 검사·순차 배치 추론·출력 조립·평균을 포함한다. 전체 Validation 1회 워밍업 후 5회 교차 순서 측정의 중앙값을 사용한다. 학습 시간과 구분해야 하며 현재 시스템 부하에 따른 실측값이다.", "",
              "| 후보 | 중앙값 초 / 81,000행 | 최소–최대 초 | ms/행 |", "|---|---:|---:|---:|"]
    for r in timings:
        lines.append(f"| {r['candidate']} | {r['median_seconds']:.6f} | {r['minimum_seconds']:.6f}–{r['maximum_seconds']:.6f} | {r['median_ms_per_sample']:.6f} |")
    lines += ["", "## 체크포인트 및 확정 사항", ""]
    for m in manifest["members"]:
        lines.append(f"- `{m['id']}`, Best Epoch {m['best_epoch']}: `{m['path']}`; SHA-256 `{m['sha256']}`.")
    lines += ["", "전체 설정·체크포인트 경로와 SHA-256은 model_manifest.json, 평가 입력과 설정은 evaluation_config.json, 무결성 점검은 verification.json에 저장한다. 예측 열 순서는 layer_1..layer_4, 행 순서는 validation_indices.npy다.", "",
              "단일 MLP와 앙상블을 모두 이후 잡음 안정성 평가 대상으로 확정한다. Hold-out 평가 이후 성능을 보고 후보 구성·가중치를 변경하지 않는다. Hold-out·Gaussian Noise·추가 학습은 이번에 수행하지 않았다.", "",
              "STEP 9-6은 구조 검증/Smoke Test/A 부분 학습의 예비 실험으로 종료했다. A/B/D/E 전체 비교는 미완료이며 A는 epoch 182까지 저장됐다. B/D/E의 신규 전체 학습 결과는 없다. 추가 구조 학습을 진행하지 않는다.", "",
              "## STEP 10 재사용", "", "프로젝트 루트에서 다음 명령은 고정 후보로 Validation만 다시 평가하며 새 결과 폴더를 만든다.", "", "```powershell",
              f'.venv/Scripts/python.exe -m src.final_candidates --manifest "{(run / "model_manifest.json").relative_to(ROOT).as_posix()}"', "```", "",
              "향후 STEP 10 평가 코드에서는 동일 manifest를 로드하고, 기존 Scaler로 변환한 float32 입력에 아래 API를 사용한다. 지금은 Hold-out이나 잡음 데이터를 로드하지 않는다.", "", "```python", "from src.final_candidates import FrozenCandidates", "import torch", "torch.set_num_threads(4)",
              f'candidates = FrozenCandidates("{(run / "model_manifest.json").relative_to(ROOT).as_posix()}")',
              'single_predictions = candidates.predict("single_mlp", X_scaled)',
              'ensemble_predictions = candidates.predict("lr_mean_ensemble", X_scaled)', "```", "",
              "입력 컬럼 순서는 outputs/preprocessors/preprocessing_config.json의 feature_columns, 학습된 Scaler는 outputs/preprocessors/x_scaler.joblib이다. 재학습/재적합하지 않는다."]
    (run / "STEP_9_7_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Reuse the frozen STEP 9-7 model manifest")
    parser.add_argument("--output-root", default="outputs/final_validation")
    args = parser.parse_args()
    evaluate_validation(args.manifest, args.output_root)
