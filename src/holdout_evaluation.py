"""Fixed clean Hold-out evaluation and read-only inputs for STEP 11."""
from datetime import datetime
import csv
import shutil
import statistics
import time
import uuid

import joblib
import numpy as np
import torch

from .final_candidates import (ROOT, NAMES, FrozenCandidates, comparison_rows,
                               identity, read_json, save_json, write_csv, plot_comparison)
from .data_loader import CHANNELS, TARGETS
from .preprocessing import _load_labeled_chunks, _load_split_indices, transform_features

MANIFEST = ROOT / 'outputs/ensemble_comparison/20261108_224103_c493d55d/model_manifest.json'


def load_holdout():
    """Return saved standardized inputs, unscaled targets and original row indices."""
    splits, _ = _load_split_indices(ROOT / 'outputs/splits/data_split_indices.npz')
    x = np.load(ROOT / 'outputs/preprocessed/X_holdout_test_scaled.npy', allow_pickle=False)
    y = np.load(ROOT / 'outputs/preprocessed/y_holdout_test.npy', allow_pickle=False)
    indices = splits['holdout_test']
    if x.shape != (81000, 226) or y.shape != (81000, 4) or len(indices) != len(y):
        raise ValueError('Unexpected Hold-out dimensions')
    if x.dtype != np.float32 or y.dtype != np.float32 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid Hold-out arrays')
    return x, y, indices, len(splits['train'])


def load_raw_holdout(indices, chunksize=20000):
    """Extract CSV data-row positions, restoring the exact requested index order."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices)==0 or np.any(indices<0) or len(np.unique(indices))!=len(indices):
        raise ValueError('Invalid original row indices')
    order = np.argsort(indices, kind='stable')
    sorted_indices = indices[order]
    x = np.empty((len(indices),226),dtype=np.float32)
    y = np.empty((len(indices),4),dtype=np.float32)
    start, found = 0, 0
    for frame in _load_labeled_chunks(ROOT / 'train.csv', chunksize):
        end = start+len(frame)
        lo, hi = np.searchsorted(sorted_indices, [start,end])
        if hi>lo:
            positions = order[lo:hi]
            selected = frame.iloc[sorted_indices[lo:hi]-start]
            x[positions] = selected[CHANNELS].to_numpy(dtype=np.float32)
            y[positions] = selected[TARGETS].to_numpy(dtype=np.float32)
            found += hi-lo
        start=end
        if start>sorted_indices[-1]:
            break
    if found!=len(indices) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Raw Hold-out extraction failed')
    return x,y


def evaluate_predictions(y, predictions):
    """STEP 9-7 definitions: pooled element MAE/MAPE and inclusive true-target bins."""
    return comparison_rows(y,predictions)


def diagnostic_plots(run,y,predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for row,name in enumerate(NAMES):
        for layer in range(4):
            ax=axes[row,layer]
            density=ax.hexbin(y[:,layer],predictions[name][:,layer],gridsize=55,mincnt=1,bins='log')
            ax.plot([10,300],[10,300],color='red',lw=1)
            ax.set(title=f'{name} / Layer {layer+1}',xlabel='True thickness',ylabel='Predicted thickness')
            fig.colorbar(density,ax=ax,label='Count (log scale)')
    fig.tight_layout(); fig.savefig(run / 'actual_vs_predicted.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    stats=[]; cases=[]
    for layer in range(4):
        errors=[predictions[n][:,layer]-y[:,layer] for n in NAMES]
        edges=np.histogram_bin_edges(np.concatenate(errors),bins=100)
        abs_edges=np.histogram_bin_edges(np.abs(np.concatenate(errors)),bins=100)
        for name,error in zip(NAMES,errors):
            absolute=np.abs(error)
            axes[0,layer].hist(error,bins=edges,histtype='step',label=name)
            axes[1,layer].hist(absolute,bins=abs_edges,histtype='step',label=name)
            stats.append(dict(candidate=name,layer=layer+1,mean_signed_error=float(error.mean()),
                              median_absolute_error=float(np.median(absolute)),
                              p90_absolute_error=float(np.quantile(absolute,.9)),
                              p95_absolute_error=float(np.quantile(absolute,.95)),
                              p99_absolute_error=float(np.quantile(absolute,.99)),max_absolute_error=float(absolute.max())))
            for position in np.argsort(-absolute,kind='stable')[:10]:
                cases.append(dict(candidate=name,layer=layer+1,holdout_position=int(position),
                                  actual=float(y[position,layer]),prediction=float(predictions[name][position,layer]),
                                  signed_error=float(error[position]),absolute_error=float(absolute[position])))
        for row in range(2):
            axes[row,layer].set(title=f'Layer {layer+1}',xlabel='Signed error' if row==0 else 'Absolute error',ylabel='Count',yscale='log')
            axes[row,layer].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(run / 'layer_error_distributions.png',dpi=160); plt.close(fig)
    return stats,cases


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    run=ROOT / 'outputs/holdout_evaluation' / (datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    print(run.relative_to(ROOT),flush=True)
    predictor=FrozenCandidates(MANIFEST)
    sources=[MANIFEST,ROOT/'train.csv',ROOT/'outputs/splits/data_split_indices.npz',
             ROOT/'outputs/preprocessed/X_holdout_test_scaled.npy',ROOT/'outputs/preprocessed/y_holdout_test.npy',
             MANIFEST.parent/'validation_summary.csv']
    sources += [ROOT/r['path'] for r in predictor.manifest['members']+predictor.manifest['preprocessing_artifacts']+[predictor.manifest['common_epoch200_checkpoint']]]
    before=[identity(p) for p in sources]
    shutil.copyfile(MANIFEST,run/'model_manifest.json')
    config=read_json(ROOT/'outputs/preprocessors/preprocessing_config.json')
    assert config['feature_columns']==CHANNELS and config['target_columns']==TARGETS
    assert config['target_scaling'] is False and config['feature_scaler_fit_split']=='train only'
    x,y,indices,n_train=load_holdout()
    scaler=joblib.load(ROOT/'outputs/preprocessors/x_scaler.joblib')
    assert scaler.n_features_in_==226 and int(scaler.n_samples_seen_)==n_train
    zeros=(y==0).sum(axis=0).tolist()
    if any(zeros):
        raise ValueError('Zero targets: standard MAPE undefined; no denominator substitution')
    print('Verifying every raw Hold-out row and saved scaler output...',flush=True)
    raw,raw_y=load_raw_holdout(indices)
    np.testing.assert_array_equal(raw_y,y)
    np.testing.assert_array_equal(transform_features(raw,scaler=scaler),x)
    for filename,array in [('holdout_indices.npy',indices),('holdout_raw_reflectance.npy',raw),('holdout_targets.npy',y)]:
        np.save(run/filename,array,allow_pickle=False)
    del raw,raw_y
    print('Raw mapping and transform exactly match. Running fixed candidates...',flush=True)
    predictions={n:predictor.predict(n,x) for n in NAMES}
    times={n:[] for n in NAMES}
    for repeat in range(5):
        for name in (NAMES if repeat%2==0 else NAMES[::-1]):
            start=time.perf_counter(); current=predictor.predict(name,x); times[name].append(time.perf_counter()-start)
            np.testing.assert_array_equal(current,predictions[name])
    overall,bins=evaluate_predictions(y,predictions)
    with (MANIFEST.parent/'validation_summary.csv').open(encoding='utf-8-sig',newline='') as stream:
        validation={r['candidate']:r for r in csv.DictReader(stream)}
    for row in overall[:2]:
        name=row['candidate']
        row.update(model_count=1 if name==NAMES[0] else 3,inference_median_seconds=statistics.median(times[name]))
        for metric in ['mae','mape_percent']:
            baseline=float(validation[name][metric]); row['validation_'+metric]=baseline
            row[metric+'_holdout_minus_validation']=row[metric]-baseline
            row[metric+'_change_from_validation_percent']=(row[metric]-baseline)/baseline*100
    write_csv(run/'overall_metrics.csv',overall[:2]); write_csv(run/'layer_metrics.csv',overall[2:])
    write_csv(run/'thickness_bin_metrics.csv',bins)
    for name in NAMES:
        np.save(run/(name+'_holdout_predictions.npy'),predictions[name],allow_pickle=False)
    plot_comparison(run,overall,bins)
    stats,cases=diagnostic_plots(run,y,predictions)
    for case in cases:
        case['source_csv_data_row_index']=int(indices[case['holdout_position']])
    write_csv(run/'error_distribution_summary.csv',stats); write_csv(run/'large_error_cases.csv',cases)
    settings=dict(split='holdout_test',rows=len(y),input_shape=list(x.shape),target_shape=list(y.shape),
                  feature_columns=CHANNELS,target_columns=TARGETS,target_scaling=False,zero_target_counts=zeros,
                  mape='mean(abs(pred-y)/abs(y))*100; no epsilon, masking or clipping',
                  bins='Closed intervals 10-100, 110-200, 210-300; each true layer target independently; pooled counts are target elements',
                  primary_metric='MAE',auxiliary_metric='MAPE (%)',device='cpu',threads=4,batch_size=1024,
                  torch_version=str(torch.__version__),timing_seconds=times,
                  timing_protocol='One full warmup per candidate; five alternating full-set repeats; median. Includes predict validation, batching, output assembly and FP64 arithmetic mean. Excludes loading, preprocessing, metrics.',
                  source_artifacts=before,no_training=True,no_noise_generated=True,no_model_reselection=True,
                  raw_mapping='train.csv zero-based data-row positions excluding header; exact equality of all targets and transformed inputs verified',
                  scaler_train_rows=n_train,raw_cache=identity(run/'holdout_raw_reflectance.npy'),
                  reused_functions=['FrozenCandidates.predict','load_holdout','load_raw_holdout','transform_features','evaluate_predictions'])
    save_json(run/'config.json',settings)
    print('Checking source integrity...',flush=True)
    assert before==[identity(p) for p in sources]
    save_json(run/'verification.json',dict(all_sources_unchanged=True,raw_targets_exact_match=True,
              raw_transform_exact_match=True,repeated_predictions_identical=True,zero_target_counts=zeros,
              original_split_partition_verified=True,scaler_sample_count_matches_training=True))
    lines=['# STEP 10: Clean Hold-out evaluation','',
           '고정된 두 후보를 동일한 Hold-out 81,000행에서 평가했다. MAE가 주 지표, MAPE가 보조 지표다. 정답 0은 네 층 모두 없고 분모 보정은 없다. 추가 학습, 재분할, Scaler fit, 모델 재선택, 잡음 생성은 수행하지 않았다.','',
           '## 전체·층별 성능','', '| 범위 | 단일 MAE | 앙상블 MAE | MAE 개선 % | 단일 MAPE % | 앙상블 MAPE % | MAPE 개선 % |','|---|---:|---:|---:|---:|---:|---:|']
    for a,b in zip(overall[::2],overall[1::2]):
        lines.append(f"| {a['scope']} | {a['mae']:.8f} | {b['mae']:.8f} | {b['mae_improvement_percent']:.4f} | {a['mape_percent']:.8f} | {b['mape_percent']:.8f} | {b['mape_percent_improvement_percent']:.4f} |")
    lines+=['','개선율=(단일−앙상블)/단일×100。 음수는 악화다。','', '## Validation 대비 변화','']
    for r in overall[:2]:
        lines.append(f"- {r['candidate']}: MAE 변화 {r['mae_holdout_minus_validation']:+.8f} ({r['mae_change_from_validation_percent']:+.4f}%), MAPE 변화 {r['mape_percent_holdout_minus_validation']:+.8f}%p ({r['mape_percent_change_from_validation_percent']:+.4f}%).")
    lines+=['','## 두께 구간별 성능','','구간은 각 층의 실제 두께 기준 닫힌 구간이다. count는 타깃 원소 수이며 구간 밖 값은 없다.','',
            '| 범위 | 구간 | Count | 단일 MAE | 앙상블 MAE | 단일 MAPE % | 앙상블 MAPE % |','|---|---|---:|---:|---:|---:|---:|']
    for a,b in zip(bins[::2],bins[1::2]):
        lines.append(f"| {a['scope']} | {a['thickness_range']} | {a['target_count']} | {a['mae']:.6f} | {b['mae']:.6f} | {a['mape_percent']:.6f} | {b['mape_percent']:.6f} |")
    lines+=['','## 악화된 항목','']
    for r in overall[1::2]+bins[1::2]:
        for metric in ['mae','mape_percent']:
            if r[metric+'_improvement_percent']<0:
                lines.append(f"- {r['scope']} {r.get('thickness_range','전체')}: {metric} 개선율 {r[metric+'_improvement_percent']:.6f}%.")
    lines+=['','## 추론 시간','']
    for r in overall[:2]:
        lines.append(f"- {r['candidate']} ({r['model_count']} models): {r['inference_median_seconds']:.6f}초 / 81,000행.")
    lines+=['','CPU 4 threads, batch 1024, 1회 warmup 후 5회 교차 순서 측정 중앙값. 로딩·전처리·지표 계산은 제외했다.','',
            '## 오차 분포와 큰 오차 사례','','error_distribution_summary.csv에 signed bias, 절대오차 중앙값·P90·P95·P99·최댓값을 저장했다. large_error_cases.csv에는 모델별·층별 절대오차 상위 10개와 원본 CSV 행 인덱스를 기록했다. 극단 사례는 사후 진단이며 모델 선택에 사용하지 않았다.','']
    for s in stats:
        lines.append(f"- {s['candidate']} Layer {s['layer']}: P95 {s['p95_absolute_error']:.4f}, P99 {s['p99_absolute_error']:.4f}, max {s['max_absolute_error']:.4f}, bias {s['mean_signed_error']:.4f}.")
    lines+=['','![성능 비교](mae_mape_comparison.png)','','![구간 비교](thickness_range_comparison.png)','','![실제값·예측값](actual_vs_predicted.png)','','![오차 분포](layer_error_distributions.png)','',
            '## STEP 11 준비','','원본 train.csv의 저장 인덱스로 전체 Hold-out을 추출해 모든 정답과 Scaler 변환 결과가 저장 배열과 정확히 일치함을 확인했다. Scaler의 학습 표본 수는 기존 Train 행 수와 일치하고 기존 전처리 기록은 train-only fit이다. 이번에 Scaler를 fit하지 않았다. 원본 반사율·정답·인덱스 캐시와 SHA-256을 저장했으므로 같은 순서로 재사용할 수 있다.','',
            '```python','from src.final_candidates import FrozenCandidates','from src.holdout_evaluation import load_holdout, load_raw_holdout, evaluate_predictions','from src.preprocessing import transform_features',
            f'candidates = FrozenCandidates("{(run/"model_manifest.json").relative_to(ROOT).as_posix()}")',
            'x_scaled, y, indices, _ = load_holdout()',
            'predictions = {name: candidates.predict(name, x_scaled) for name in ("single_mlp", "lr_mean_ensemble")}',
            'overall, bins = evaluate_predictions(y, predictions)','```','',
            '잡음 실험에서는 원본 반사율에 잡음을 적용한 다음 transform_features를 사용해야 한다. 잡음 수준·반복 횟수·시드는 이번에 결정하거나 실행하지 않았다. 두 후보와 동일 비중 구성은 STEP 9-7 그대로 유지한다. 서로 독립 초기화한 Deep Ensemble이 아니라 공통 epoch-200 모델의 LR별 후속 체크포인트 평균이다.']
    (run/'STEP_10_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'run':run.relative_to(ROOT).as_posix(),'overall':overall[:2]},flush=True)


if __name__=='__main__':
    main()
