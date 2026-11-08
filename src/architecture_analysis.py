"""Produce an evidence-based report from saved best-validation predictions only."""
import csv
import json
from pathlib import Path
import numpy as np


def analyze(run_dir):
    run_dir = Path(run_dir)
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    target = np.load(str(root / 'outputs/preprocessed/y_validation.npy'), mmap_mode='r')
    indices = np.load(str(run_dir / 'validation_indices.npy'), allow_pickle=False)
    results = sorted(summary['results'], key=lambda r: r['best_selection_validation_mae'])
    residuals, checks, histories = {}, [], {}
    reference = config['historical_reference']['validation_mae']
    lines = ['# STEP 9-6 후속: A/B/D/E 전체 구조 비교', '',
             'Adam, LR 0.0003, batch 1024, 최대 200 epoch, patience 15, seed 42, CPU 4 threads. '
             '모든 모델을 새로 초기화했으며 동일한 Train 648,000행 / Validation 81,000행을 사용했다. '
             'Hold-out은 사용하지 않았다. 전처리·분할·기존 체크포인트와 Smoke Test 결과는 보존했다.', '',
             '순위는 학습 중 최고 Validation MAE 기준이다. 아래 정밀 MAE는 Best Checkpoint 예측을 float64로 집계하여 '
             '학습 중 float32 loss 집계와 미세한 반올림 차이가 있다. 단일 seed 결과이므로 통계적 우열을 확정하지 않는다.', '',
             '| 순위 | 모델 | 파라미터 | 완료 epoch | 최적 epoch | 선택 MAE | 정밀 MAE | MAPE % | 학습 분 | 관측 최대 RSS MiB | 기존 대비 MAE 차이 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rank, r in enumerate(results, 1):
        directory = (root / r['checkpoint']).parent
        pred = np.load(str(directory / 'validation_predictions.npy'), allow_pickle=False)
        assert pred.shape == target.shape == (len(indices), 4)
        assert np.isfinite(pred).all()
        error = pred.astype(np.float64) - target
        residuals[r['id']] = error
        mae = np.abs(error).mean(axis=0)
        mape = (np.abs(error) / np.maximum(np.abs(target.astype(np.float64)), 1e-8)).mean(axis=0) * 100
        assert np.allclose(mae, [r[f'layer_{i}_mae'] for i in range(1,5)], atol=1e-9, rtol=0)
        assert np.allclose(mape, [r[f'layer_{i}_mape_percent'] for i in range(1,5)], atol=1e-8, rtol=0)
        with (directory / 'history.csv').open(encoding='utf-8') as stream:
            history = list(csv.DictReader(stream))
        histories[r['id']] = history
        best = min(history, key=lambda row: float(row['validation_mae']))
        assert int(best['epoch']) == r['best_epoch']
        assert float(best['validation_mae']) == r['best_selection_validation_mae']
        peak = max(float(row['working_set_mib']) for row in history)
        checks.append(dict(id=r['id'], prediction_shape=list(pred.shape), metrics_verified=True,
                           history_best_verified=True, observed_epoch_max_rss_mib=peak))
        lines.append(f"| {rank} | {r['id']} {r['name']} | {r['parameters']:,} | {r['epochs_completed']} | {r['best_epoch']} | {r['best_selection_validation_mae']:.10f} | {r['validation_mae']:.8f} | {r['validation_mape_percent']:.5f} | {r['training_loop_seconds']/60:.2f} | {peak:.2f} | {r['validation_mae']-reference:+.6f} |")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    panels, axes = plt.subplots(2, 2, figsize=(12, 8))
    for panel, r in zip(axes.flat, results):
        history = histories[r['id']]
        epochs = [int(row['epoch']) for row in history]
        validation = [float(row['validation_mae']) for row in history]
        training = [float(row['training_mae']) for row in history]
        ax.plot(epochs, validation, label=r['id']+' '+r['name'])
        panel.plot(epochs, training, label='Train (during updates)')
        panel.plot(epochs, validation, label='Validation')
        panel.axvline(r['best_epoch'], color='gray', linestyle=':', label='Best epoch')
        panel.set(title=r['id']+' '+r['name'], xlabel='Epoch', ylabel='MAE')
        panel.legend(fontsize=8)
    ax.axhline(reference, color='black', linestyle='--', label='Historical reference (different budget)')
    ax.set(xlabel='Epoch (fresh initialization)', ylabel='Validation MAE', title='Equal-budget architecture comparison')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(run_dir/'comparison.png'), dpi=150)
    plt.close(fig)
    panels.tight_layout()
    panels.savefig(str(run_dir/'learning_curves_comparison.png'), dpi=150)
    plt.close(panels)
    lines += ['', f'기존 최적 MLP: **{reference:.10f}**, epoch 244. LR 탐색/연속 학습과 학습 예산이 달라 참고값으로만 표시하고 위 순위에서 제외했다. 음수 차이는 기존보다 MAE가 낮음을 뜻한다.', '',
              '학습 시간은 epoch loop(학습·Validation·체크포인트 저장)를 측정하며 최종 재평가/그림 저장은 제외한다. '
              'RSS는 epoch 끝에서 관측한 프로세스 Working Set의 최댓값이며 데이터 매핑과 런타임을 포함한다. '
              'memory_comparison.csv의 process peak는 프로세스 누적 최고치로 모델 고유 메모리와 구분해야 한다.', '',
              '| 모델 | 전체 MAE / MAPE % | 층 1 MAE / MAPE % | 층 2 MAE / MAPE % | 층 3 MAE / MAPE % | 층 4 MAE / MAPE % |',
              '|---|---:|---:|---:|---:|---:|']
    for r in results:
        parts = [f"{r['validation_mae']:.6f} / {r['validation_mape_percent']:.4f}"]
        parts += [f"{r[f'layer_{i}_mae']:.6f} / {r[f'layer_{i}_mape_percent']:.4f}" for i in range(1,5)]
        lines.append('| '+r['id']+' | '+' | '.join(parts)+' |')
    lines += ['', 'MAPE = mean(abs(pred-y)/max(abs(y),1e-8))×100. 전체 지표는 네 층의 동일 가중 평균이다.', '',
              '![Validation 학습 곡선](comparison.png)', '',
              '![모델별 Train/Validation 곡선](learning_curves_comparison.png)', '',
              '모델별 학습/Validation 곡선은 각 모델 폴더의 learning_curve.png, 전체 epoch 기록은 history.csv에 저장했다.', '',
              '## 구조 해석', '']
    lookup = {r['id']:r for r in results}
    for a,b,text in [('A','B','B는 A보다 얕고 폭도 좁다. 깊이·폭·파라미터 수가 함께 변하므로 깊이만의 효과가 아니다.'),
                     ('A','D','D는 A에 256-unit 은닉층을 추가한다. 깊이 증가와 파라미터 증가가 동반되므로 깊이 단독 효과로 해석할 수 없다.'),
                     ('A','E','E와 A의 파라미터 수는 비슷하지만 E는 일정 폭, Residual 연결, BatchNorm, Dropout, head 구조가 함께 달라진다. Residual 연결만의 효과는 분리할 수 없다.')]:
        if a in lookup and b in lookup:
            delta = lookup[b]['validation_mae'] - lookup[a]['validation_mae']
            lines.append(f"- {b} − {a} MAE = {delta:+.6f}. {text}")
    lines += ['', '## 앙상블 후보와 오차 상관관계', '',
              '저장된 Validation 예측에서 부호 있는 잔차(pred−y)의 Pearson 상관계수를 계산했다. '
              '전체는 모든 층의 잔차를 펼친 값이며 층별 상관계수도 함께 저장한다. 낮은 상관만으로 앙상블 개선이 보장되지는 않는다.', '',
              '| 모델 쌍 | 전체 잔차 상관 | 층 1 | 층 2 | 층 3 | 층 4 | 단순 평균 MAE |',
              '|---|---:|---:|---:|---:|---:|---:|']
    pairs=[]
    for i,a in enumerate(results):
        for b in results[i+1:]:
            ea,eb=residuals[a['id']],residuals[b['id']]
            corr=float(np.corrcoef(ea.ravel(),eb.ravel())[0,1])
            layers=[float(np.corrcoef(ea[:,j],eb[:,j])[0,1]) for j in range(4)]
            ensemble=float(np.abs((ea+eb)/2).mean())
            pairs.append(dict(a=a['id'],b=b['id'],residual_correlation=corr,
                              layer_correlations=layers,equal_weight_validation_mae=ensemble))
            lines.append(f"| {a['id']} + {b['id']} | {corr:.5f} | "+' | '.join(f'{v:.5f}' for v in layers)+f' | {ensemble:.6f} |')
    if pairs:
        bestpair=min(pairs,key=lambda p:p['equal_weight_validation_mae'])
        winner=results[0]
        lines += ['',f"개별 최우수 모델은 {winner['id']}. 단순 평균 후보 중 최저 MAE는 {bestpair['a']}+{bestpair['b']} "
                  f"({bestpair['equal_weight_validation_mae']:.6f}); 개별 최우수 대비 차이는 "
                  f"{bestpair['equal_weight_validation_mae']-winner['validation_mae']:+.6f}. "
                  '같은 Validation에서 선택한 탐색적 결과이며 별도 일반화 검증은 수행하지 않았다. 가중치 최적화나 stacking은 실행하지 않았다.']
    if results:
        lines += ['', f"C/F 추가 판단의 기준: 이번 동등 예산 최저 MAE는 {results[0]['id']}의 {results[0]['validation_mae']:.6f}이다."]
    if 'A' in lookup and 'D' in lookup:
        if lookup['D']['validation_mae'] < lookup['A']['validation_mae']:
            lines += ['D가 A보다 낮은 MAE를 보여 용량 확대 실험 C를 추가할 근거가 있다. 다만 폭 확장의 개선은 별도로 확인해야 한다.']
        else:
            lines += ['D의 깊이·용량 증가가 A보다 낮은 MAE로 이어지지 않아, C의 폭 확대도 자동으로 유리하다고 볼 수 없다.']
    if 'E' in lookup and results:
        if lookup['E']['validation_mae'] <= min(r['validation_mae'] for r in results if r['id'] != 'E'):
            lines += ['E가 이번 개별 성능에서 경쟁력이 있어 F의 추가 용량을 확인할 우선순위가 높다.']
        else:
            lines += ['E의 개별 MAE가 최우수는 아니므로 F는 E의 잔차 다양성·평균 예측 개선과 추가 CPU 비용을 함께 보고 결정한다.']
    lines += ['', '## C/F 후속 실험', '',
              'C는 A와 같은 은닉층 수에서 폭을 키우므로 용량 증가의 방향을 확인하는 후속 후보이고, '
              'F는 E보다 폭과 Residual block 수가 모두 증가하므로 Residual 계열의 용량 증가를 확인하는 후보이다. '
              '둘 다 단일 요인 ablation은 아니다. 11월 10일 비교에는 우선 완료된 A/B/D/E를 사용하고, '
              '추가 CPU 예산이 확보되면 동일 프로토콜로 C/F를 수행한다. '
              'Residual 연결의 인과 효과를 확인하려면 동일 폭·깊이·BN·Dropout에서 skip 유무만 바꾼 별도 실험이 필요하다.', '',
              '## 저장 및 검증', '',
              'validation_indices.npy의 행 순서와 각 모델의 validation_predictions.npy (float32, 81000×4)가 대응한다. '
              '열 순서는 layer_1, layer_2, layer_3, layer_4이며 원래 타깃 단위다. '
              'summary.json, comparison.csv, memory_comparison.csv, 각 best.pt·history.csv·summary.json을 보존한다.', '',
              f"실패 기록: {json.dumps(summary['failures'],ensure_ascii=False)}. 원본/전처리 보존 확인: {summary['protected_sources_unchanged']}."]
    (run_dir/'STEP_9_6_followup_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (run_dir/'prediction_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (run_dir/'ensemble_diagnostics.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
    print(json.dumps(dict(results=[{k:r[k] for k in ('id','validation_mae','validation_mape_percent','best_epoch','epochs_completed','training_loop_seconds')} for r in results],pairs=pairs),indent=2))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    analyze(parser.parse_args().run_dir)
