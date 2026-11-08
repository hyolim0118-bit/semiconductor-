# 실험 기록

## Status

STEP 9-6은 예비 실험으로 종료했다. A/B/D/E 전체 비교는 미완료이며 추가 구조 학습을 진행하지 않는다.
STEP 9-7 평가 완료: 단일 MLP와 확정 3개 LR 평균 앙상블을 모두 이후 잡음 안정성 평가 및 STEP 10의 고정 후보로 확정했다.
Hold-out 평가 후 성능을 보고 모델 구성·가중치를 변경하지 않는다.

## Workspace decisions

- 모델: 기존 A/B/D/E, 새로운 모델 구조·전처리·분할은 설계하지 않는다.
- 학습: Adam / L1Loss / LR 0.0003 / batch 1024 / 최대 200 epoch / patience 15 / seed 42 / CPU 4 threads.
- 평가: 저장된 Train/Validation 분할만 사용, Hold-out 미사용. MAE와 MAPE를 전체 및 4개 층별로 보고한다.
- 기존 최적 MLP 14.4135215038은 별도 학습 예산이므로 참고값으로만 표시한다.
- 기존 체크포인트·Smoke Test 결과는 유지하며 신규 모델에 로드하지 않는다.

## History

- STEP 9-6: A–F forward/backward 및 B/E 1 epoch Smoke Test 완료(기존 산출물).
- 후속 실험: `outputs/architecture_comparison/20261107_215548_full_5231182c`, A/B/D/E fresh full-budget 중단. A: 마지막/Best Epoch 182, MAE 20.8389014286. B/D/E 신규 전체 학습 미수행. 기존 구조 검증과 B/E Smoke Test 보존.

## Data understanding (EDA)

이번 작업은 기존 전처리 배열과 분할을 그대로 재사용한다. EDA/전처리 재생성은 요청 범위에 포함하지 않는다.

## Backlog

추가 구조 학습 및 새 앙상블 조합 탐색을 진행하지 않는다. Gaussian Noise 평가는 아직 미실행이다.
후속 작업은 고정된 단일 MLP와 3개 LR 평균 앙상블의 잡음 안정성 및 STEP 10 평가다.

## STEP 9-7 평가 및 후보 고정

- 출처: 사용자 지정 프로토콜. 새 학습·파이프라인 설계·분할 선택 없이 기존 체크포인트를 Validation에서 추론했다.
- 단일: LR 0.0003, Best Epoch 244. MAE 14.4135214647, MAPE 22.8725069088%.
- 앙상블: LR 0.001(epoch 249), 0.0003(epoch 244), 0.0001(epoch 250)의 동일 가중 평균. MAE 14.0754881097, MAPE 22.5518806761%.
- 개선: MAE 2.3452516851%, MAPE 1.4017975117%. 네 층 모두 MAE 개선, Layer 2/4 MAPE 소폭 악화.
- 세 모델은 동일 epoch-200 체크포인트에서 파생됐으며 독립 초기화 모델이 아니다.
- 조합을 이전 Validation에서 탐색했다는 사실을 명시하며, 이번에는 확정된 두 후보만 평가했다.
- 결과/모델 고정 manifest: outputs/final_validation/20261107_230149_b28d96f8/model_manifest.json.
- 상세 보고서: outputs/final_validation/20261107_230149_b28d96f8/STEP_9_7_report.md.
- 검증: 표준 MAPE의 0 타깃/구간 경계/동일 가중 평균 및 순서 테스트 3건 통과. 저장된 표·예측을 별도 재검산했고 이전 감사 예측과 정확히 일치했다.
- 전처리·Scaler·체크포인트·분할 보존 확인. Hold-out 및 Gaussian Noise 미접근/미평가.

## Pre-flight evidence (STEP 9-7)

JOURNAL 및 Workspace decisions 확인. 사용자 지정 평가·기록 모드이며 새 experiments/ 스크립트나 모델 설계는 없다.
학습/실행 승인: 이번 사용자 요청의 Validation 평가만 실행. 재학습·새 조합 탐색 없음.
평가 범위: 기존 PyTorch checkpoint 추론이므로 sklearn/skore splitter 변경은 해당 없음.
설계/EDA/새 외부 API 조회: 해당 없음. 기록·보고서·무결성 검증 완료.

## STEP 10 Clean Hold-out (2026-11-08)

- 사용자 지정 STEP 9-7 manifest의 두 후보를 유지했다. 학습·Scaler fit·분할 변경·모델 재선택·잡음 생성 없음.
- 단일: MAE 14.4612229648, MAPE 23.2505274849%. 앙상블: MAE 14.1080883739, MAPE 22.8981397812%.
- 앙상블 개선율: MAE 2.4419414026%, MAPE 1.5156116520%. 네 층 모두 MAE 개선, Layer 2/4 MAPE 악화.
- Validation 대비 MAE 변화: 단일 +0.330950%, 앙상블 +0.231610%.
- 전체 Hold-out 81,000행의 원본 CSV 정답 및 기존 Scaler 변환 입력이 저장 배열과 정확히 일치. 기존 소스·분할·체크포인트·Scaler 해시/크기/mtime 보존 확인.
- 결과: `outputs/holdout_evaluation/20261108_231434_d1635539/STEP_10_report.md`. 모델 manifest, 예측 배열, 원본 반사율·인덱스·정답 캐시 저장.
- 재사용: `src/holdout_evaluation.py`의 load_holdout/load_raw_holdout/evaluate_predictions, 기존 transform_features와 FrozenCandidates.predict. STEP 11 잡음 실험은 미실행.
- 검증: 관련 테스트 5개 통과. 저장된 두 예측 배열의 MAE/MAPE 독립 재계산 및 인덱스 순서 일치 확인.
