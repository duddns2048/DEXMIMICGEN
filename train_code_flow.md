# train.py 코드 흐름 및 BC-RNN 핵심 포인트

## 목적
`robomimic/scripts/train.py` 실행 시의 전체 코드 흐름을 파악하고,
BC-RNN 논문의 핵심 개념이 코드 어디에 구현되어 있는지 매핑한다.

---

## 전체 실행 흐름 (main → train)

```
train.py main()
  ├─ 1. Config 로드                        main():571
  ├─ 2. Dataset 로드                       train():204  → train_utils.py:load_data_for_training
  ├─ 3. 환경(Env) 생성 (rollout용)          train():133  → EnvUtils.create_env_from_metadata
  ├─ 4. 모델 생성                           train():172  → algo_factory()
  ├─ 5. 학습 루프 (epoch 반복)
  │     ├─ run_epoch() — 학습              train():295
  │     ├─ run_epoch() — validation        train():336  (선택)
  │     └─ rollout_with_stats() — 평가     train():373  (N epoch마다)
  └─ 6. 결과 저장 및 로그                   train():452
```

---

## 단계별 핵심 코드 위치

### 1. 모델 생성: `algo_factory`
- **파일**: `algo/algo.py:52`
- Config의 `algo_name: "bc"` → `bc.py`의 `algo_config_to_class()` 호출
- BC 변형 선택 분기: `bc.py:24-75`
  - `rnn=True` → `BC_RNN` 또는 `BC_RNN_GMM` 반환
- **BC_RNN 네트워크 생성**: `bc.py:492` `_create_networks()`
  - `PolicyNets.RNNActorNetwork` 인스턴스화
  - RNN hidden state 관련 변수 초기화 (`_rnn_hidden_state`, `_rnn_horizon`, `_rnn_counter`)

### 2. 데이터셋 로드: `SequenceDataset`
- **파일**: `utils/dataset.py:19`
- HDF5에서 시퀀스 단위로 샘플링
- `__getitem__` → `get_item()`: `dataset.py:433`
  - demo 내에서 `seq_length` 길이의 연속 구간 추출
  - obs shape: `[B, T, obs_dim]`, action shape: `[B, T, ac_dim]`
- 시퀀스 경계 처리 (패딩): `dataset.py:525` `get_sequence_from_demo()`

### 3. 학습 한 스텝: `run_epoch` → `train_on_batch`
- **파일**: `utils/train_utils.py:759`

```
run_epoch()
  ├─ process_batch_for_training()     bc.py:513   시퀀스 전처리 (open-loop 처리)
  ├─ postprocess_batch_for_training() algo.py     정규화 적용
  ├─ train_on_batch()                 bc.py:119   ← 핵심
  │     ├─ _forward_training()        bc.py:546   RNN 순전파 [B,T,ac_dim]
  │     ├─ _compute_losses()          bc.py:167   손실 계산
  │     └─ _train_step()             bc.py:200   역전파 + optimizer.step()
  └─ log_info()                                   메트릭 기록
```

### 4. 손실 함수: `_compute_losses`
- **파일**: `bc.py:167` (BC 기본) / `bc.py:664` (BC_RNN_GMM)

| 모델 | 손실 | 수식 |
|------|------|------|
| BC_RNN (deterministic) | L2 + L1 + Cosine 가중합 | `bc.py:189-196` |
| BC_RNN_GMM | Negative Log-Likelihood | `-mean(log_prob(action))` → `bc.py:672` |

- Cosine loss 구현: `utils/loss_utils.py:11`

### 5. 네트워크 순전파: `RNNActorNetwork`
- **파일**: `models/policy_nets.py:563`

```
obs_dict [B, T, obs_shape]
  → ObservationGroupEncoder      obs_nets.py:363   (이미지: ResNet18 CNN, low-dim: flatten)
  → Flat features [B, T, D]
  → RNN_Base (LSTM, hidden=400, layers=2)  base_nets.py:304
  → RNN output [B, T, 400]
  → MLP (1024 → 1024)
  → ObservationDecoder (Linear → ac_dim)   obs_nets.py:290
  → tanh squashing
  → actions [B, T, ac_dim]
```

- RNN 기본 설정 (변경 가능): `config/bc_config.py:81` (`rnn_hidden_dim=400`, `rnn_num_layers=2`)

### 6. Rollout 평가 (추론 시 hidden state 관리)
- **파일**: `bc.py:564` `get_action()`
- 매 step마다 `_rnn_counter` 증가
- `_rnn_horizon` 스텝마다 hidden state 리셋 (`bc.py:577-584`)
- Open-loop 모드: 최초 obs만 사용, 이후는 hidden state로만 예측 (`bc.py:586-589`)
- 실제 환경 루프: `train_utils.py:318` `run_rollout()`

---

## 논문 개념 ↔ 코드 매핑

| 논문 개념 | 코드 위치 |
|-----------|-----------|
| BC (행동 복제) 손실 | `bc.py:167` `_compute_losses` |
| RNN 시퀀스 처리 | `base_nets.py:304` `RNN_Base.forward()` |
| GMM 분포 (다봉 액션) | `policy_nets.py:831` `RNNGMMActorNetwork.forward_train()` |
| 시퀀스 데이터 샘플링 | `dataset.py:441` `get_item()` |
| Open-loop 추론 | `bc.py:513` `process_batch_for_training()` |
| Tanh 액션 스케일링 | `policy_nets.py:699` `RNNActorNetwork.forward()` |
| Action normalization | `algo.py:537` `RolloutPolicy.__call__()` |

---

## 코드 읽는 순서 추천

이해 목적에 따라 두 가지 경로 추천:

**A. 데이터 → 손실 (학습 원리 파악)**
```
dataset.py:441 → bc.py:513 → bc.py:546 → bc.py:167 → bc.py:200
```

**B. 네트워크 구조 파악**
```
bc.py:492 → policy_nets.py:563 → obs_nets.py:363 → base_nets.py:304
```

**C. 추론(Rollout) 흐름**
```
train_utils.py:318 → algo.py:537 → bc.py:564 → base_nets.py:358
```
