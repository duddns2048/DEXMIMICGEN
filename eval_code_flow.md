# eval_pouring_201.sh 실행 시 코드 흐름

## 목적
`eval_pouring_201.sh` → `run_trained_agent.py` 실행 시
체크포인트에서 정책을 복원하고 환경에서 rollout하여 성공률을 측정하는
전체 코드 흐름을 정리한다.

---

## 진입점: 쉘 명령

```bash
python ./robomimic/robomimic/scripts/run_trained_agent.py \
 --agent  .../model_epoch_600.pth \
 --n_rollouts 50 \
 --horizon 400 \
 --seed 0 \
 --logdir .../eval/epoch_600
```

| 인자 | 의미 |
|------|------|
| `--agent` | 평가할 체크포인트(.pth) |
| `--n_rollouts` | 시뮬레이션 에피소드 수 |
| `--horizon` | 에피소드당 최대 step 수 |
| `--seed` | rollout 시드 |
| `--logdir` | 비디오와 결과 JSON 저장 폴더 |

---

## 전체 실행 흐름

```
run_trained_agent.py main()
  ├─ 1. 체크포인트 로드 → 정책 복원              run_trained_agent.py:70-75
  ├─ 2. Config 추출                              run_trained_agent.py:76
  ├─ 3. 환경(Env) 생성                            run_trained_agent.py:89-112
  ├─ 4. video / logdir 준비                       run_trained_agent.py:118-124
  ├─ 5. rollout_with_stats() — N번 에피소드 실행 run_trained_agent.py:126
  └─ 6. 결과 JSON 저장                           run_trained_agent.py:139-145
```

---

## 단계별 핵심 코드 위치

### 1. 체크포인트 로드 & 정책 복원
- **파일**: `utils/file_utils.py:380` `policy_from_checkpoint()`
- 한 줄 호출(`run_trained_agent.py:75`)에서 다음을 모두 수행:
  - `.pth` 파일을 torch.load → `ckpt_dict` 획득
  - `algo_name_from_checkpoint` (`file_utils.py:235`) → `"bc"` 추출
  - `config_from_checkpoint` (`file_utils.py:343`) → 학습 시 config 그대로 복원
  - `ObsUtils.initialize_obs_utils_with_config` — obs 모달리티(rgb/low_dim) 등록
  - `env_metadata`, `shape_metadata` 추출 — 환경 생성에 필요한 정보
  - obs/action normalization stats 복원
  - `algo_factory` 호출 → `BC_RNN` 인스턴스화 (train.py와 동일 경로)
  - `model.deserialize(ckpt_dict["model"])` → 학습된 가중치 로드
  - **`RolloutPolicy`로 래핑하여 반환** (`algo.py:477`)
- 반환된 `rollout_model`은 `policy(ob)` 형태로 직접 호출 가능

### 2. 환경 생성
- **파일**: `run_trained_agent.py:89-112` `create_env_helper()`
- `env_meta`(체크포인트에 저장된 환경 메타)와 `shape_meta`로 생성
- 핵심 호출: `EnvUtils.create_env_from_metadata` (`run_trained_agent.py:95`)
  - `env_meta["env_name"]` → 예: `"TwoArmPouring"`
  - `render_offscreen=True` → 비디오 저장용 오프스크린 렌더링
  - `use_image_obs`, `use_depth_obs` → 카메라 obs 사용 여부
- `dexmimicgen` import 부수효과로 환경이 robosuite에 등록되어야 함 (`run_trained_agent.py:55`)
- `--n_envs > 1`이면 `SubprocVectorEnv`로 병렬 평가 (`run_trained_agent.py:106-109`)

### 3. Rollout 실행: `rollout_with_stats`
- **파일**: `utils/train_utils.py:442` `rollout_with_stats()`
- 한 환경에 대해 `num_episodes`(`--n_rollouts`)번 반복:
  - 비디오 writer 준비 (`train_utils.py:516-526`)
  - `run_rollout()` 호출 (`train_utils.py:249`) — 한 에피소드 실행
  - 성공/실패, return 카운트 누적 (`train_utils.py:576-577`)
- 모든 에피소드 완료 후 평균 메트릭 집계 (`train_utils.py:587-597`)
- 반환: `all_rollout_logs` (성공률, return, horizon) + `video_paths`

### 4. 한 에피소드: `run_rollout`
- **파일**: `utils/train_utils.py:249`
- 핵심 루프 (`train_utils.py:301-388`):

```
ob_dict = env.reset()                       # 시뮬레이터 초기화
policy.start_episode()                       # RNN hidden state 리셋 (bc.py:596)

for step_i in range(horizon):                # 최대 horizon 스텝
    ac = policy(ob=ob_dict)                  # 정책 추론 → 액션
    ob_dict, r, done, info = env.step(ac)    # 시뮬레이터 1 step
    success = success | info["is_success"]   # 성공 누적
    if video_writer:                         # 비디오 프레임 저장
        video_writer.append(env.render(...))
    if done or (terminate_on_success and success["task"]):
        break                                # 조기 종료
```

- 반환: `{"Return": ..., "Horizon": ..., "Success_Rate": 0 or 1, ...}`

### 5. 정책 추론 한 스텝: `RolloutPolicy.__call__`
- **파일**: `algo/algo.py:537`
- 매 step 호출 시 흐름:
  1. `_prepare_observation` (`algo.py:505`) — numpy → tensor, batchify, device 이동, 정규화
  2. `policy.get_action(obs_dict)` 호출 — 알고리즘별 (BC_RNN의 경우 `bc.py:564`)
  3. BC_RNN `get_action`의 동작 (**핵심**):
     - **첫 step 또는 `_rnn_horizon` 스텝마다** hidden state 리셋 (`bc.py:577-584`)
     - 그 외 step은 직전 hidden state를 이어받아 RNN 1 step만 진행 (`forward_step`)
     - LSTM의 sequence-by-sequence 추론을 매 스텝 hidden state로 효율화
  4. 액션 후처리 — numpy 변환, 정규화 해제, 6D rotation 변환 (`algo.py:560-576`)
  5. 환경에 넣을 수 있는 action vector 반환

### 6. 결과 저장
- **비디오**: `$logdir/videos/{env_name}_epoch_*.mp4` (`train_utils.py:516`)
- **메트릭 JSON**: `$logdir/TwoArmPouring.json` (`run_trained_agent.py:144`)
  ```json
  {
    "Return": ...,
    "Success_Rate": 0.66,
    "Horizon": ...,
    "Num_Episodes": 50,
    ...
  }
  ```

---

## 학습 vs 평가 흐름 차이

| 단계 | 학습(train.py) | 평가(run_trained_agent.py) |
|------|----------------|----------------------------|
| 모델 생성 | config → `algo_factory` | 체크포인트 → `policy_from_checkpoint` (내부에서 `algo_factory` + 가중치 로드) |
| 데이터 | HDF5 dataset 로딩 | 사용 안 함 |
| 학습 루프 | `run_epoch` (loss/backward) | 없음 |
| Rollout | N epoch마다 평가용 | **메인 작업** |
| 옵티마이저 | Adam step 매 배치 | 없음 (eval 모드 고정) |
| 저장 | checkpoint .pth | 비디오 + 통계 JSON |

→ 평가는 학습 루프의 **rollout 부분만 추출한 형태**라고 볼 수 있다.

---

## 코드 읽는 순서 추천

**A. 체크포인트 → 정책 복원 (어떻게 학습된 모델이 살아나는가)**
```
run_trained_agent.py:70-76
  → file_utils.py:380 policy_from_checkpoint
  → algo.py:477 RolloutPolicy.__init__
```

**B. Rollout 한 에피소드 (정책이 환경과 어떻게 상호작용하는가)**
```
train_utils.py:442 rollout_with_stats
  → train_utils.py:249 run_rollout
  → algo.py:537 RolloutPolicy.__call__
  → bc.py:564 BC_RNN.get_action
```

**C. 비디오/메트릭 저장 (결과가 어디에 어떻게 남는가)**
```
train_utils.py:516-526  (video writer 생성)
train_utils.py:587-597  (메트릭 집계)
run_trained_agent.py:139-145  (JSON 저장)
```
