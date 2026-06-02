# BC-RNN 네트워크 구조 (TwoArmPouring 학습 설정 기준)

> Config: `bc_rnn_image_ds_two_arm_pouring_humanoid_D0_seed_201.json`
> 데이터: `datasets/generated/two_arm_pouring.hdf5`

---

## 1. 전체 개요 (한 장 요약)

```
┌──────────────────────────────────────────────────────────────────────┐
│                       INPUT  (한 timestep)                             │
│                                                                       │
│  RGB images (3개)                  Low-dim states (6개)               │
│  ─────────────────                ────Z───────────────                 │
│  agentview_image      [3,84,84]   robot0_right_eef_pos       [3]      │
│  eye_in_left_hand     [3,84,84]   robot0_right_eef_quat      [4]      │
│  eye_in_right_hand    [3,84,84]   robot0_right_gripper_qpos  [11]     │
│                                    robot0_left_eef_pos        [3]      │
│         │                          robot0_left_eef_quat       [4]      │
│         │                          robot0_left_gripper_qpos   [11]     │
│         ▼                                  │                          │
│  ┌─────────────────────┐                   │                          │
│  │  VisualCore × 3     │                   │                          │
│  │  ResNet18 + Spatial │                   │                          │
│  │  Softmax (32 kp)    │                   │                          │
│  │  → 64-dim per image │                   │                          │
│  └──────────┬──────────┘                   │                          │
│             │  192-dim                     │  36-dim                  │
│             └────────────────┬─────────────┘                          │
│                              │  concat                                │
│                              ▼                                        │
│                     [B, T, 228]                                       │
│                              │                                        │
│                              ▼                                        │
│              ┌─────────────────────────────┐                          │
│              │  LSTM (hidden=1000, 2층)     │                          │
│              │  sequence-to-sequence        │                          │
│              └──────────────┬──────────────┘                          │
│                             │ [B, T, 1000]                            │
│                             ▼                                         │
│              ┌─────────────────────────────┐                          │
│              │  Linear  (1000 → 20)         │  (actor_layer_dims=[])   │
│              │  + tanh                      │                          │
│              └──────────────┬──────────────┘                          │
│                             │                                         │
│                             ▼                                         │
│                       OUTPUT [B, T, 20]                               │
│                                                                       │
│  20 = right_abs_pos(3) + right_abs_rot_6d(6)                          │
│     + left_abs_pos(3)  + left_abs_rot_6d(6)                           │
│     + right_gripper(1) + left_gripper(1)                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Input 상세

### 2.1 학습 배치 모양

- `batch_size = 16`
- `seq_length = 10` (RNN horizon과 같음)
- `frame_stack = 1`

batch dict 구조:
```python
batch["obs"][k] : [B=16, T=10, *obs_shape_k]
batch["actions"]: [B=16, T=10, 20]
```

### 2.2 사용되는 observation (config의 `observation.modalities.obs`)

데이터셋(HDF5)에는 obs key가 25개 있지만 **학습에 실제 사용되는 건 9개**입니다.

#### Low-dim (총 36-dim)
| Key | 차원 | 의미 |
|-----|------|------|
| `robot0_right_eef_pos` | 3 | 오른팔 end-effector 위치 (xyz) |
| `robot0_right_eef_quat` | 4 | 오른팔 end-effector 자세 (quaternion) |
| `robot0_right_gripper_qpos` | 11 | 오른손 손가락 관절 위치 (dexterous hand 11 joint) |
| `robot0_left_eef_pos` | 3 | 왼팔 end-effector 위치 |
| `robot0_left_eef_quat` | 4 | 왼팔 end-effector 자세 |
| `robot0_left_gripper_qpos` | 11 | 왼손 손가락 관절 위치 |
| **합계** | **36** | |

#### RGB (84×84 컬러 이미지 3개)
| Key | 원본 shape (HDF5) | 학습 시 shape |
|-----|------------------|---------------|
| `agentview_image` | `(T, 84, 84, 3)` uint8 | `(B, T, 3, 84, 84)` float32 |
| `robot0_eye_in_left_hand_image` | 동일 | 동일 |
| `robot0_eye_in_right_hand_image` | 동일 | 동일 |

HDF5는 channels-last (HWC) + uint8이지만, ObsUtils가 `[0,1]` float + channels-first (CHW)로 변환합니다.

### 2.3 Goal observation

config의 `observation.modalities.goal`이 빈 리스트이므로 **goal-conditioned 아님** → goal 입력 없음.

---

## 3. Encoder 상세

### 3.1 RGB encoder: VisualCore × 3 (이미지마다 별도 weight)

`models/obs_core.py:VisualCore`

각 이미지 한 장의 처리 흐름:
```
[3, 84, 84] uint8/255
   │
   ▼  CropRandomizer (학습 시): 84 → 76 무작위 crop
[3, 76, 76]
   │
   ▼  ResNet18Conv  (backbone_class=ResNet18Conv, pretrained=False)
[512, h', w']   ← ResNet18의 마지막 conv 출력
   │
   ▼  SpatialSoftmax (num_kp=32)
[32, 2]         ← 32개 keypoint의 (x, y) 좌표
   │  flatten
[64]
   │
   ▼  Linear → feature_dimension=64
[64]
```

→ 이미지 1장 = **64-dim 벡터**, 3장이면 **192-dim**

**SpatialSoftmax**(논문 *Levine et al. 2016*)는 feature map을 확률 분포로 보고 32개의 (x,y) 기댓값을 keypoint로 추출 → 공간 정보를 보존하면서 차원을 강하게 압축합니다. 손/물체 위치 같은 시각적 단서를 잡는 데 효과적.

### 3.2 Low-dim encoder

config의 `encoder.low_dim.core_class = null` → **별도 처리 없이 그대로 concat**.

### 3.3 Concatenation

```
rgb_features:    [B, T, 192]   (3 × 64)
low_dim_features:[B, T, 36]
                  ↓ concat
combined:        [B, T, 228]   ← RNN 입력
```

구현: `models/obs_nets.py:ObservationGroupEncoder.forward()`

---

## 4. RNN backbone

`config.algo.rnn`:
```json
{
  "enabled": true,
  "horizon": 10,
  "hidden_dim": 1000,
  "rnn_type": "LSTM",
  "num_layers": 2,
  "open_loop": false,
  "kwargs": {"bidirectional": false}
}
```

`models/base_nets.py:RNN_Base` (line 304):
```python
nn.LSTM(input_size=228, hidden_size=1000, num_layers=2,
        batch_first=True, bidirectional=False)
```

### 4.1 학습 시 forward
- 입력: `[B=16, T=10, 228]`
- 출력: `[B=16, T=10, 1000]` — **모든 timestep의 hidden state**
- hidden_state h_0, c_0 = zeros (학습은 시퀀스를 처음부터 굴림)

### 4.2 추론(rollout) 시 forward (`bc.py:564` `get_action`)
- 매 step마다 **1 timestep만** 처리: `[1, 1, 228] → [1, 1, 1000]`
- hidden state를 외부에서 유지 (`_rnn_hidden_state`)
- **`rnn_horizon=10` step마다 hidden state 리셋** (`bc.py:577-584`)
  - 학습 시퀀스 길이가 10이므로, 추론도 같은 길이의 chunk로 끊어 진행
  - 너무 긴 시퀀스에서 drift가 누적되는 것을 막음
- `open_loop=False`이므로 매 step의 현재 obs를 입력으로 사용 (closed-loop)

---

## 5. Decoder (Action head)

`config.algo.actor_layer_dims = []` → **MLP 없이 바로 linear**

`models/obs_nets.py:ObservationDecoder` (line 290):
```python
Linear(1000, 20)
   ↓
tanh   ← [-1, 1]로 squashing
```

출력: `[B, T, 20]`

---

## 6. Output 상세 (Action layout)

`config.train.action_keys` 순서대로 concat된 20-dim 벡터:

| 인덱스 | Key | 차원 | 의미 | Normalization |
|--------|-----|------|------|---------------|
| 0–2 | `action_dict/right_abs_pos` | 3 | 오른팔 EEF 목표 위치 (xyz, world frame) | `min_max` (→ [-1,1]) |
| 3–8 | `action_dict/right_abs_rot_6d` | 6 | 오른팔 EEF 목표 자세 (Zhou et al. 6D 표현) | 없음 |
| 9–11 | `action_dict/left_abs_pos` | 3 | 왼팔 EEF 목표 위치 | `min_max` |
| 12–17 | `action_dict/left_abs_rot_6d` | 6 | 왼팔 EEF 목표 자세 | 없음 |
| 18 | `action_dict/right_gripper` | 1 | 오른손 그리퍼 명령 | `min_max` |
| 19 | `action_dict/left_gripper` | 1 | 왼손 그리퍼 명령 | `min_max` |

### 왜 6D rotation인가?

쿼터니언/axis-angle은 회전 공간에서 **불연속점**을 가져 학습에 불리합니다.
6D 표현(*Zhou et al., CVPR 2019*)은 3D 회전 행렬의 앞 두 컬럼(6 값)만 출력하고
Gram-Schmidt로 직교화해서 완전한 회전을 복원하는 연속·매끄러운 표현입니다 → 학습 안정성↑.

### Rollout 시 후처리 (`algo.py:537` `RolloutPolicy.__call__`)

```
network output [20]  
   ↓  vector_to_dict       — 키별로 쪼개기
   ↓  unnormalize          — min_max 역변환
   ↓  rot_6d_to_axis_angle — 6D → 3D axis-angle 변환 (각 팔)
   ↓  dict_to_vector       — 환경이 받는 최종 액션 벡터로 재조립
env.step(action)
```

→ 네트워크 출력은 **20-dim**이지만, 실제 환경이 받는 액션은 6D → 3D axis-angle로 줄어들어 **다른 차원** (HDF5의 `actions`는 24-dim 형태). 평가/디버깅 시 이 차이를 헷갈리지 마세요.

---

## 7. Loss

`config.algo.loss`:
```json
{
  "l2_weight": 1.0,   ← 사용
  "l1_weight": 0.0,
  "l1_ns_weight": 0.0,
  "cos_weight": 0.0
}
```

즉 **순수 MSE**:
```python
loss = mean((predicted_action[B,T,20] - target_action[B,T,20])^2)
```

GMM/Gaussian/VAE는 모두 `enabled: false`로 끄여 있어, **deterministic BC-RNN**입니다.

---

## 8. Training vs Inference shape 요약

| 단계 | obs[key] (low_dim) | obs[key] (rgb) | RNN input | RNN output | Action output |
|------|--------------------|-----------------|-----------|------------|---------------|
| **학습 한 배치** | `[16, 10, dim]` | `[16, 10, 3, 84, 84]` | `[16, 10, 228]` | `[16, 10, 1000]` | `[16, 10, 20]` |
| **Rollout 1 step** | `[1, dim]` | `[1, 3, 84, 84]` | `[1, 1, 228]` | `[1, 1, 1000]` | `[1, 20]` |

추론 시에는 unsqueeze로 time dim을 추가하고 `forward_step`이 마지막 step만 추출 (`policy_nets.py:704`).

---

## 9. 파라미터 카운트 대략 추산

| 컴포넌트 | 파라미터 수 (대략) |
|----------|-------------------|
| ResNet18 × 3 (가중치 공유 안 함) | ≈ 11M × 3 = 33M |
| SpatialSoftmax (학습 파라미터 거의 없음) | ~0 |
| VisualCore output linear (per image) | (32×2 → 64): 4.2K × 3 = 12.6K |
| LSTM (input=228, hidden=1000, 2층) | ≈ 4×(228+1000+1)×1000 + 4×(1000+1000+1)×1000 ≈ 12.9M |
| Decoder Linear (1000 → 20) | 20K |
| **합계** | **≈ 46M** |

대부분의 파라미터는 ResNet18 3개에 몰려있습니다.

---

## 10. 학습/추론 데이터 흐름 추적용 핵심 코드

| 단계 | 파일:라인 |
|------|-----------|
| 데이터셋 시퀀스 샘플링 | `utils/dataset.py:441` `get_item` |
| 배치 전처리 (open-loop 처리 포함) | `algo/bc.py:513` `process_batch_for_training` |
| 정규화 적용 | `algo/algo.py:postprocess_batch_for_training` |
| 네트워크 forward | `algo/bc.py:546` `_forward_training` → `models/policy_nets.py:666` `RNNActorNetwork.forward` |
| ObservationGroupEncoder | `models/obs_nets.py:363` |
| VisualCore (ResNet18 + SpatialSoftmax) | `models/obs_core.py:61` |
| LSTM | `models/base_nets.py:304` `RNN_Base` |
| Action decoder + tanh | `models/obs_nets.py:290` `ObservationDecoder` |
| Loss 계산 | `algo/bc.py:167` `_compute_losses` |
| Backward + optimizer step | `algo/bc.py:200` `_train_step` |
| Rollout get_action | `algo/bc.py:564` `BC_RNN.get_action` |
| Action 후처리 (6D→axis-angle) | `algo/algo.py:537` `RolloutPolicy.__call__` |

---

## 11. 실제 shape를 직접 확인하고 싶다면

`train.sh`를 debug 모드로 1 epoch만 실행하면 됩니다:

```bash
# launch.json의 "Train BC-RNN" config가 이미 --debug로 설정됨
# CLI라면:
python ./robomimic/robomimic/scripts/train.py \
  --config ./dexmimicgen/datasets/train_configs/bcrnn_action_dict/bc_rnn_image_ds_two_arm_pouring_humanoid_D0_seed_201_debug.json \
  --name debug --debugz
```

또는 `bc.py:546` `_forward_training`에 한 줄 추가:
```python
def _forward_training(self, batch):
    obs = batch["obs"]
    for k, v in obs.items():
        print(f"obs[{k}]: {v.shape}")
    actions = self.nets["policy"](obs_dict=obs, goal_dict=batch.get("goal_obs"))
    print(f"action out: {actions.shape}")
    return dict(actions=actions)
```
→ 첫 batch에서 모든 텐서 shape가 출력됩니다.
