# 학습 루프 상세 흐름 (BC-RNN 기준)

`train_code_flow.md`의 "학습 루프" 부분을 한 batch가 dataset에서 시작되어
모델 파라미터가 업데이트되기까지의 **데이터 흐름·tensor shape·loss·optimizer step**
관점에서 자세히 정리한 문서이다.

기본 가정 (DexMimicGen `bc_rnn_image.json` 류 config 기준):
- `train.batch_size = B` (예: 16)
- `train.seq_length = T = 10` (= `algo.rnn.horizon`)
- `algo.rnn.hidden_dim = H = 400`, `algo.rnn.num_layers = 2`
- `algo.actor_layer_dims = [1024, 1024]` (RNN 뒤 MLP)
- 관측: low-dim (예: `robot0_eef_pos (3,)`, `robot0_eef_quat (4,)`, …) + 이미지 (예: `agentview_image (3, 84, 84)`)
- 행동 차원: `ac_dim = A` (양손 14차원 또는 normalize 후 차원)

---

## 0. 한눈에 보는 전체 한 스텝

```
DataLoader
  │   batch = {"data": {"obs": {...}, "actions": [B,T,A], ...}}
  ▼
process_batch_for_training         ─ bc.py:513   (BC_RNN)
  │   - seq 유지, open-loop이면 obs[:,0]만 복제
  ▼
postprocess_batch_for_training     ─ algo.py:208 (이미지 normalize, uint8→float)
  │
  ▼
train_on_batch                     ─ bc.py:119
  │  ├─ _forward_training          ─ bc.py:546   (RNN 순전파, actions [B,T,A])
  │  ├─ _compute_losses            ─ bc.py:167   (L2+L1+Cos 합) / bc.py:664 (GMM NLL)
  │  └─ _train_step                ─ bc.py:200   ─→ backprop_for_loss (torch_utils.py:168)
  ▼
log_info                            ─ bc.py:219
```

---

## 1. DataLoader → batch dict

### 1.1 SequenceDataset에서 만드는 한 샘플
- 코드: `utils/dataset.py:441` `get_item()`
- 한 샘플은 한 demo 안의 연속한 `seq_length=T` 길이 구간을 잘라낸 것:

```python
meta["obs"][k]    # shape (T, *obs_shape_k), e.g. low-dim: (T, 3)  이미지: (T, 3, 84, 84)
meta["actions"]   # shape (T, A)        ← action_keys 들이 concat & normalize 됨 (dataset.py:514-518)
meta["goal_obs"]  # (선택) shape (*obs_shape_k)   ← seq dim 없음 (dataset.py:499)
meta["index"]     # scalar
```

### 1.2 DataLoader가 B개를 묶은 결과
- `train_utils.py:797` 에서 `iter(data_loader[k])`, `train_utils.py:805` 에서 `next()`
- 외부에 `"data"` 키 한 겹이 더 붙음 (`bc.py:111` `batch["data"]` 가정과 일치):

```python
batch = {
  "data": {
    "obs":       { "robot0_eef_pos": [B, T, 3],
                   "agentview_image": [B, T, 3, 84, 84], ... },
    "actions":   [B, T, A],
    "goal_obs":  None  (또는 {key: [B, *shape]}),
    "index":     [B],
  }
}
```

---

## 2. `process_batch_for_training` — RNN용 시퀀스 유지
파일: `algo/bc.py:513` (`BC_RNN`)

```python
input_batch["obs"]      = batch["obs"]          # [B, T, ...]  ← T 그대로 유지
input_batch["goal_obs"] = batch.get("goal_obs", None)
input_batch["actions"]  = batch["actions"]      # [B, T, A]

if self._rnn_is_open_loop:                       # bc.py:534
    # obs 시퀀스를 첫 step만 가지고 T번 복제 → "처음 관측만 보고 T step 예측"
    obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
    input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=T, dim=1)
```

> 비교: 기본 `BC` (RNN 아님) 는 `bc.py:112` 에서 `obs[:,0,:]`, `actions[:,0,:]` 만 사용 (단일 step BC).

이후 `to_device → to_float` 변환 (`bc.py:544`).

---

## 3. `postprocess_batch_for_training` — 이미지/정규화
파일: `algo/algo.py:208`

- `obs`, `next_obs`, `goal_obs` 키에 대해 `ObsUtils.process_obs_dict` 적용
  - 이미지: `uint8 → float / 255.`, channel-last → channel-first
- `obs_normalization_stats` 가 주어지면 `(x - mean) / std`

shape 자체는 변하지 않음 — 단 이미지 dtype만 float32로 변경.

---

## 4. `train_on_batch` — 한 스텝의 핵심
파일: `algo/bc.py:119`

```python
with TorchUtils.maybe_no_grad(no_grad=validate):
    predictions = self._forward_training(batch)        # 4-1
    losses      = self._compute_losses(predictions, batch)  # 4-2
    if not validate:
        step_info = self._train_step(losses)           # 4-3 (backprop + step)
```

---

## 4-1. `_forward_training` — RNN 순전파
파일: `algo/bc.py:546` (`BC_RNN`)

```python
actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
predictions["actions"] = actions     # shape [B, T, A]
```

내부 호출 체인:

```
RNNActorNetwork.forward                policy_nets.py:666
  └─ RNN_MIMO_MLP.forward              obs_nets.py:769
        ├─ ObservationGroupEncoder    obs_nets.py:363
        │     ├─ (이미지) VisualCore + ResNet18Conv  → flat feature
        │     └─ (low-dim) flatten
        │     └─ concat                                → [B, T, D_obs]
        ├─ time_distributed(encoder)                   ← obs_nets.py:795
        ├─ RNN_Base.forward(LSTM, H=400, layers=2)    base_nets.py:401
        │                                                output: [B, T, 400]
        ├─ MLP(400 → 1024 → 1024)                     obs_nets.py:703 (per-step net)
        └─ ObservationDecoder Linear(1024 → A)        obs_nets.py:290
              → output dict {"action": [B, T, A]}
  └─ torch.tanh(actions["action"])     policy_nets.py:697   → 값은 [-1, 1]
```

| 단계 | tensor shape (예시, B=16, T=10) |
|------|----------------------------------|
| 입력 low-dim `eef_pos`         | `[16, 10, 3]` |
| 입력 image `agentview_image`   | `[16, 10, 3, 84, 84]` (process 후 float) |
| Encoder 출력 (concat된 flat)   | `[16, 10, D_obs]` (D_obs는 modality 합) |
| LSTM 출력                       | `[16, 10, 400]` |
| MLP 출력                        | `[16, 10, 1024]` |
| Decoder Linear 출력             | `[16, 10, A]` |
| `tanh` 후 최종 actions         | `[16, 10, A]` ∈ [-1, 1] |

> RNN hidden state는 학습 시 매 batch마다 0으로 초기화 (`RNN_Base.forward`, `base_nets.py:419-420`).
> Rollout 시에는 `BC_RNN._rnn_hidden_state` 로 따로 관리됨 (`bc.py:577-593`).

### GMM 변형 (`BC_RNN_GMM`)
- `_forward_training`: `bc.py:636` → `RNNGMMActorNetwork.forward_train` (`policy_nets.py:831`)
- Decoder 출력 3개: `mean [B,T,M,A]`, `scale [B,T,M,A]`, `logits [B,T,M]` (`policy_nets.py:820-829`, `M=num_modes`)
- 분포: `MixtureSameFamily(Categorical(logits), Independent(Normal(mean, scale), 1))`
  - `batch_shape = [B, T]`, `event_shape = [A]`
- 예측 결과는 분포 자체이며, `log_prob(actions)` shape: `[B, T]`

---

## 4-2. `_compute_losses` — 손실 계산

### (a) `BC_RNN` (deterministic, default)
파일: `algo/bc.py:167`

```python
a_target = batch["actions"]     # [B, T, A]
actions  = predictions["actions"]  # [B, T, A]

losses["l2_loss"]    = nn.MSELoss()(actions, a_target)            # scalar
losses["l1_loss"]    = nn.SmoothL1Loss()(actions, a_target)        # scalar
losses["l1_ns_loss"] = nn.L1Loss()(actions, a_target)              # scalar
losses["cos_loss"]   = LossUtils.cosine_loss(actions[..., :3],     # eef 위치 3D 방향 정합
                                              a_target[..., :3])    # loss_utils.py:11

action_loss = w_l2*l2 + w_l1*l1 + w_l1ns*l1_ns + w_cos*cos
losses["action_loss"] = action_loss     # ← backprop 대상 (bc.py:197)
```

가중치는 config의 `algo.loss.{l2_weight, l1_weight, l1_ns_weight, cos_weight}` (`bc.py:189-194`).

> 모든 timestep을 동등하게 supervise: shape `[B, T, A]` 위에서 `MSELoss`가 평균을 내므로
> "T개 step을 다 맞추도록" 학습됨. (Transformer 변형에서는 `supervise_all_steps`로 마지막 step만 쓰는 옵션이 있음.)

### (b) `BC_RNN_GMM`
파일: `algo/bc.py:664`

```python
log_probs   = dists.log_prob(batch["actions"])  # [B, T]
action_loss = -log_probs.mean()                  # NLL, scalar
```

---

## 4-3. `_train_step` — 역전파 & optimizer
파일: `algo/bc.py:200`

```python
policy_grad_norms = TorchUtils.backprop_for_loss(
    net=self.nets["policy"],
    optim=self.optimizers["policy"],
    loss=losses["action_loss"],
)
```

내부 (`torch_utils.py:168` `backprop_for_loss`):

```python
optim.zero_grad()              # 이전 grad 초기화
loss.backward()                # autograd → 모든 파라미터에 .grad 채움
# (옵션) torch.nn.utils.clip_grad_norm_(...)
grad_norms = Σ ||p.grad||^2  for p in net.parameters()
optim.step()                   # Adam 등으로 파라미터 업데이트
return grad_norms
```

- Optimizer는 `Algo._create_optimizers()` (`algo.py:160` 근처)에서 만들어짐 — config의 `algo.optim_params.policy.learning_rate.initial` 등으로 Adam 생성.
- LR 스케줄러는 epoch 끝에 `on_epoch_end` (`algo.py:293`)에서 step.

업데이트되는 파라미터: `self.nets["policy"]` 의 모든 서브모듈 — 이미지 encoder(ResNet18 포함), LSTM, MLP, Decoder Linear. (encoder freeze 옵션이 없는 한 전부 학습됨.)

---

## 5. `run_epoch` 의 step 루프
파일: `utils/train_utils.py:759`

```python
for _ in range(num_steps):                     # train_utils.py:798
    batch = next(data_loader_iter[k])           # 2.
    input_batch = model.process_batch_for_training(batch)        # 3.
    input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats)
    info        = model.train_on_batch(input_batch, epoch, validate=validate)  # 4.
    step_log    = model.log_info(info)
    step_log_all.append(step_log)
```

- `num_steps`: `train.num_data_workers`, `train.batch_size`, `train.num_epochs`, 그리고 `experiment.epoch_every_n_steps` 등 config 값으로 결정됨 — 명시되지 않으면 `len(data_loader)` 전체.
- epoch 끝에 timing/loss 평균을 dict로 반환 (`train_utils.py:830-844`).

상위 `train()` 루프에서 호출되는 위치: `scripts/train.py:295` (train), `:336` (val), `:373` (rollout).

---

## 6. 입력 → 출력 → loss 한눈 요약 표

| 항목 | 학습 시 | 추론 시 (`get_action`) |
|------|---------|------------------------|
| obs 입력 shape | `[B, T, ...]` | `[B, 1, ...]` (또는 step별 `[B, ...]` → `to_sequence` 후 `[B, 1, ...]`) |
| RNN hidden | 매 batch 0으로 리셋 | `_rnn_hidden_state` 에 누적, `horizon`마다 리셋 (`bc.py:577`) |
| 네트워크 출력 | `[B, T, A]` (deterministic) / GMM `MixtureSameFamily` | `[B, A]` (`forward_step`, `policy_nets.py:704`) |
| 활성화 | `tanh` (det) / `tanh(mean)` (GMM) | 동일 |
| Loss | L2+L1+Cos / `-log_prob.mean()` | — |
| Update | `Adam.step()` on `nets["policy"]` 전체 | — |

---

## 7. 추천 코드 읽기 순서 (이 흐름을 따라가려면)

1. **batch 모양**: `utils/dataset.py:441` `get_item` (한 샘플) → `utils/train_utils.py:797` (배치 묶음)
2. **시퀀스 보존**: `algo/bc.py:513` `BC_RNN.process_batch_for_training`
3. **이미지 후처리**: `algo/algo.py:208` `postprocess_batch_for_training`
4. **순전파 진입점**: `algo/bc.py:546` → `models/policy_nets.py:666` `RNNActorNetwork.forward`
5. **공유 구조**: `models/obs_nets.py:769` `RNN_MIMO_MLP.forward` (encoder → RNN → MLP → decoder)
6. **LSTM 본체**: `models/base_nets.py:401` `RNN_Base.forward`
7. **손실**: `algo/bc.py:167` (det) / `bc.py:664` (GMM)
8. **업데이트**: `algo/bc.py:200` → `utils/torch_utils.py:168` `backprop_for_loss`
9. **루프 컨테이너**: `utils/train_utils.py:759` `run_epoch`
