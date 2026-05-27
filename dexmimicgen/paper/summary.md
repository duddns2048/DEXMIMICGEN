# DexMimicGen 논문 요약

> Jiang et al., *DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning*, ICRA 2025.

## 1. 핵심 아이디어 (2-3문장)
DexMimicGen은 **양손 dexterous manipulation(이중 팔 + 다지 손) 시나리오**에서 소수(약 60개)의 사람 시연만으로 21K개의 학습 데이터를 자동 생성하는 시스템이다. MimicGen의 단일 팔 데이터 증강 아이디어를 확장하여, 두 팔이 **독립적으로(parallel)**, **동기화되어(coordination)**, 또는 **순서대로(sequential)** 동작해야 하는 세 가지 subtask 유형을 분리하고, 각 팔마다 별도의 action queue를 두는 비동기 실행 + 동기화 + 순서 제약 메커니즘으로 양팔 협조 동작을 자동 합성한다.

## 2. 주요 기여도 (bullet points)
- **DexMimicGen 시스템**: 양손/다지손 manipulation에 특화된 자동 데이터 생성 파이프라인. 비동기 per-arm 실행, 동기화, 순서 제약 등 핵심 설계 요소 도입.
- **9개 시뮬레이션 환경**: 3가지 embodiment(양손 Panda + 평행 그리퍼, 양손 Panda + dexterous hand, GR1 휴머노이드 + dexterous hand)와 3가지 협조 유형을 포괄. 60개 source demo로 **21K 데모 생성**.
- **데이터 생성 / 정책 학습 선택지의 영향 분석**: dataset 크기, transformation 방식(Transform vs Replay), 순서 제약 사용 여부, 정책 아키텍처 비교.
- **Real2Sim2Real 파이프라인**: 실제 Fourier GR1 휴머노이드에 대해 디지털 트윈 기반으로 데이터를 합성, **Can Sorting 태스크에서 실세계 90% 성공률** 달성 (source demo만으로는 0%).
- 데이터셋·시뮬레이션 환경 공개로 후속 연구 기반 제공.

## 3. 방법론 (알고리즘/기술 핵심만)
**Subtask 유형별 처리 (3가지):**
- **Parallel subtask**: 두 팔이 독립적인 sub-goal 수행. 팔마다 별도 subtask 시퀀스 `{S^a1_i}, {S^a2_j}`를 정의하고 **per-arm action queue**로 비동기 실행. 한 큐가 비면 다음 subtask의 변환된 trajectory로 채움.
- **Coordination subtask**: 두 팔의 상대 pose가 source demo와 일치해야 하는 구간. (1) source segmentation 단계에서 두 팔의 coordination subtask 끝을 같은 timestep에 맞춤. (2) 실행 중 먼저 끝난 팔이 다른 팔을 기다리는 **synchronization 전략**. (3) 두 팔 trajectory에 **동일한 SE(3) 변환** 적용.
- **Sequential subtask**: pre-subtask(예: 공 따르기)가 끝나야 post-subtask(예: 그릇 옮기기)로 진입하는 **ordering constraint** 강제.

**Trajectory 변환:** MimicGen과 동일하게 SE(3) equivariance 활용. 참조 객체의 source pose `T^{o_i}_W`와 현재 pose의 상대 변환 `T^{o_i'}_W (T^{o_i}_W)^{-1}`을 source trajectory에 적용. **Transform scheme**(객체 기준 변환) vs **Replay scheme**(원본 그대로 재생) 두 가지를 지원하며, handover 등 운동학적 제약이 빡빡한 단계에서는 Replay가 유리.

**컨트롤러:** Panda는 OSC(Operational Space Control)로 delta EEF → joint torque. 휴머노이드는 단일 torso 기반 IK 컨트롤러(mink 라이브러리)로 글로벌 EEF target → joint position. 손가락은 joint position 직접 제어 + source의 finger trajectory replay.

## 4. 실험 결과 (주요 성과)
- **Source demo 대비 압도적 향상** (BC-RNN, 1000 demo, Table I):
  - Drawer Cleanup: 0.7% → **76.0%**
  - Threading: 1.3% → **69.3%**
  - Piece Assembly: 3.3% → **80.7%**
- **Demo-Noise baseline 대비 평균 58% 이상 우위** (Table III): noise 기반 증강만으로는 부족하며, object-centric 변환이 핵심.
- **데이터 크기 효과** (Fig. 5): 100→500→1000 demo에서 성능이 크게 오르지만, 1000→5000은 태스크에 따라 diminishing returns.
- **확장된 초기 상태 분포(D1, D2)에서도 일반화** (Table II): broader reset distribution 데이터셋에서도 안정적인 성능.
- **정책 아키텍처 비교** (Table I): **Diffusion Policy가 대체로 가장 강하지만, dexterous hand 태스크에서는 BC-RNN-GMM이 BC-RNN/DP보다 강하다** — RoboMimic 결과와 상반.
- **Real2Sim2Real**: 디지털 트윈에서 40개 생성 데모 → 실세계 Can Sorting **90% 성공**, source demo 4개만 사용 시 **0%**.
- **다른 벤치마크(BiGym)에서도 적용 성공**: FlipCup 29.1%, DishwasherLoadPlates 43.6%, CupBoardsCloseAll 76.4%.

## 5. 한계점 및 향후 연구
- **여전히 사람 source demo가 필요**: parallel-gripper는 태스크당 10개, dexterous hand는 5개. 완전한 zero-demo는 아님.
- **수동 subtask segmentation 의존**: per-arm subtask 경계를 heuristic 또는 사람 annotation으로 정의해야 함.
- **데이터 생성 성공률이 태스크 의존적**: 일부 태스크(예: BiGym FlipCup 29.1%)는 생성 성공률 자체가 낮아 비효율.
- **환경 reset은 여전히 사람 필요** (실세계 deployment 시).
- **데이터 크기의 수확 체감**: 1000→5000 demo에서 성능 정체. 더 큰 데이터에서의 활용 방안 미해결.
- **향후 연구 방향**: bimanual + dexterous 세팅에서 모방학습 알고리즘 차이 분석, 자동 segmentation, 더 다양한 embodiment 확장.

## 6. 코드 구현 시 주요 고려사항
- **Source demo segmentation**: 태스크별로 per-arm subtask 경계 + reference object를 manual하게 지정해야 함. 이 메타데이터는 `ep_meta`로 HDF5에 저장되며, replay 시 `set_ep_meta` 호출로 환경에 주입됨.
- **두 팔 action queue 동기화**: coordination subtask 진입 직전에 먼저 도착한 팔이 wait. 구현 시 팔별 step counter와 "이번 subtask가 coordination인가" 플래그 관리 필요.
- **변환 방식 선택 (Transform vs Replay)**: handover(Transport, Can Sorting) 같이 kinematic 제약이 빡빡한 단계는 Replay 권장 — Transform 적용 시 trajectory가 IK/충돌 한계를 벗어나 실패율이 급증.
- **Embodiment별 컨트롤러 분기**: Panda는 OSC(robosuite 기본), 휴머노이드 GR1은 **mink 기반 IK 컨트롤러**. 학습 config의 `action_keys`도 이에 맞춰 분기됨 — Panda는 `right_rel_pos + rot_axis_angle` (정규화 없음), 휴머노이드는 `right_abs_pos + rot_6d` (min-max 정규화). 섞으면 안 됨.
- **Finger 동작**: end-effector 변환과 무관하게 source의 joint 시퀀스를 그대로 replay. OmniH2O retargeter로 사람 손 pose → 로봇 joint angle 변환.
- **객체 pose 관측 가정 (A3)**: 데이터 생성 시 매 subtask 시작 직전 reference object pose를 정확히 알아야 함. 실세계 적용 시 GroundingDINO + RGB-D 기반 pose estimation 필요.
- **실패 trajectory 필터링**: 생성된 demo는 `_check_success()`로 검증 후 성공한 것만 저장. dataset 생성 스크립트에 success rate 로깅을 반드시 포함시킬 것.
- **Digital twin 정렬**: real2sim2real에서 head-mounted camera로 RGB-D + GroundingDINO 마스크 → depth 평균으로 객체 x, y 초기화. 시뮬레이션 환경의 placement initializer와 좌표계 매핑을 신중히 설계.
- **재현 평가 프로토콜**: paper 따라 **3 seeds, 시드별 max success rate** 보고. 본 repo의 27개 학습 설정(9 태스크 × 3 시드)이 정확히 이 프로토콜에 맞춰져 있음.

