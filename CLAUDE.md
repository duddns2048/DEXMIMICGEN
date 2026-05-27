# CLAUDE.md

This file provides guidance to Claude Code when working from the top-level `DEXMIMICGEN/` directory.

## Repository Structure

이 디렉토리는 세 개의 독립 패키지를 함께 관리하는 작업 루트입니다:

| 폴더 | 역할 |
|------|------|
| `dexmimicgen/` | DexMimicGen 환경 패키지 (9개 bimanual manipulation 환경, 데이터 다운로드·재생 스크립트) |
| `robomimic/` | BC-RNN 학습 코드 (`dexmimicgen` 브랜치). **학습 실행은 이 폴더에서** |
| `robosuite/` | 시뮬레이션 프레임워크 (robosuite upstream 포크, 환경의 기반) |

세 패키지 모두 `pip install -e .`로 설치되어 있으며 서로 의존합니다:
`robosuite` → `dexmimicgen` → `robomimic` (학습 시에만)

## Where to Run Commands

```bash
# 환경 테스트 / 데이터 다운로드·재생 → dexmimicgen/ 기준
cd dexmimicgen
python scripts/demo_random_action.py --env TwoArmThreading --render
python scripts/download_hf_dataset.py --tasks TwoArmBoxCleanup

# 학습 config 생성 → dexmimicgen/ 기준
python scripts/generate_training_config.py \
  --dataset_dir ./datasets \
  --config_dir  ./datasets/train_configs/bcrnn_action_dict \
  --output_dir  ./datasets/train_results/bcrnn_action_dict

# 학습 실행 → robomimic/ 기준
cd robomimic
python scripts/train.py --config /path/to/config.json
```

## Subdirectory CLAUDE.md Files

- `dexmimicgen/CLAUDE.md` — 환경 아키텍처, 데이터셋 포맷, 전체 task 목록, 의존성 버전 제약 등 상세 내용
- 코드 탐색 시 해당 CLAUDE.md가 자동으로 추가 로드됩니다.
