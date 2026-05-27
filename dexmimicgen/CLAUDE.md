# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context Files

For paper-related questions (algorithm details, why a design choice was made, what the experiments showed), do **not** start by grepping the code or guessing from variable names. Use these files instead:

- **`paper/summary.md`** — Korean-language summary of the DexMimicGen paper, organized into 6 sections (핵심 아이디어 / 주요 기여도 / 방법론 / 실험 결과 / 한계점 / 코드 구현 시 주요 고려사항). **Read this first** — it is curated and faster than scanning the PDF.
- **`paper/DexMimicGen_ Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning (1).pdf`** — original ICRA 2025 paper. Consult this when `summary.md` does not have the specific detail (e.g. exact equation, full table numbers, figure context, reference citation). Use the `Read` tool with the `pages` parameter for targeted page ranges.

**Workflow**: `summary.md` 먼저 → 부족하면 원본 PDF의 해당 섹션만 추가로 확인.

## Project Overview

DexMimicGen is the official release of simulation environments and BC-RNN training pipeline for the ICRA 2025 paper *"DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning"* (NVlabs). The repo provides:

- Nine two-arm manipulation environments built on top of [robosuite](https://github.com/ARISE-Initiative/robosuite) (the `dexmimicgen` Python package).
- Scripts to download, play back, and train on HDF5 demonstration datasets hosted on HuggingFace (`MimicGen/dexmimicgen_datasets`).

## Required External Dependencies

The package is `pip install -e .`-able, but two pinned upstream forks must be installed separately *before* this repo for environments and training to work:

- **robosuite** — `git clone https://github.com/ARISE-Initiative/robosuite && pip install -e robosuite`. This package's environments subclass `robosuite.environments.manipulation.two_arm_env.TwoArmEnv` and rely on robosuite >= 1.4 APIs (`edit_model_xml`, composite controllers, `mjviewer` renderer).
- **robomimic (`dexmimicgen` branch)** — only needed for training: `git clone https://github.com/ARISE-Initiative/robomimic.git -b dexmimicgen && pip install -e .`. `scripts/generate_training_config.py` imports `robomimic.utils.hyperparam_utils.ConfigGenerator` and uses `robomimic/exps/templates/bc.json` as the base config.

`requirements.txt` pins `numpy==1.23.3` and `numba==0.56.4` — do not bump these casually; robosuite/MuJoCo bindings are sensitive to NumPy ABI.

## Common Commands

```bash
# Sanity check that environments load (random actions, on-screen viewer)
python scripts/demo_random_action.py --env TwoArmThreading --render
# Drop --render on a headless machine.

# Download datasets from HuggingFace (default: ./datasets/, all 9 tasks)
python scripts/download_hf_dataset.py --path /path/to/save/datasets
python scripts/download_hf_dataset.py --tasks TwoArmBoxCleanup  # single task

# Play back a demonstration (on-screen render, or video file when --render omitted)
python scripts/playback_datasets.py --dataset datasets/generated/two_arm_threading.hdf5 --n 1 --render
python scripts/playback_datasets.py --dataset xxx.hdf5 --use-actions      # open-loop action replay
python scripts/playback_datasets.py --dataset xxx.hdf5 --use-obs          # video from stored image obs (offline)
python scripts/playback_datasets.py --dataset xxx.hdf5 --use_current_model  # use current env XML instead of dataset's

# Convenience wrappers (set TASK env var before running run_replay.sh)
bash run_demo.sh                 # TwoArmThreading random action
TASK=coffee bash run_replay.sh   # replay datasets/generated/two_arm_${TASK}.hdf5

# Generate robomimic BC-RNN training configs from downloaded datasets
python scripts/generate_training_config.py \
  --dataset_dir ./datasets \
  --config_dir  ./datasets/train_configs/bcrnn_action_dict \
  --output_dir  ./datasets/train_results/bcrnn_action_dict

# Train (run from the robomimic repo, with a config produced above)
python scripts/train.py --config /path/to/config.json
```

Available task names (`--env` / suffix for downloaded HDF5s):
`TwoArmThreading`, `TwoArmThreePieceAssembly`, `TwoArmTransport` (Panda + parallel gripper);
`TwoArmBoxCleanup`, `TwoArmDrawerCleanup`, `TwoArmLiftTray` (Panda + dexterous hand `PandaDexRH`/`PandaDexLH`);
`TwoArmCoffee`, `TwoArmPouring` (humanoid `GR1FixedLowerBody`);
`TwoArmCanSortRandom` (humanoid `GR1ArmsOnly`).
The task→robot mapping lives in `ENV_ROBOTS` at `scripts/demo_random_action.py:25`.

There is no test suite. `pre-commit` is configured with `black` and `isort --profile black`; install it with `pre-commit install` if you intend to commit.

## Architecture

### Package layout (`dexmimicgen/`)

- `__init__.py` — imports every env class at package import time. **This import side effect is what registers the environments with robosuite** (`robosuite.make(env_name=...)`). Scripts must `import dexmimicgen` even if they never reference the symbol — see the comment in `scripts/demo_random_action.py:23`.
- `environments/` — one file per task, each subclassing `TwoArmDexMGEnv` (`two_arm_dexmg_env.py`), which itself subclasses robosuite's `TwoArmEnv`. The base class handles:
  - Per-robot base pose offsets for single-robot (humanoid) vs. dual-robot (two Pandas) setups (`_load_model`).
  - Object placement sampling on reset via `placement_initializer` (`_reset_internal`).
  - **XML mesh/texture path rewriting** (`edit_model_xml`): when a dataset's stored XML references `dexmimicgen/...` paths from another machine, this rewrites them to the local install location. Important when replaying datasets across machines.
  - `set_robot_state(init_state)` and `get_state()` for deterministic state save/restore.
- `models/objects/` — task-specific MuJoCo objects (`composite/`, `composite_body/`, `xml_objects.py` for coffee machine, drawers, etc.). Imported by env classes.
- `models/assets/` — meshes and textures referenced by the object XMLs. `models/__init__.py` exposes `assets_root` for path resolution.
- `utils/config_utils.py` — robomimic BC-RNN hyperparameter helpers (`set_learning_settings_for_bc_rnn`, etc.) consumed by `scripts/generate_training_config.py`.
- `utils/transform_utils.py`, `utils/mjcf_utils.py` — small geometry / XML helpers.

### Dataset format

Datasets are HDF5 in robomimic format. Top-level `data/` group contains `demo_N` subgroups with `states` (flattened MuJoCo sim state per step), `actions`, and `obs/{camera}_image` keys. The episode's MuJoCo XML is stored as `attrs["model_file"]`, optional language/metadata as `attrs["ep_meta"]`. Environment kwargs are JSON-encoded in `data.attrs["env_args"]`. `playback_datasets.py:reset_to` is the canonical example of reloading both states and the stored XML.

### Training config generation flow

`scripts/generate_training_config.py` builds a list of `dict` settings (per-task: dataset paths, image keys, low-dim keys, horizon). For each, it instantiates a robomimic `ConfigGenerator` against `robomimic/exps/templates/bc.json`, applies `set_learning_settings_for_bc_rnn` (from `dexmimicgen/utils/config_utils.py`), and writes one JSON config per task to `--config_dir`. The output `--output_dir` is baked into the generated configs as the training run output path. Tasks are grouped into `panda_settings` (parallel gripper / dex hand) and `humanoid_settings` blocks inside `make_generators`.
