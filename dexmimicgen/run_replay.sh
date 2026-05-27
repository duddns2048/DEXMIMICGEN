#!/usr/bin/env bash
# 동작: 데이터 replay (9개 태스크 중 하나 선택)
# 사용: bash run_replay.sh <task>
#   <task>: threading | three_piece_assembly | transport
#           box_cleanup | drawer_cleanup | lift_tray
#           coffee | pouring | can_sort_random
# 인자 없으면 threading 기본값. --render 빼면 mp4로 저장.

# Panda + parallel gripper
# TASK="threading" # deault
# TASK="three_piece_assembly"
# TASK="transport"

# Panda + dexterous hand
# TASK="box_cleanup"
# TASK="drawer_cleanup"
# TASK="lift_tray"

# Humanoid + dexterous hand
# TASK="coffee"
# TASK="pouring"
TASK="can_sort_random"

# TASK="can_sort_random"

python dexmimicgen/scripts/playback_datasets.py --dataset dexmimicgen/datasets/generated/two_arm_${TASK}.hdf5 --n 10 --render \
#  --use-obs
