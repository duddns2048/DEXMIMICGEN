#!/bin/bash

python ./robomimic/robomimic/scripts/train.py \
 --config ./dexmimicgen/datasets/train_configs/bcrnn_action_dict/bc_rnn_image_ds_two_arm_can_sort_humanoid_D0_seed_201.json \
 --wandb_project_name DEXMIMICGEN \
 --name can_sorting_201_base
