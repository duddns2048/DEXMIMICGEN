#!/bin/bash

python ./robomimic/robomimic/scripts/train.py \
 --config ./dexmimicgen/datasets/train_configs/bcrnn_action_dict/bc_rnn_image_ds_two_arm_coffee_humanoid_D0_seed_201.json \
 --wandb_project_name DEXMIMICGEN \
 --name coffee_201_base
