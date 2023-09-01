#!/bin/bash

# --------------
# Velocity ICL
# --------------
# Train expert + generate demos
python script/train_cpo.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix aug_speed_0.75 --seed 0
python script/test_cpo.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix aug_speed_0.75 --seed 0

# Single-task ICL
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix anneal_aug_speed_0.75 --seed 0
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix anneal_aug_speed_0.75 --seed 1
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix anneal_aug_speed_0.75 --seed 2

# Generate plots
python script/gen_icl_plots.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix anneal_aug_speed_0.75

# --------------
# Position ICL
# --------------
# Train expert + generate demos
python script/train_cpo.py --task AntBulletEnv-v0 --constraint_type Position --suffix aug_slope_0.5 --seed 0
python script/test_cpo.py --task AntBulletEnv-v0 --constraint_type Position --suffix aug_slope_0.5 --seed 0

# Single-task ICL
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Position --suffix anneal_aug_slope_0.5 --seed 0
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Position --suffix anneal_aug_slope_0.5 --seed 1
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Position --suffix anneal_aug_slope_0.5 --seed 2

# Generate plots
python script/gen_icl_plots.py --task AntBulletEnv-v0 --constraint_type Position --suffix anneal_aug_slope_0.5