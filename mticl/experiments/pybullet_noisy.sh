#!/bin/bash

# --------------
# Noisy Velocity
# --------------
# Gen demos
python script/gen_noisy_demos.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix aug_speed_0.75 --seed 0

# Run ICL
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix 5e_noisy_speed_0.75 --seed 0 --use_noisy True --use_bc False
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix 5e_noisy_speed_0.75 --seed 1 --use_noisy True --use_bc False
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix 5e_noisy_speed_0.75 --seed 2 --use_noisy True --use_bc False

# Get plots
python script/gen_icl_plots.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix 5e_noisy_speed_0.75 --use_noisy True

# --------------
# Noisy Position
# --------------
# Gen demos
python script/gen_noisy_demos.py --task AntBulletEnv-v0 --constraint_type Position --suffix aug_slope_0.5 --seed 0

# Run ICL
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Position --suffix final_noisy_slope_0.5 --seed 0 --use_noisy True --use_bc False
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Position --suffix final_noisy_slope_0.5 --seed 1 --use_noisy True --use_bc False
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Position --suffix final_noisy_slope_0.5 --seed 2 --use_noisy True --use_bc False

# Get plots
python script/gen_icl_plots.py --task AntBulletEnv-v0 --constraint_type Position --suffix final_noisy_slope_0.5 --use_noisy True
