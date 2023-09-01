#!/bin/bash

# Single-task Position Baseline
python script/train_baseline.py --task AntBulletEnv-v0 --constraint_type Position --suffix baseline_test --seed 0 --baseline True
python script/train_baseline.py --task AntBulletEnv-v0 --constraint_type Position --suffix baseline_test --seed 1 --baseline True
python script/train_baseline.py --task AntBulletEnv-v0 --constraint_type Position --suffix baseline_test --seed 2 --baseline True
python script/gen_table.py --constraint_type Position

# Single-task Velocity Baseline
python script/train_baseline.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix baseline_test --seed 0 --baseline True
python script/train_baseline.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix baseline_test --seed 1 --baseline True
python script/train_baseline.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix baseline_test --seed 2 --baseline True
python script/gen_table.py --constraint_type Velocity
