#!/bin/bash

# Train cardinal ants + generate demos
python script/train_cpo.py --task Ant-v4 --constraint_type Velocity --constraint_limit 1.5 --dim 1 --suffix plus_y --direction [0,1]
python script/train_cpo.py --task Ant-v4 --constraint_type Velocity --constraint_limit 1.5 --dim 1 --suffix minus_y --direction [0,-1]
python script/train_cpo.py --task Ant-v4 --constraint_type Velocity --constraint_limit 1.5 --dim 1 --suffix minus_x --direction [-1,0]
python script/train_cpo.py --task Ant-v4 --constraint_type Velocity --constraint_limit 1.5 --dim 1 --suffix plus_x --direction [1,0]
python script/gen_ant_demos.py

# Single-task and multi-task ICL
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_0 --exp_name icl_0 --maze_task 0
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_1 --exp_name icl_1 --maze_task 1
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_2 --exp_name icl_2 --maze_task 2
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_3 --exp_name icl_3 --maze_task 3
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_4 --exp_name icl_4 --maze_task 4
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_5 --exp_name icl_5 --maze_task 5
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_6 --exp_name icl_6 --maze_task 6
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_7 --exp_name icl_7 --maze_task 7
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_8 --exp_name icl_8 --maze_task 8
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_9 --exp_name icl_9 --maze_task 9
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix mticl_0 --exp_name mticl_0 --maze_task -1
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix mticl_1 --exp_name mticl_1 --maze_task -1
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix mticl_2 --exp_name mticl_2 --maze_task -1
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix mticl_3 --exp_name mticl_3 --maze_task -1
python script/planner_icl.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix mticl_4 --exp_name mticl_4 --maze_task -1

# Generate plots
python script/gen_icl_plots.py --task AntMaze_UMazeDense-v3 --constraint_type Maze --suffix mticl_0
python script/gen_maze_plots.py --icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.suffix icl_0 --exp_name icl_0 --maze_task 0