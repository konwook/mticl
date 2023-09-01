# Learning Shared Safety Constraints from Multi-Task Demonstrations
This project provides the implementation of [Learning Shared Safety Constraints from Multi-Task Demonstrations]().
<p align="center">
  <img width="600" src="/assets/icl_ffig.png">
</p>

If you found this repository useful in your research, please consider citing our paper:

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```

## Table of Contents
- [Setup](#Setup)
- [Experiments](#Experiments)
- [Acknowledgments](#Acknowledgments)

The high-level structure of this repository is as follows:
```
├── mticl  # package folder
│   ├── saferl # safe RL library
│   ├── script # training, testing, plotting scripts
│   ├── ├── train/test_cpo.py # CRL
│   ├── ├── train/test_icl.py # Single-task ICL
│   ├── ├── train_baseline.py # Chou et al. baseline
│   ├── ├── planner_icl.py # Multi-task ICL
│   ├── utils # utility functions
|   ├── demos # generated expert demos
│   ├── plots # plots for the paper
│   ├── experiments # experiments for the paper
├── log # experiment results
```
> [!NOTE]
> Please see [here](https://github.com/konwook/mticl/blob/main/mticl/README.md) for a detailed overview of the codebase.

## Setup 
### Installation 
```
conda create -n mticl python=3.10.8
conda activate mticl
pip install -r requirements.txt
export PYTHONPATH=mticl:$PYTHONPATH
```
> [!IMPORTANT]
> All scripts should be run from under ```mticl/```. 

## Experiments

Scripts for replicating results from the paper are provided under the ```experiments/``` directory. 

### AntBulletEnv-v0 (Velocity / Position Constraint)
```
./experiments/pybullet_baseline.sh
./experiments/pybullet_icl.sh
./experiments/pybullet_noisy.sh
```

### AntMaze_UMazeDense-v3 (Maze Constraint)
```
./experiments/ant_maze.sh
```

## Acknowledgements 
The core `saferl/` code was developed from an early version of the [FSRL](https://github.com/liuzuxin/FSRL) repository which provides fast, high-quality implementations of safe RL algorithms. 
> [!WARNING]
> This code is not fully compatible with the latest version of FSRL. See [here](https://github.com/konwook/mticl/blob/main/mticl/README.md) for more information.
