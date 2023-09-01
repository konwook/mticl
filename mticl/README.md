# mticl

This document provides an overview of the `mticl` codebase for users interested in trying out the code or extending it for their own research.

## Code Structure

The critical components of the codebase are labeled below and discussed in further detail.
### `saferl`
The core `saferl` code was developed from an early version of the [FSRL](https://github.com/liuzuxin/FSRL) repository which provides fast, high-quality implementations of safe RL algorithms. We develop our ICL algorithms on top of this constrained policy optimization implementation.

> [!NOTE]
> One major change we make is that we augment the policy state with the raw value of the constraint function. For CRL / ICL, this is the constraint violation of the ground-truth / learned constraint respectively. We find that this is an important implementation detail for getting our algorithms to work well.

To support this change, we modify the `saferl` codebase in the following ways:
* `data/collector.py`: extend the preprocessing step to compute the cost using the latest constraint
* `policy/base_policy.py`: augment the policy state and update it with the latest constraint

Additionally, we modify the Lagrangian optimizer in `utils/optim_util.py` to support a cost limit 
dependent on the expert constraint violation.

### `script`
Roughly speaking, we provide three types of scripts:
1. `gen_*_plots/gen_table.py`: generate plots and tables for the paper
2. `gen_*_demos.py`: generate expert demonstrations (see [here](demos/README.md) for more information)
3. `train/test_*.py`: train/test a given policy on a given environment

For the exact usage of the experimental scripts, please see `experiments/` and `utils/config.py`
for hyperparameters.

### `utils`
Most of the infrastructure and utility functions for our experiments live here. 
* `config.py`: the configuration format for our experiments, built with [Pyrallis](https://github.com/eladrich/pyrallis)
* `constraints.py`: infrastructure for representing and learning constraints
* `envs.py`: environment wrappers
* `setup.py`: setup functions for experiments
* `policy.py`: utility functions for creating, loading, and saving policies
* `maze_utils.py`: utility functions for maze environments
* `render.py`: for visualizing environments
* `gym_patches.py`: for patching gym environments to the new 0.26 API

## Customization

This section provides some guidelines on how to extend this codebase for custom experiments with ICL. At a high level, the following steps are required:
* Creating a [custom environment and constraint](#custom-environments-and-constraints)
* Training an [expert](#custom-experts) to satisfy the ground-truth constraint
* Generating [expert demonstrations](#generating-expert-demonstrations)
* [Running ICL](#custom-experiments) using the expert demonstrations to learn a constraint



### Custom Environments and Constraints
Our experiments build on top of environments from the Pybullet and MuJoCo suites. To extend this codebase with a custom environment, we expect the following steps to be roughly sufficient for single-task experiments:
* Create a new environment wrapper in `utils/envs.py`
    * It must support augmented state observations (see `utils/envs.py` for examples)
    * `info` must contain `constraint` and `constraint_input` keys
* Modify `setup_env` in `utils/setup.py` to return the new environment
* Create a new constraint class in `utils/constraints.py`
    * Modify `ConstraintLearner` to support a learnable version and a ground truth version of the new constraint

### Custom Experts
Once the environment is setup and a configuration is approriately defined in `utils/config.py`, an expert can be trained using the `train_cpo.py` script. This will train a policy to satisfy the ground truth constraint using CRL. For example, the velocity expert was trained using the following command:
```
python script/train_cpo.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix aug_speed_0.75 --seed 0
```
See `experiments/pybullet_icl.sh` for more examples. 

### Generating Expert Demonstrations

Once an expert has been trained, demonstrations can be collected by rolling out the expert policy in the environment. We provide several scripts for doing this: 
* `test_cpo.py`: demo collection for single-task ICL (position / velocity constraints)
* `gen_noisy_demos.py`: ablation for collecting noisy demos for single-task ICL
* `gen_ant_demos.py`: demo collection for single / multi-task ICL (maze constraint)

The simplest way to get started is to modify `test_cpo.py` to support the new constraint type. For our velocity expert, we used the following command:
```
python script/test_cpo.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix aug_speed_0.75 --seed 0
```
See [here](demos/README.md) for more information about the expert demonstrations we used in our experiments.

### Custom Experiments
Finally, given expert demonstrations, single-task ICL can be run with the new environment by following a similar procedure to the existing `experiments/pybullet_icl.sh` script. For example, the following command runs single-task ICL for the velocity constraint:
```
python script/train_icl.py --task AntBulletEnv-v0 --constraint_type Velocity --suffix anneal_aug_speed_0.75 --seed 0
```
For multi-task ICL, demonstrations must be generated for all tasks and the constraint learning procedure is slightly different. See `gen_ant_demos.py` and `planner_icl.py` for an example of how to do this on the AntMaze environment with the maze constraint.

## Baseline
We compare our ICL algorithms to a baseline based on prior work from [Chou et al.](https://par.nsf.gov/servlets/purl/10316776) Please refer to Appendix C of the paper for an explanation as to how our algorithm differs from this prior work and how we implement the baseline. Our implementation is provided in `train_baseline.py` and a table comparing our results is generated using `gen_table.py`. See `experiments/pybullet_baseline.sh` for the exact usage of these scripts.


## Integration with FSRL
> [!WARNING]
> Unfortunately, the current version of this code is not fully compatible with the latest version of FSRL. We are working on developing a cleaner integration with FSRL, but for now, we recommend using the `saferl` module provided in this repository.
