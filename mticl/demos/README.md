## Expert Demos

This directory contains expert demonstrations used for 3 major ICL and MT-ICL experiments.
1. Single-task ICL for Position / Velocity Constraints
    * See `test_cpo.py` for details
    * `aug_slope_0.5_demos.npz`
    * `aug_speed_0.75_demos.npz`
2. Noisy ablation of single-task ICL for Position / Velocity Constraints
    * See `gen_noisy_demos.py` for details
    * `noisy_aug_slope_0.5_demos.npz`
    * `noisy_aug_speed_0.75_demos.npz`
3. Single / Multi-task ICL for Maze Constraint
    * See `gen_ant_demos.py` for details
    * `maze_goal_{i}_demos.npz`
    * `expert_trajs_{i}.png` (visualized expert trajectories)

## Demo Structure
Each set of demonstrations is saved as a zipped archive of numpy arrays with the same set of keys:
* `trajs`: trajectories
* `acts`: actions 
* `rews`: rewards 
* `constraint_input`: minimal constraint input (e.g. `[x, y]` for position constraint)
* `constraint`: raw value of the ground-truth constraint (see how `info['constraint']` is computed in `envs.py`)