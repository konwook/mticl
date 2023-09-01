import os.path as osp

import numpy as np
import pyrallis
from script.planner_icl import ExpConfig, MazeConstraintLearner
from utils import setup_plot_settings


@pyrallis.wrap()
def gen_plots(args: ExpConfig):
    setup_plot_settings()
    cl = MazeConstraintLearner(args.icl_config, args.maze_task, args.exp_name)
    plot_path = "plots/maze"

    # Single-Task ICL Constraints
    icl_grids = []
    for task in range(10):
        log_path = args.icl_config.log_path.replace("icl_0", f"icl_{task}")
        task_grid = np.load(osp.join(log_path, "constraint_4.npy"))
        icl_grids.append(task_grid)
        cl.plot_grid(
            task_grid,
            "Single-Task ICL Constraint",
            f"{plot_path}/icl_{task}_constraint.pdf",
            thresh=True,
        )

    # Max and Avg of Single-Task ICL Constraints
    max_constraint_grid = np.maximum.reduce(icl_grids)
    cl.plot_grid(
        max_constraint_grid,
        "Max of Single-Task ICL Constraints",
        f"{plot_path}/maxicl_constraint.pdf",
        thresh=True,
    )

    avg_constraint_grid = np.mean(icl_grids, axis=0)
    cl.plot_grid(
        avg_constraint_grid,
        "Average of Single-Task ICL Constraints",
        f"{plot_path}/avgicl_constraint.pdf",
        thresh=True,
    )

    # Multi-Task ICL Constraints
    for i in range(5):
        log_path = args.icl_config.log_path.replace("icl_0", f"mticl_{i}")
        mticl_grid = np.load(osp.join(log_path, "constraint_4.npy"))
        cl.plot_grid(
            mticl_grid,
            "Multi-Task ICL Constraint",
            f"{plot_path}/mticl_{i}_constraint.pdf",
            thresh=True,
        )

    # Single-Task Setup
    goals = np.zeros((10, 10))
    goals[0][0] = goals[9][0] = goals[2][9] = 1
    annot = [["" for _ in range(10)] for _ in range(10)]
    annot[2][9] = "G"
    annot[0][0] = "S"
    annot[9][0] = "S"
    cl.plot_grid(
        goals,
        "Single-Task ICL Setup",
        f"{plot_path}/icl_setup.pdf",
        thresh=True,
        annot=annot,
    )

    # Multi-Task Setup
    goals = np.zeros((10, 10))
    goals[0:6, 2:5] = 1
    goals[4:10, 6:9] = 1
    cl.plot_grid(
        goals,
        "Ground-Truth Constraint",
        f"{plot_path}/ground_truth.pdf",
        thresh=True,
        annot=True,
    )


if __name__ == "__main__":
    gen_plots()
