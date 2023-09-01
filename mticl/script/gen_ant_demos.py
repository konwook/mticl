import itertools
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from utils import MazePlanner


def go():
    for goal_row in range(10):
        print(f"Generating demos for goal row {goal_row}")

        task_trajs, task_acts, task_constraint_input, task_dirs = [], [], [], []
        task_rewards = []

        maze = np.zeros((10, 10))
        maze[0:6, 2:5] = 1
        maze[4:10, 6:9] = 1
        for start_row, _ in itertools.product([0, 9], range(10)):
            # gen demo
            start, goal = (start_row, 0), (goal_row, 9)
            planner = MazePlanner(start, goal, maze)
            traj, acts, ci, dirs, rewards = planner.gen_valid_demo()

            # task
            task_trajs.extend(traj)
            task_acts.extend(acts)
            task_constraint_input.extend(ci)
            task_dirs.extend(dirs)
            task_rewards.append(rewards)

        # make sure expert trajs are reasonable
        expert_ci = np.array(task_constraint_input)
        plt.scatter(expert_ci[:, 0], expert_ci[:, 1], c="r")
        plt.savefig(f"demos/expert_trajs_{goal_row}.png")
        plt.clf()

        np.savez(
            osp.join("demos/", f"maze_goal_{goal_row}_demos"),
            trajs=np.array(task_trajs),
            acts=np.array(task_acts),
            constraint_input=np.array(task_constraint_input),
            dirs=np.array(task_dirs),
            rewards=np.array(task_rewards),
        )


if __name__ == "__main__":
    go()
