import math
import os
import os.path as osp
from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import seaborn as sns
import torch
from torch.optim import Adam
from tqdm import tqdm
from utils import ICLConfig, MazePlanner, setup_plot_settings, to_torch
from utils.constraints import MazeConstraint


@dataclass
class ExpConfig:
    exp_name: str = "exp"
    maze_task: int = -1
    icl_config: ICLConfig = ICLConfig()
    baseline: bool = False
    epochs: int = 5


class MazeConstraintLearner:
    def __init__(
        self, args: ICLConfig, maze_task: int, exp_name: str, baseline: bool = False
    ):
        self.args = args
        self.maze_task = maze_task
        self.exp_name = exp_name
        self.constraint = MazeConstraint()
        self.learner_buffer = None

        if self.maze_task in range(10):
            self.demos = np.load(
                f"demos/maze_goal_{self.maze_task}_demos.npz",
                allow_pickle=True,
            )["constraint_input"][:, :2]
        elif self.maze_task == -1:
            self.demos = np.concatenate(
                [
                    np.load(f"demos/maze_goal_{i}_demos.npz", allow_pickle=True,)[
                        "constraint_input"
                    ][:, :2]
                    for i in range(10)
                ],
                axis=0,
            )
        else:
            raise NotImplementedError(f"Invalid maze task: {self.maze_task}")

    def sample_batch(self):
        expert_indices = np.random.choice(
            self.demos.shape[0], self.args.constraint_batch_size, replace=False
        )
        learner_indices = np.random.choice(
            self.learner_buffer.shape[0], self.args.constraint_batch_size, replace=False
        )

        expert_batch = to_torch(self.demos[expert_indices])
        learner_batch = to_torch(self.learner_buffer[learner_indices])

        assert expert_batch.shape == learner_batch.shape
        return expert_batch, learner_batch

    def collect_task(self, task, outer_epoch):
        self.constraint.eval()
        constraint_grid = (
            np.zeros((10, 10)) if outer_epoch == 0 else self.discretize_constraint()
        )
        task_trajs, task_reward, task_constraint = [], [], []
        for start_row in [0, 9]:
            start, goal = (start_row, 0), (task, 9)
            planner = MazePlanner(start, goal, constraint_grid)
            for _ in range(10):
                _, _, ci, _, rewards = planner.gen_valid_demo()
                task_trajs.extend(ci)
                task_reward.append(sum(rewards))
                task_constraint.append(
                    sum([planner.compute_constraint(pos) for pos in ci])
                )
        return (
            task_trajs,
            np.array(task_reward).mean(),
            np.array(task_constraint).mean(),
        )

    def collect_trajs(self, outer_epoch):
        learner_trajs = []
        reward, constraint = 0, 0

        # Collect learner trajs
        if self.maze_task == -1:
            for i in range(10):
                trajs, task_rewards, task_constraints = self.collect_task(
                    i, outer_epoch
                )
                learner_trajs.extend(trajs)
                reward += task_rewards
                constraint += task_constraints
            reward /= 10
            constraint /= 10
        else:
            learner_trajs, task_rewards, task_constraints = self.collect_task(
                self.maze_task, outer_epoch
            )
            reward += task_rewards
            constraint += task_constraints

        learner_trajs = np.array(learner_trajs)[:, :2]
        # Update buffer
        self.learner_buffer = (
            np.concatenate([self.learner_buffer, learner_trajs], axis=0)
            if self.learner_buffer is not None
            else learner_trajs
        )
        print(f"Learner buffer size: {self.learner_buffer.shape[0]}")
        return reward, constraint

    def update_constraint(self):
        self.constraint.train()
        self.c_opt = Adam(self.constraint.parameters(), lr=self.args.constraint_lr)
        for _ in (pbar := tqdm(range(self.args.constraint_steps))):
            self.c_opt.zero_grad()
            expert_batch, learner_batch = self.sample_batch()

            c_learner = self.constraint.raw_forward(learner_batch.float())
            c_expert = self.constraint.raw_forward(expert_batch.float())
            c_output = torch.concat([c_expert, c_learner])
            c_labels = torch.concat(
                [-1 * torch.ones(c_expert.shape), torch.ones(c_learner.shape)]
            )
            c_loss = torch.mean((c_output - c_labels) ** 2)

            c_loss.backward()
            pbar.set_description(f"Constraint Loss {c_loss.item()}")
            self.c_opt.step()

        self.constraint.eval()

    def discretize_constraint(self) -> np.ndarray:
        constraint_grid = np.zeros((10, 10))
        for x, y in product(range(-18, 20, 4), range(-18, 20, 4)):
            points = (x, y) + np.random.uniform(
                -2.0, 2.0, size=(self.args.sample_points, 2)
            )
            with torch.no_grad():
                constraint_points = (
                    self.constraint(to_torch(points)).detach().cpu().numpy()
                )
            i, j = math.floor(max(20 + x, 0) / 4), math.floor(max(20 - y, 0) / 4)
            constraint_grid[j][i] = constraint_points.mean()
        return constraint_grid

    def plot_grid(self, grid, title, path, thresh=False, annot=True, fmt=""):
        ax = sns.heatmap(
            grid > 0.5 if thresh else grid,
            annot=annot,
            fmt="" if annot is not True else ".2g",
            linewidth=0.25,
            cmap="Blues",
            cbar=False,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        plt.title(title, size=17.0)
        plt.savefig(path)
        plt.clf()

    def visualize_constraint(self, epoch):
        self.constraint.eval()
        constraint_grid = self.discretize_constraint()
        np.save(
            osp.join(self.args.log_path, f"constraint_{epoch}.npy"), constraint_grid
        )
        self.plot_grid(
            constraint_grid,
            ("Multi" if self.maze_task == -1 else "Single") + "-Task ICL Constraint",
            osp.join(self.args.log_path, f"{self.exp_name}_constraint_{epoch}.pdf"),
            True,
        )
        self.plot_grid(
            constraint_grid,
            f"Single-Task ICL {self.exp_name}: Epoch: {epoch}",
            osp.join(self.args.log_path, f"{self.exp_name}_raw_constraint_{epoch}.png"),
            False,
        )


@pyrallis.wrap()
def render_constraint(args: ExpConfig):
    start, goal = (9, 0), (args.maze_task, 9)
    constraint_grid = np.load(osp.join(args.icl_config.log_path, "constraint_4.npy"))
    planner = MazePlanner(start, goal, constraint_grid, render_mode="rgb_array_list")
    traj, acts, ci, dirs, rewards = planner.gen_valid_demo()


@pyrallis.wrap()
def train(args: ExpConfig):
    setup_plot_settings()
    os.makedirs(args.icl_config.log_path, exist_ok=True)

    cl = MazeConstraintLearner(
        args.icl_config, args.maze_task, args.exp_name, args.baseline
    )
    cl.visualize_constraint(-1)

    stats = {
        "rewards": [],
        "constraints": [],
    }
    for outer_epoch in range(args.epochs):
        # Collect Trajs
        reward, constraint = cl.collect_trajs(outer_epoch)
        stats["rewards"].append(reward)
        stats["constraints"].append(constraint)

        # Constraint Update
        cl.update_constraint()

        # Visualize Constraint
        cl.visualize_constraint(outer_epoch)
        torch.save(
            cl.constraint.state_dict(),
            osp.join(args.icl_config.log_path, f"constraint_{outer_epoch}.pt"),
        )

    np.savez(
        osp.join(args.icl_config.log_path, f"{args.exp_name}.npz"),
        **stats,
    )


if __name__ == "__main__":
    train()
    # render_constraint()
