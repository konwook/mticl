import itertools
import math
from enum import IntEnum
from typing import Optional

import av
import gymnasium
import gymnasium_robotics
import numpy as np
from tianshou.data import Batch
from utils.config import CPOConfig
from utils.envs import AntMazeWrapper
from utils.policy import load_ant_policies

ANT_POLICIES = load_ant_policies(CPOConfig())


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    FREEZE = 4


class WaypointGenerator:
    def __init__(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        constraint_grid: np.ndarray,
        baseline: bool = False,
        maze_dim: int = 10,
    ):
        self.start, self.goal = start, goal
        self.maze = np.where(constraint_grid > 0.5, 1.0, 0.0)
        self.maze[start[0], start[1]] = 0
        self.maze[goal[0], goal[1]] = 0

        self.reward = np.zeros((10, 10))
        self.reward[goal[0], goal[1]] = 1
        if baseline:
            for i in range(10):
                for j in range(10):
                    self.reward[i, j] = np.exp(
                        -np.linalg.norm(np.array(self.goal) - np.array((i, j)))
                    )
        self.num_rows = self.num_cols = maze_dim
        self.actions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

    def is_valid_action(self, row: int, col: int, act: Direction):
        if row == 0 and act == Direction.UP:
            return False
        elif row == self.num_rows - 1 and act == Direction.DOWN:
            return False
        elif col == 0 and act == Direction.LEFT:
            return False
        elif col == self.num_cols - 1 and act == Direction.RIGHT:
            return False
        return True

    def Q_value_iteration(self, reward: np.ndarray, horizon: int, gamma: float):
        self.Q = np.stack([np.zeros_like(self.maze) for _ in range(4)], axis=-1)
        for t in range(horizon):
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if self.is_valid_action(r, c, Direction.UP):
                        self.Q[r, c, Direction.UP] = reward[r, c] + gamma * np.max(
                            self.Q[r - 1, c]
                        )
                    if self.is_valid_action(r, c, Direction.DOWN):
                        self.Q[r, c, Direction.DOWN] = reward[r, c] + gamma * np.max(
                            self.Q[r + 1, c]
                        )
                    if self.is_valid_action(r, c, Direction.LEFT):
                        self.Q[r, c, Direction.LEFT] = reward[r, c] + gamma * np.max(
                            self.Q[r, c - 1]
                        )
                    if self.is_valid_action(r, c, Direction.RIGHT):
                        self.Q[r, c, Direction.RIGHT] = reward[r, c] + gamma * np.max(
                            self.Q[r, c + 1]
                        )

    # randomized to deal with symmetries
    def greedy_policy(self, row: int, col: int) -> Direction:
        acts = []
        q_vals = []
        for act in self.actions:
            if self.is_valid_action(row, col, act):
                acts.append(act)
                q_vals.append(self.Q[row, col, act])
        acts = np.array(acts)
        q_vals = np.array(q_vals)
        return acts[np.random.choice(np.where(q_vals == q_vals.max())[0])]

    def train_expert(self, horizon=30, gamma=0.99):
        self.Q_value_iteration(
            self.reward - 100 * self.maze, horizon=horizon, gamma=0.99
        )
        np.max(self.Q, axis=-1)
        self.expert = lambda r, c: self.greedy_policy(r, c)

    def compute_waypoints(self) -> tuple[list[tuple[int, int]], list[Direction]]:
        row, col = self.start
        traj, acts = [], [Direction.FREEZE]  # noop
        while row != self.goal[0] or col != self.goal[1]:
            act = self.expert(row, col)
            traj.append([row, col])
            acts.append(act)
            if act == Direction.UP:
                row -= 1
            elif act == Direction.DOWN:
                row += 1
            elif act == Direction.LEFT:
                col -= 1
            elif act == Direction.RIGHT:
                col += 1
        traj.append([row, col])
        return traj, acts

    def transform_traj(self, waypoints: list[tuple[int, int]]):
        return [(y, self.num_rows - 1 - x) for (x, y) in waypoints]

    def gen_waypoints(self) -> tuple[list[tuple[int, int]], list[Direction]]:
        self.train_expert()
        waypoints, dirs = self.compute_waypoints()
        return self.transform_traj(waypoints), dirs


class MazePlanner:
    def __init__(
        self,
        start,
        goal,
        constraint_grid,
        render_mode: Optional[str] = None,
        thresh: float = 0.5,
    ):
        self.start, self.goal = start, goal
        self.constraint_grid = constraint_grid
        self.render_mode = render_mode
        self.thresh = thresh
        self.policies = ANT_POLICIES
        self.re_init()

    def re_init(self):
        self.env = self.create_maze_env(self.start, self.goal)
        self.solver = WaypointGenerator(self.start, self.goal, self.constraint_grid)
        self.waypoints, self.dirs = self.solver.gen_waypoints()
        print(self.waypoints, len(self.waypoints))

    def create_maze_env(self, start: tuple[int, int], goal: tuple[int, int]):
        self.ground_truth: list[list[int | str]] = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        maze_map: list[list[int | str]] = (
            self.ground_truth
            if (self.render_mode == "human" or self.render_mode == "rgb_array_list")
            else [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        maze_map[start[0] + 1][start[1] + 1] = "r"
        maze_map[goal[0] + 1][goal[1] + 1] = "g"
        env = AntMazeWrapper(
            gymnasium.make(
                "AntMaze_UMazeDense-v3",
                maze_map=maze_map,
                render_mode=self.render_mode,
                max_episode_steps=1500,
            )
        )
        return env

    def cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        row, col = rowcol_pos
        return np.array([-18 + row * 4, -18 + col * 4])

    def compute_constraint(self, pos):
        x_row = math.floor((pos[0] + 20) / 4)
        y_row = math.floor((20 - pos[1]) / 4)
        return self.ground_truth[y_row + 1][x_row + 1]

    def get_next_goal(self, info):
        if len(self.waypoints) == 0:
            return info["goal"], Direction.FREEZE
        else:
            return (
                self.cell_rowcol_to_xy(np.array(self.waypoints[0])),
                self.dirs[0],
            )

    def plan(self, obs: np.ndarray, info):
        next_waypoint, direction = self.get_next_goal(info)
        dist = np.linalg.norm(info["pos"] - next_waypoint)

        # If we're close enough to the current waypoint, focus on the next one
        if dist < self.thresh:
            assert len(self.waypoints) == len(self.dirs)
            if len(self.waypoints) > 0:
                self.waypoints.pop(0)
                self.dirs.pop(0)
            next_waypoint, direction = self.get_next_goal(info)
        else:
            dx, dy = next_waypoint - info["pos"]
            if abs(dx) > abs(dy):
                if dx > 0:
                    direction = Direction.RIGHT
                else:
                    direction = Direction.LEFT
            else:
                if dy > 0:
                    direction = Direction.UP
                else:
                    direction = Direction.DOWN

        return self.controller(obs, direction)

    def controller(
        self, obs: np.ndarray, direction: Direction
    ) -> tuple[np.ndarray, Direction]:
        if direction == Direction.FREEZE:
            return np.zeros(8), Direction.FREEZE
        policy = self.policies[direction]
        raw_action = policy(Batch(obs=np.array([obs]))).act[0].detach().numpy()
        policy_act = policy.map_action(raw_action)
        return policy_act, direction

    def gen_demo(self, lim: int = 1500):
        obs, info = self.env.reset()
        end_goal = info["goal"]
        steps, rewards, traj, acts, ci, dirs = 0, [], [], [], [], []

        dist_to_goal = np.linalg.norm(end_goal - info["pos"])
        while dist_to_goal > 0.5:
            act, direction = self.plan(obs, info)
            obs, reward, terminated, truncated, info = self.env.step(act)
            traj.append(obs)
            acts.append(act)
            dirs.append(direction)
            rewards.append(reward)
            ci.append(np.concatenate([info["pos"], info["goal"]]))
            dist_to_goal = np.linalg.norm(end_goal - info["pos"])
            steps += 1

            if steps >= lim:
                break

        return steps, dist_to_goal, traj, acts, ci, dirs, rewards

    def gen_valid_demo(self, lim: int = 1500):
        valid_traj = False
        while not valid_traj:
            self.re_init()
            steps, dist_to_goal, traj, acts, ci, dirs, rewards = self.gen_demo(lim)
            print(
                f"Current demo took {steps} steps and was"
                f" {dist_to_goal} from goal with reward {sum(rewards)}"
            )
            if not (valid_traj := steps <= lim and dist_to_goal < self.thresh):
                print("RETRYING...")
            else:
                print("FINISHED!")
                self.render()
                return traj, acts, ci, dirs, rewards

    def render(self):
        if self.render_mode == "rgb_array_list":
            frames = self.env.render()

            container = av.open("ant_maze.mp4", mode="w")
            stream = container.add_stream("mpeg4", rate=60)
            stream.width = 480
            stream.height = 480
            stream.pix_fmt = "yuv420p"

            for frame in frames:
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

            container.close()
