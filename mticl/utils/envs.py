import gymnasium
import numpy as np
from gymnasium.spaces import Box


class AntMazeWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = Box(
            low=-np.inf * np.ones(env.observation_space["observation"].shape[0] + 1),
            high=np.inf * np.ones(env.observation_space["observation"].shape[0] + 1),
        )

    def add_info(self, info, obs_dict):
        info["goal"] = np.array(obs_dict["desired_goal"])
        info["pos"] = np.array(obs_dict["achieved_goal"])
        return info

    def reset(self):
        obs_dict, info = self.env.reset()
        return np.concatenate(
            [
                obs_dict["observation"],
                [0],
            ]
        ), self.add_info(info, obs_dict)

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        vel = np.array([info["x_velocity"], info["y_velocity"]])
        constraint = np.linalg.norm(vel) - 1.5
        next_state = np.concatenate(
            [
                obs_dict["observation"],
                [constraint],
            ],
        )
        return next_state, reward, terminated, truncated, self.add_info(info, obs_dict)


class AugVelocityWrapper(gymnasium.Wrapper):
    def __init__(self, env, direction):
        super().__init__(env)
        self.env = env
        self.goal = direction
        self.observation_space = Box(
            low=-np.inf * np.ones(env.observation_space.shape[0] + 1),
            high=np.inf * np.ones(env.observation_space.shape[0] + 1),
        )

    def reset(self):
        obs, info = self.env.reset()
        return np.concatenate([obs, [0]]), info

    def step(self, action):
        observation, old_reward, terminated, truncated, info = self.env.step(action)
        vel = np.array([info["x_velocity"], info["y_velocity"]])
        reward_prog = np.dot(vel, self.goal) / np.linalg.norm(self.goal)
        reward = reward_prog + info["reward_survive"] + info["reward_ctrl"]

        next_state = np.concatenate([observation, [0]], dtype=np.double)

        info["constraint"] = np.linalg.norm(vel)
        info["constraint_input"] = np.array([[info["constraint"]]])
        return next_state, reward, terminated, truncated, info


class PybulletVelocityWrapper(gymnasium.Wrapper):
    def __init__(self, env, delta):
        super().__init__(env)
        self.env = env
        self.pybullet_env = env.env.env.env.env
        self.dt = 0.0165  # self.pybullet_env.scene.dt
        self.delta = delta

        # augment state space to include velocity
        self.observation_space = Box(
            low=-np.inf * np.ones(env.observation_space.shape[0] + 1),
            high=np.inf * np.ones(env.observation_space.shape[0] + 1),
        )

    def reset(self):
        obs, info = self.env.reset()
        return np.concatenate([obs, [0]]), info

    def step(self, action):
        p = np.array(self.pybullet_env.robot.parts["torso"].get_position()[:2])
        next_state, reward, termination, truncation, info = self.env.step(action)
        p_prime = self.pybullet_env.robot.parts["torso"].get_position()[:2]
        vel = np.linalg.norm((p_prime - p) / self.dt)
        next_state = np.concatenate([next_state, [0]], dtype=np.double)
        info["constraint"] = vel  # only for debugging purposes
        info["constraint_input"] = np.array([[vel]])
        return next_state, reward, termination, truncation, info

    def seed(self, seed):
        self.pybullet_env.seed(seed)


class PybulletPositionWrapper(gymnasium.Wrapper):
    def __init__(self, env, m):
        super().__init__(env)
        self.env = env
        self.pybullet_env = env.env.env.env.env
        self.m = m

        # augment state space to include position cost
        self.observation_space = Box(
            low=-np.inf * np.ones(env.observation_space.shape[0] + 1),
            high=np.inf * np.ones(env.observation_space.shape[0] + 1),
        )

    def reset(self):
        obs, info = self.env.reset()
        return np.concatenate([obs, [0]]), info

    def step(self, action):
        next_state, reward, termination, truncation, info = self.env.step(action)
        p = self.pybullet_env.robot.parts["torso"].get_position()[:2]
        next_state = np.concatenate([next_state, [0]], dtype=np.double)
        info["constraint"] = p[0] * self.m - p[1]  # only for debugging purposes
        info["constraint_input"] = p
        return next_state, reward, termination, truncation, info

    def seed(self, seed):
        self.pybullet_env.seed(seed)


class PybulletBaselinePositionWrapper(PybulletPositionWrapper):
    def step(self, action):
        next_state, reward, termination, truncation, info = super().step(action)
        info["cost"] = 0
        return next_state, reward, termination, truncation, info


class PybulletBaselineVelocityWrapper(PybulletVelocityWrapper):
    def step(self, action):
        next_state, reward, termination, truncation, info = super().step(action)
        info["cost"] = 0
        return next_state, reward, termination, truncation, info
