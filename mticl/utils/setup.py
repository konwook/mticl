import os
import random

import gym
import gymnasium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils import gym_patches
import pybullet_envs
import seaborn as sns
import torch
from saferl.data import SRLCollector
from saferl.utils import TensorboardLogger
from tianshou.data import VectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from utils.envs import (
    AugVelocityWrapper,
    PybulletPositionWrapper,
    PybulletVelocityWrapper,
    PybulletBaselinePositionWrapper,
    PybulletBaselineVelocityWrapper,
)


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_collector(args, policy, constraint=None):
    training_num = min(args.training_num, args.episode_per_collect)
    train_envs = ShmemVectorEnv([lambda: setup_env(args) for _ in range(training_num)])
    test_envs = ShmemVectorEnv(
        [lambda: setup_env(args) for _ in range(args.testing_num)]
    )

    train_collector = SRLCollector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
        constraint=constraint,
    )
    test_collector = SRLCollector(
        policy,
        test_envs,
        VectorReplayBuffer(args.buffer_size, len(test_envs)),
        constraint=constraint,
    )
    return train_collector, test_collector


def setup_logger(args, checkpoint_fn=None):
    logger = TensorboardLogger(args.log_path, log_txt=True, name=args.exp_name)
    if not args.no_save and checkpoint_fn is not None:
        logger.setup_checkpoint_fn(checkpoint_fn)

    return logger


def setup_env(args, render_mode=None):
    if args.task == "AntBulletEnv-v0":
        env = gym.make(args.task, render_mode=render_mode, apply_api_compatibility=True)
        if args.constraint_type == "Velocity":
            if args.baseline:
                return PybulletBaselineVelocityWrapper(env, args.constraint_limit)
            else:
                return PybulletVelocityWrapper(env, args.constraint_limit)
        elif args.constraint_type == "Position":
            if args.baseline:
                return PybulletBaselinePositionWrapper(env, args.constraint_limit)
            else:
                return PybulletPositionWrapper(env, args.constraint_limit)
    elif args.task == "Ant-v4":
        env = gymnasium.make(args.task, render_mode=render_mode)
        return AugVelocityWrapper(env, args.direction)
    else:
        raise NotImplementedError


def setup_plot_settings():
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rc("font", family="serif", serif=["Palatino"])
    sns.set(font="serif", font_scale=1.4)
    sns.set_style(
        "white",
        {
            "font.family": "serif",
            "font.weight": "normal",
            "font.serif": ["Times", "Palatino", "serif"],
            "axes.facecolor": "white",
            "lines.markeredgewidth": 1,
        },
    )


def setup_plot():
    plt.figure(dpi=100, figsize=(5.0, 3.0))
    ax = plt.subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=15)
    ax.tick_params(direction="in")
