from typing import Any, Callable, Dict, Optional, Union

import gym
import numpy as np

from tianshou.data import (
    Batch,
    ReplayBuffer,
)
from tianshou.env import BaseVectorEnv
from tianshou.data import Collector

from saferl.policy import BasePolicy


class SRLCollector(Collector):
    """Collector enables the policy to interact with different types of envs with \
    exact number of steps or episodes.

    :param policy: an instance of the :class:`~saferl.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data. Default to None.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer, see issue #42 and :ref:`preprocess_fn`. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified
        with corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

    The "preprocess_fn" is a function called before the data has been added to the
    buffer with batch format. It will receive only "obs" and "env_id" when the
    collector resets the environment, and will receive the keys "obs_next", "rew",
    "terminated", "truncated, "info", "policy" and "env_id" in a normal env step.
    Alternatively, it may also accept the keys "obs_next", "rew", "done", "info",
    "policy" and "env_id".
    It returns either a dict or a :class:`~tianshou.data.Batch` with the modified
    keys and values. Examples are in "test/base/test_collector.py".

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.

    .. note::

        In past versions of Tianshou, the replay buffer that was passed to `__init__`
        was automatically reset. This is not done in the current implementation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
        constraint=None,
    ) -> None:
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)
        # a trick to track the episodic cost
        self._preprocess_fn = preprocess_fn
        self.preprocess_fn = self.compute_episodic_cost
        self.total_constraints = 0
        self.total_costs = 0
        self.constraint = constraint

    def update_constraint(self, constraint):
        self.constraint = constraint

    def compute_episodic_cost(self, info, **kwargs):
        """Called after each env step, before adding data to the buffer."""
        if "constraint" in info:
            self.total_constraints += np.sum(info["constraint"])
            if self.constraint is not None:
                batch_size = info["constraint"].shape[0]
                ci = np.concatenate(
                    [
                        kwargs["obs_next"][:, :-1],
                        info.constraint_input.reshape(batch_size, -1),
                    ],
                    axis=1,
                )
                cost = self.constraint.eval_trajs(ci)
                self.total_costs += np.sum(cost)
                if "no_aug" not in info:
                    kwargs["obs_next"][:, -1] = self.constraint.eval_trajs(
                        ci, act=False
                    )
            else:
                # No constraint to learn, so cost is 0
                kwargs["cost"] = 0
                self.total_costs = 0

        if self._preprocess_fn:
            return self._preprocess_fn(**kwargs)
        else:
            return Batch()

    def collect(
        self,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False.
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            We recommend to use n_episode to collect data rather than n_step because
            it can facilitate the episodic cost computation.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
            * ``rew`` mean of episodic rewards.
            * ``total_cost`` cumulative costs in this collect.
            * ``cost`` mean of episodic costs.
            * ``len`` mean of episodic lengths.
            * ``rew_std`` standard error of episodic rewards.
            * ``len_std`` standard error of episodic lengths.
        """
        stats = super().collect(
            None, n_episode, random, render, no_grad, gym_reset_kwargs
        )
        stats["total_cost"] = self.total_costs
        stats["constraint"] = self.total_constraints / stats["n/ep"]
        stats["cost"] = self.total_costs / stats["n/ep"]
        self.total_constraints = 0
        self.total_costs = 0
        print(f"Cost: {stats['cost']}")
        print(f"Average Traj. Constraint: {stats['constraint'] / 1000.0}")
        return stats
