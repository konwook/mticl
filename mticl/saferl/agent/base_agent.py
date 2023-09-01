from abc import ABC
from typing import Optional

import gym
from saferl.utils import BaseLogger
from saferl.utils.exp_util import seed_all
from tianshou.env import BaseVectorEnv, DummyVectorEnv


class BaseAgent(ABC):
    """The base class for a default agent.


    The base class follows a similar structure as `Tianshou <https://github.com/thu-ml/tianshou>`_.
    All of the policy classes must inherit :class:`~saferl.policy.BasePolicy`.
    """

    def __init__(
        self,
        task: str,
        logger: BaseLogger,
        training_env_num: int = 1,
        testing_env_num: int = 1,
        worker: BaseVectorEnv = DummyVectorEnv,
        seed: Optional[int] = 0,
    ) -> None:
        env = gym.make(task)
        env.action_space.high[0]

        train_envs = worker([lambda: gym.make(task) for _ in range(training_env_num)])
        # test_envs = gym.make(args.task)
        test_envs = worker([lambda: gym.make(task) for _ in range(testing_env_num)])
        seed_all(seed, [train_envs, test_envs])
