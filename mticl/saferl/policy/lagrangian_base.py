from typing import Any, Dict, List, Optional, Type, Union

import gym
import numpy as np
import torch
from saferl.policy import BasePolicy
from saferl.utils import BaseLogger
from saferl.utils.optim_util import LagrangianOptimizer
from tianshou.utils import MultipleLRSchedulers
from torch import nn


class LagrangianPolicy(BasePolicy):
    """Implementation of Lagrangian-based method.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~saferl.policy.BasePolicy`. (s -> logits)
    :param List[torch.nn.Module] critics: a list of the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param LagrangianOptimizer lagrangian_optim: the optimizer for lagrangian.
    :param dist_fn: distribution class for computing the action.
        :type dist_fn: Type[torch.distributions.Distribution]
    :param bool rescaling: whether use the rescaling trick for Lagrangian multiplier,
        see Alg. 1 in http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close to 1.
        Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the
        model; should be as large as possible within the memory constraint.
        Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~saferl.policy.BasePolicy` or the original
        :class:`~tianshou.policy.BasePolicy` for more detailed explanation.
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger,
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: List = [0.05, 0.005, 0],
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Based policy common arguments
        discount_factor: float = 0.99,
        max_batchsize: int = 256,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        # ICL args
        constraint=None,
        # Optional
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[
            Union[torch.optim.lr_scheduler.LambdaLR, MultipleLRSchedulers]
        ] = None,
    ) -> None:
        super().__init__(
            actor,
            critics,
            dist_fn,
            logger,
            discount_factor,
            max_batchsize,
            reward_normalization,
            deterministic_eval,
            action_scaling,
            action_bound_method,
            observation_space,
            action_space,
            lr_scheduler,
            constraint,
        )
        self.rescaling = rescaling
        self.use_lagrangian = use_lagrangian
        self.cost_limit = (
            [cost_limit] * (self.critics_num - 1)
            if np.isscalar(cost_limit)
            else cost_limit
        )
        # suppose there are M constraints, then critics_num = M + 1
        if self.use_lagrangian:
            assert len(self.cost_limit) == (
                self.critics_num - 1
            ), "cost_limit must has equal len of critics_num"
            self.lag_optims = [
                LagrangianOptimizer(lagrangian_pid) for _ in range(self.critics_num - 1)
            ]
        else:
            self.lag_optims = []

    def update_expert_cost(self, expert_cost):
        for lag_opt in self.lag_optims:
            lag_opt.update_expert_cost(expert_cost)

    def reset_errors(self):
        for lag_opt in self.lag_optims:
            lag_opt.reset_errors()

    def pre_update_fn(self, stats_train: Dict, **kwarg) -> Any:
        cost_values = stats_train["cost"]
        self.update_lagrangian(cost_values)

    def update_cost_limit(self, cost_limit):
        self.cost_limit = (
            [cost_limit] * (self.critics_num - 1)
            if np.isscalar(cost_limit)
            else cost_limit
        )

    def update_lagrangian(self, cost_values: Union[List, float]) -> Any:
        """Update the Lagrangian multiplier before updating the policy."""
        if np.isscalar(cost_values):
            cost_values = [cost_values]
        for i, lag_optim in enumerate(self.lag_optims):
            lag_optim.step(cost_values[i], self.cost_limit[i])

    def get_extra_state(self):
        """Save the lagrangian optimizer's parameters.
        This function is called when call the policy.state_dict(),
        see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state
        """
        if len(self.lag_optims):
            return [optim.state_dict() for optim in self.lag_optims]
        else:
            return None

    def set_extra_state(self, state):
        """Load the lagrangian optimizer's parameters. 
        This function is called from load_state_dict() \
        to handle any extra state found within the state_dict.
        """
        if "_extra_state" in state:
            lag_optim_cfg = state["_extra_state"]
            if lag_optim_cfg and self.lag_optims:
                for i, state_dict in enumerate(lag_optim_cfg):
                    self.lag_optims[i].load_state_dict(state_dict)

    def safety_loss(self, values: List):
        """Compute the safety loss based on Lagrangian and return the scaling factor"""
        # get a list of lagrangian multiplier
        lags = [optim.get_lag() for optim in self.lag_optims]
        # Alg. 1 of http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
        rescaling = 1.0 / (np.sum(lags) + 1) if self.rescaling else 1
        assert len(values) == len(lags), "lags and values length must be equal"
        stats = {"loss/rescaling": rescaling}
        loss_safety_total = 0.0
        for i, (value, lagrangian) in enumerate(zip(values, lags)):
            loss = torch.mean(value * lagrangian)
            loss_safety_total += loss
            suffix = "" if i == 0 else "_" + str(i)
            stats["loss/lagrangian" + suffix] = lagrangian
            stats["loss/actor_safety" + suffix] = loss.item()
        return loss_safety_total, stats
