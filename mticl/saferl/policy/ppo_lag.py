from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from saferl.policy import LagrangianPolicy
from saferl.utils.net.common import ActorCritic
from saferl.utils import BaseLogger


class PPOLagrangian(LagrangianPolicy):
    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger,
        # PPO specific arguments
        vf_coef: float = 0.5,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: List = [0.05, 0.005, 0],
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Base policy common arguments
        discount_factor: float = 0.99,
        max_batchsize: int = 256,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        # ICL arg
        constraint=None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            actor,
            critics,
            dist_fn,
            logger,
            use_lagrangian,
            lagrangian_pid,
            cost_limit,
            rescaling,
            discount_factor,
            max_batchsize,
            reward_normalization,
            deterministic_eval,
            action_scaling,
            action_bound_method,
            constraint,
            **kwargs
        )
        self.optim = optim
        self._lambda = gae_lambda
        self._weight_vf = vf_coef
        self._grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        assert (
            dual_clip is None or dual_clip > 1.0
        ), "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        if not self._rew_norm:
            assert (
                not self._value_clip
            ), "value clip is available only when `reward_normalization` is True"
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self.constraint is not None:
            batch_size = batch.obs_next.shape[0]
            ci = np.concatenate(
                [
                    batch.obs_next[:, :-1]
                    if "no_aug" not in batch.info
                    else batch.obs_next,
                    batch.info.constraint_input.reshape(batch_size, -1),
                ],
                axis=1,
            )
            batch.info["cost"] = self.constraint.eval_trajs(ci)

        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        # batch get 3 new keys: values, rets, advs
        batch = self.compute_gae_returns(batch, buffer, indices, self._lambda)
        batch.act = to_torch_as(batch.act, batch.values[..., 0])
        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(
                self._max_batchsize, shuffle=False, merge_last=True
            ):
                old_log_prob.append(self(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        return batch

    def critics_loss(self, minibatch):
        critic_losses = 0
        stats = {}
        for i, critic in enumerate(self.critics):
            value = critic(minibatch.obs).flatten()
            ret = minibatch.rets[..., i]
            if self._value_clip:
                value_target = minibatch.values[..., i]
                v_clip = value_target + (value - value_target).clamp(
                    -self._eps_clip, self._eps_clip
                )
                vf1 = (ret - value).pow(2)
                vf2 = (ret - v_clip).pow(2)
                vf_loss = torch.max(vf1, vf2).mean()
            else:
                vf_loss = (ret - value).pow(2).mean()
            critic_losses += vf_loss

            stats["loss/vf" + str(i)] = vf_loss.item()
        stats["loss/vf_total"] = critic_losses.item()
        return critic_losses, stats

    def policy_loss(self, batch: Batch, dist: Type[torch.distributions.Distribution]):

        ratio = (dist.log_prob(batch.act) - batch.logp_old).exp().float()
        ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
        if self._norm_adv:
            for i in range(self.critics_num):
                adv = batch.advs[..., i]
                mean, std = adv.mean(), adv.std()
                batch.advs[..., i] = (adv - mean) / std  # per-batch norm

        # compute normal ppo loss
        rew_adv = batch.advs[..., 0]
        surr1 = ratio * rew_adv
        surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * rew_adv
        if self._dual_clip:
            clip1 = torch.min(surr1, surr2)
            clip2 = torch.max(clip1, self._dual_clip * rew_adv)
            loss_actor_rew = -torch.where(rew_adv < 0, clip2, clip1).mean()
        else:
            loss_actor_rew = -torch.min(surr1, surr2).mean()

        # compute safety loss
        values = (
            [ratio * batch.advs[..., i] for i in range(1, self.critics_num)]
            if self.use_lagrangian
            else []
        )
        loss_actor_safety, stats_actor = self.safety_loss(values)

        rescaling = stats_actor["loss/rescaling"]
        loss_actor_total = rescaling * (loss_actor_rew + loss_actor_safety)

        stats_actor.update(
            {
                "loss/actor_rew": loss_actor_rew.item(),
                "loss/actor_total": loss_actor_total.item(),
            }
        )
        return loss_actor_total, stats_actor

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self.compute_gae_returns(
                    batch, self._buffer, self._indices, self._lambda
                )
            for minibatch in batch.split(batch_size, merge_last=True):
                # obtain the action distribution
                dist = self.forward(minibatch).dist
                # calculate policy loss
                loss_actor, stats_actor = self.policy_loss(minibatch, dist)
                # calculate loss for critic
                loss_vf, stats_critic = self.critics_loss(minibatch)

                loss = loss_actor + self._weight_vf * loss_vf
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                self.gradient_steps += 1

                ent = dist.entropy().mean()
                self.logger.store(**stats_actor)
                self.logger.store(**stats_critic)
                self.logger.store(total=loss.item(), entropy=ent.item(), tab="loss")

        self.logger.store(gradient_steps=self.gradient_steps, tab="update")
