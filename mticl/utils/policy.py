import os.path as osp

import torch
from saferl.policy import PPOLagrangian
from saferl.utils import DummyLogger
from saferl.utils.net.common import ActorCritic
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal
from utils.config import CPOConfig
from utils.constraints import ConstraintLearner
from utils.setup import setup_env, setup_logger


def setup_policy(args, env, no_log=False):
    if args.device == "cpu":
        print("Using cpu with 4 threads.")
        torch.set_num_threads(4)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # model
    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(net, action_shape, max_action=max_action, device=args.device).to(
        args.device
    )
    critic = [
        Critic(
            Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
            device=args.device,
        ).to(args.device)
        for _ in range(2)
    ]
    actor_critic = ActorCritic(actor, critic)

    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    def checkpoint_fn():
        if policy.constraint is None:
            return {"model": policy.state_dict()}
        else:
            return {
                "model": policy.state_dict(),
                "constraint": policy.constraint.state_dict(),
            }

    logger = DummyLogger() if no_log else setup_logger(args, checkpoint_fn)
    policy = PPOLagrangian(
        actor,
        critic,
        optim,
        dist,
        logger=logger,
        cost_limit=args.cost_limit,
        use_lagrangian=args.use_lagrangian,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=env.action_space,
        max_batchsize=20000,
    )

    return policy


def load_policy(policy, cl, ckpt_path, device):
    if osp.exists(ckpt_path):
        print(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(checkpoint["model"], strict=False)
        if cl is not None and cl.constraint is not None:
            cl.constraint.load_state_dict(checkpoint["constraint"], strict=True)
            policy.update_constraint(cl.constraint)
        print("Successfully restore policy.")
    else:
        raise ValueError(f"Failed to restore checkpoint file: {ckpt_path}.")


def restore_policy(args):
    policy = setup_policy(args, setup_env(args), no_log=True)
    cl = ConstraintLearner(args)

    ckpt_path = osp.join(args.log_path, "checkpoint/model_best.pt")
    load_policy(policy, cl, ckpt_path, args.device)
    policy.eval()
    return policy, cl


def load_ant_policies(args: CPOConfig):
    policies = []
    for task in ["plus_y", "minus_y", "minus_x", "plus_x"]:
        policy = setup_policy(args, setup_env(args), no_log=True)
        ckpt_path = osp.join(args.log_path + f"aug_{task}", "checkpoint/model_best.pt")

        load_policy(policy=policy, cl=None, ckpt_path=ckpt_path, device=args.device)
        policy.eval()
        policies.append(policy)

    return policies
