import os.path as osp

import numpy as np
import pyrallis
import torch.nn as nn
from saferl.trainer import OnpolicyTrainer
from torch.optim import Adam
from tqdm import tqdm
from utils import ICLConfig, setup_collector, setup_env, setup_policy, setup_seed
from utils.constraints import ConstraintLearner, to_torch


def BC(demos, pi, loss_fn=nn.MSELoss(), lr=3e-4, steps=int(2e4), wd=1e-3):
    X, U = demos["trajs"], demos["acts"]
    X[:, -1] = 0
    pi.train()
    optimizer = Adam(pi.parameters(), lr=lr)
    for step in (pbar := tqdm(range(steps))):
        idx = np.random.choice(len(X), 128)

        states = to_torch(X[idx])
        actions = to_torch(U[idx])
        states.requires_grad = True
        actions.requires_grad = True

        optimizer.zero_grad()

        outputs = pi(states.float())[0][0]

        loss = loss_fn(outputs, actions.float())
        loss.backward()

        pbar.set_description(f"Loss {loss.item()}")
        optimizer.step()

    return pi


def test_policy(policy, cl, args, test_collector):
    policy.eval()
    result = test_collector.collect(n_episode=10)
    test_collector.reset()
    rews, lens, constraint = result["rews"], result["lens"], result["constraint"]
    with open(f"{args.exp_name}.txt", "a") as f:
        f.write(
            f"Final eval reward: {rews.mean()}, length: {lens.mean()}, "
            f"std: {rews.std()}, raw constraints: {constraint}\n"
        )
    return {"rewards": rews.mean(), "raw_constraints": constraint}


@pyrallis.wrap()
def train(args: ICLConfig):
    setup_seed(args.seed)

    # ----- Policy + Constraint -----
    policy = setup_policy(args, setup_env(args))
    cl = ConstraintLearner(args)
    cl.set_norm_constraint()
    policy.update_constraint(cl.norm_constraint)
    train_collector, test_collector = setup_collector(args, policy, cl.norm_constraint)

    # ------ Train ------
    if args.use_bc:
        BC(cl.demos, policy.actor)
    rewards, learned_constraints, raw_constraints = [], [], []

    for outer_epoch in range(args.outer_epochs):
        cost_limit = max(args.cost_limit - outer_epoch * args.anneal_rate, 0)

        # --------- Constraint Update ---------
        cl.collect_trajs(test_collector)
        cl.update_constraint()
        if not args.full_state:
            with open(f"{args.exp_name}.txt", "a") as f:
                f.write(
                    f"Outer epoch {outer_epoch}: constraint"
                    f" is {cl.constraint.net.data.item()}"
                )

        # --------- Policy Update ---------
        if args.use_bc:
            BC(cl.demos, policy.actor)
        policy.update_constraint(cl.norm_constraint)
        train_collector.update_constraint(cl.norm_constraint)
        test_collector.update_constraint(cl.norm_constraint)

        expert_reward = cl.demos["rews"].mean() * args.traj_len
        expert_cost = cl.expert_cost()
        policy.update_expert_cost(expert_cost)
        policy.reset_errors()

        def stop_fn(reward, cost):
            return (cost - expert_cost) <= cost_limit and reward > expert_reward

        trainer = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            max_epoch=args.epoch,
            batch_size=args.batch_size,
            test_collector=test_collector,
            cost_limit=cost_limit,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.testing_num,
            episode_per_collect=args.episode_per_collect,
            stop_fn=stop_fn,
            logger=policy.logger,
            resume_from_log=args.resume,
            save_model_interval=args.save_interval,
            expert_reward=expert_reward,
        )

        for epoch, _, _ in trainer:
            print(f"Outer Epoch {outer_epoch}: ICL Epoch: {epoch}")

        # --------- Test Policy ---------
        res = test_policy(policy, cl, args, test_collector)
        rewards.append(res["rewards"])
        raw_constraints.append(res["raw_constraints"])
        if not args.full_state:
            learned_constraints.append(cl.constraint.net.data.item())

    np.savez(
        osp.join(args.log_path, "stats.npz"),
        rewards=np.array(rewards),
        raw_constraints=np.array(raw_constraints),
        learned_constraints=np.array(learned_constraints),
    )


if __name__ == "__main__":
    train()
