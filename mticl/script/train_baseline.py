import numpy as np
import pyrallis
import torch
import torch.nn as nn
from saferl.trainer import OnpolicyTrainer
from torch.optim import Adam
from tqdm import tqdm
from utils import ICLConfig, setup_collector, setup_env, setup_policy, setup_seed
from utils.constraints import ConstraintLearner, to_torch


class BaselineConstraintLearner(ConstraintLearner):
    def update_constraint(self, baseline_trajs):
        self.constraint.train()

        self.c_opt = Adam(self.constraint.parameters(), lr=self.args.constraint_lr)
        for idx in (pbar := tqdm(range(self.args.constraint_steps))):
            self.c_opt.zero_grad()

            batch_size = self.args.constraint_batch_size
            expert_indices = np.random.choice(self.expert_steps, batch_size)
            expert_data = self.expert_trajs[expert_indices]
            learner_indices = np.random.choice(baseline_trajs.shape[0], batch_size)
            learner_data = baseline_trajs[learner_indices]
            assert expert_data.shape == learner_data.shape

            expert_batch = to_torch(expert_data)
            learner_batch = to_torch(learner_data)

            c_learner = self.constraint.raw_forward(learner_batch.float())
            c_expert = self.constraint.raw_forward(expert_batch.float())
            c_output = torch.concat([c_expert, c_learner])
            c_labels = torch.concat(
                [-1 * torch.ones(c_expert.shape), torch.ones(c_learner.shape)]
            )
            c_loss = torch.mean((c_output - c_labels) ** 2)

            pbar.set_description(f"Constraint Loss {c_loss.item()}")

            c_loss.backward()
            self.c_opt.step()

        self.constraint.eval()


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


def constrain_policy(args: ICLConfig, cl):
    # ------ Policy + Constraint ------
    policy = setup_policy(args, setup_env(args))
    policy.update_constraint(cl.constraint)
    train_collector, test_collector = setup_collector(args, policy, cl.constraint)

    if args.use_bc:
        BC(cl.demos, policy.actor)

    # ------ Train ------
    expert_reward = cl.demos["rews"].mean() * args.traj_len

    def stop_fn(reward, cost):
        return reward > expert_reward and cost <= args.cost_limit

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        test_collector=test_collector,
        cost_limit=args.cost_limit,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.testing_num,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        logger=policy.logger,
        resume_from_log=args.resume,
        save_model_interval=args.save_interval,
    )

    for epoch, epoch_stat, info in trainer:
        print(f"Epoch: {epoch}")

    policy.eval()
    train_collector, test_collector = setup_collector(args, policy, cl.constraint)
    result = test_collector.collect(n_episode=10)
    rews, lens, constraint = result["rews"], result["lens"], result["constraint"]
    with open(f"{args.exp_name}.txt", "a") as f:
        f.write(
            f"Final eval reward: {rews.mean()}, length: {lens.mean()}, "
            f"std: {rews.std()}, raw constraints: {constraint}, "
            f"learned constraint: {cl.constraint.net.data.item()}\n"
        )
    res = {
        "rewards": rews.mean(),
        "raw_constraints": constraint,
        "learned_constraints": cl.constraint.net.data.item(),
    }
    np.savez(f"baseline{args.seed}_stats.npz", **res)


@pyrallis.wrap()
def train(args: ICLConfig):
    setup_seed(args.seed)

    # ----- Policy + Constraint -----
    policy = setup_policy(args, setup_env(args))
    train_collector, test_collector = setup_collector(args, policy)
    cl = BaselineConstraintLearner(args)
    expert_reward = cl.demos["rews"].mean() * args.traj_len

    # ------ Train Policy with BC + normal RL ------
    if args.use_bc:
        BC(cl.demos, policy.actor)

    def stop_fn(reward, cost):
        return reward > args.reward_threshold and cost <= args.cost_limit

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        test_collector=test_collector,
        cost_limit=args.cost_limit,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.testing_num,
        episode_per_collect=args.episode_per_collect,
        baseline=args.baseline,
        stop_fn=stop_fn,
        logger=policy.logger,
        resume_from_log=args.resume,
        save_model_interval=args.save_interval,
        expert_reward=expert_reward,
    )

    for epoch, epoch_stat, info in trainer:
        print(f"Epoch: {epoch}: {np.concatenate(trainer.baseline_trajs, axis=0).shape}")

    # --------- Constraint Update ---------
    last_dim = trainer.baseline_trajs[-1].shape[-1]
    baseline_trajs = np.concatenate(trainer.baseline_trajs, axis=0).reshape(
        -1, last_dim
    )
    with open(f"baseline{args.seed}.npy", "wb") as f:
        np.save(f, baseline_trajs)
    cl.update_constraint(baseline_trajs)

    # --------- CRL wtih New Constraint ---------
    args.use_lagrangian = True
    args.baseline = False
    args.epoch = 50 if args.constraint_type == "Position" else 10  # same as ICL
    constrain_policy(args, cl)


if __name__ == "__main__":
    train()
