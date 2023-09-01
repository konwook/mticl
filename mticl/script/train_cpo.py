import pyrallis
from saferl.trainer import OnpolicyTrainer
from utils import (
    ConstraintLearner,
    CPOConfig,
    setup_collector,
    setup_env,
    setup_policy,
    setup_seed,
)


@pyrallis.wrap()
def train(args: CPOConfig):
    setup_seed(args.seed)

    # ------ Policy + Constraint ------
    policy = setup_policy(args, setup_env(args))
    cl = ConstraintLearner(args)
    policy.update_constraint(cl.constraint)
    train_collector, test_collector = setup_collector(args, policy, cl.constraint)

    # ------ Train ------
    def stop_fn(reward, cost):
        return reward > args.reward_threshold and cost < args.cost_limit

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


if __name__ == "__main__":
    train()
