import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
from rliable import library as rly
from rliable import metrics
from utils import ICLConfig, setup_plot, setup_plot_settings


def plot(stats):
    setup_plot()
    plt.plot(stats["mean"], label=stats["learner_label"], color="#F79646")
    plt.fill_between(
        np.arange(len(stats["mean"])),
        stats["mean"] - stats["ci"],
        stats["mean"] + stats["ci"],
        color="#F79646",
        alpha=0.1,
    )
    plt.plot(
        stats["expert_mean"] * np.ones_like(stats["mean"]),
        label=stats["expert_label"],
        color="#008000",
        linestyle="--",
    )
    plt.xlabel("$\\texttt{ICL}$ Iteration")
    plt.ylabel(stats["y_label"])
    if stats["key"] == "constraint" and stats["constraint_type"] == "Velocity":
        ax = plt.gca()
        ax.set_ylim(0.35, 0.8)
    if (
        stats["key"] == "constraint"
        and stats["constraint_type"] == "Velocity"
        and stats["suffix"] == "anneal_aug_speed_0.75"
    ):
        ax = plt.gca()
        ax.set_ylim(0.5, 0.8)

    plt.legend()
    plt.title(f"{stats['constraint_type']} Constraint")
    plt.savefig(
        f"plots/icl/{stats['constraint_type']}_{stats['key']}_{stats['suffix']}.pdf",
        bbox_inches="tight",
    )


def iqm(x):
    return np.array([metrics.aggregate_iqm(x[:, :, i]) for i in range(x.shape[-1])])


def extract_stats(args: ICLConfig):
    stats = {
        "constraint_type": args.constraint_type,
        "expert_constraint": args.constraint_limit,
        "suffix": args.suffix,
    }
    learner_rewards, learner_violations, learner_constraints = [], [], []

    match args.constraint_type:
        case "Velocity" | "Position":
            limit = (
                args.constraint_limit * args.traj_len
                if args.constraint_type == "Velocity"
                else 0
            )

            # Expert stats
            expert_demos = np.load(args.expert_traj_path, allow_pickle=True)

            stats["expert_reward"] = expert_demos["rews"].mean() * args.traj_len
            stats["expert_violation"] = (
                expert_demos["constraint"].mean() * args.traj_len - limit
            )

            # Learner stats
            for seed in range(3):
                args.log_path = args.log_path.replace(f"seed{args.seed}", f"seed{seed}")
                args.seed = seed
                learner_demos = np.load(
                    osp.join(args.log_path, "stats.npz"), allow_pickle=True
                )
                learner_rewards.append(learner_demos["rewards"])
                learner_violations.append(learner_demos["raw_constraints"] - limit)
                learner_constraints.append(learner_demos["learned_constraints"])
        case "Maze":
            # Expert stats
            expert_rewards = np.concatenate(
                [
                    [
                        np.sum(x)
                        for x in np.load(
                            args.expert_traj_path.replace("0", str(i)),
                            allow_pickle=True,
                        )["rewards"]
                    ]
                    for i in range(10)
                ],
                axis=0,
            )
            stats["expert_reward"] = expert_rewards.mean()
            stats["expert_violation"] = 0

            # Learner stats
            gt = np.zeros((10, 10))
            gt[0:6, 2:5] = 1
            gt[4:10, 6:9] = 1

            for i in range(5):
                log_path = args.log_path.replace("mticl_0", f"mticl_{i}")
                data = np.load(osp.join(log_path, f"mticl_{i}.npz"))
                learner_rewards.append(data["rewards"])
                learner_violations.append(data["constraints"])

                ious = []
                for epoch in range(5):
                    mticl_grid = (
                        np.load(osp.join(log_path, f"constraint_{epoch}.npy")) > 0.5
                    )
                    intersection = np.logical_and(mticl_grid, gt)
                    union = np.logical_or(mticl_grid, gt)
                    ious.append(np.sum(intersection) / np.sum(union))
                learner_constraints.append(ious)
        case _:
            raise ValueError(f"Invalid constraint type: {args.constraint_type}")

    stats["learner_reward"] = np.stack(learner_rewards, axis=0)
    stats["learner_violation"] = np.stack(learner_violations, axis=0)
    stats["learner_constraint"] = np.stack(learner_constraints, axis=0)
    return stats


@pyrallis.wrap()
def gen_plots(args: ICLConfig):
    setup_plot_settings()

    stats = extract_stats(args)
    labels = {
        "reward": ["$J(\\cdot, r)$", "$\\pi$", "$\\pi_E$"],
        "violation": ["$J(\\cdot, c^*)$", "$\\pi$", "$\\pi_E$"],
        "constraint": [f"{args.constraint_name}", "$c_i$", "$c^*$"],
    }
    for k, label in labels.items():
        data = stats[f"learner_{k}"]
        mean, _ = rly.get_interval_estimates(
            {"alg": np.expand_dims(data, 1)}, iqm, reps=5000
        )

        stats["key"] = k
        stats["mean"] = mean["alg"]
        stats["ci"] = np.std(data, axis=0) / np.sqrt(len(data))
        stats["expert_mean"] = stats[f"expert_{k}"]

        stats["y_label"] = label[0]
        stats["learner_label"] = label[1]
        stats["expert_label"] = label[2]

        plot(stats)


if __name__ == "__main__":
    gen_plots()
