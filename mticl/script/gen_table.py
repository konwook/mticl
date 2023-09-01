import numpy as np
from itertools import product
from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    constraint_type: str


@pyrallis.wrap()
def gen_table(args: Config):
    demo_path = (
        "aug_slope_0.5" if args.constraint_type == "Position" else "aug_speed_0.75"
    )
    demos = np.load("demos/" + demo_path + "_demos.npz", allow_pickle=True)
    expert_reward = demos["rews"].mean() * 1000
    expert_violation = demos["constraint"].mean() * 1000

    print(f"Expert reward: {expert_reward}, expert violation: {expert_violation}")
    base_path = "../learners/AntBulletEnv-v0/"
    if args.constraint_type == "Position":
        base_path += "icl_Position_AntBulletEnv-v0_seed0_anneal_aug_slope_0.5"
    elif args.constraint_type == "Velocity":
        base_path += "icl_Velocity_AntBulletEnv-v0_seed0_anneal_aug_speed_0.75"
    base_path += "/stats.npz"

    last_iter = 10 if args.constraint_type == "Position" else 20
    keys = ["rewards", "raw_constraints", "learned_constraints"]
    iters = {1: {k: 0 for k in keys}, last_iter: {k: 0 for k in keys}}
    for seed in range(3):
        path = base_path.replace("seed0", f"seed{seed}")
        stats = np.load(path, allow_pickle=True)
        for iter, k in product(iters.keys(), keys):
            iters[iter][k] += stats[k][iter - 1]
        print(
            "Iteration 1: ",
            stats["rewards"][0],
            stats["raw_constraints"][0],
            stats["learned_constraints"][0],
        )
        print(
            f"Iteration {last_iter}: ",
            stats["rewards"][last_iter - 1],
            stats["raw_constraints"][last_iter - 1],
            stats["learned_constraints"][last_iter - 1],
        )

    iters["chou"] = {k: 0 for k in keys}
    for seed in range(3):
        path = f"baseline{seed}_stats.npz"
        stats = np.load(path, allow_pickle=True)
        for k in keys:
            iters["chou"][k] += stats[k]

    for iter, k in product(iters.keys(), keys):
        iters[iter][k] /= 3

    gt_constraint = 0.5 if args.constraint_type == "Position" else 0.75
    for iter in iters.keys():
        print(f"Iteration {iter}: ")
        print("|c* - c|", abs(iters[iter]["learned_constraints"] - gt_constraint))
        print("J(pi_E, r) - J(pi, r)", expert_reward - iters[iter]["rewards"])
        print(
            "J(pi_E, c*) - J(pi, c*)", expert_violation - iters[iter]["raw_constraints"]
        )


if __name__ == "__main__":
    gen_table()
