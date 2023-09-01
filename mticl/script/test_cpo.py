import os.path as osp

import numpy as np
import pyrallis
from saferl.data import SRLCollector
from tianshou.data import VectorReplayBuffer
from utils import CPOConfig, restore_policy, setup_env, visualize


def collect_demos(policy, cl, args, total_demos, demo_len):
    test_collector = SRLCollector(
        policy,
        setup_env(args),
        VectorReplayBuffer(args.buffer_size, 1),
        constraint=cl.constraint,
    )
    result = test_collector.collect(n_episode=total_demos)
    rews, lens = result["rews"], result["lens"]
    print(f"Final eval reward: {rews.mean()}, length: {lens.mean()}")

    indices = test_collector.buffer.sample_indices(0)
    data, new_shape = {}, (total_demos, demo_len, -1)
    for data_key, key in [("trajs", "obs"), ("acts", "act"), ("rews", "rew")]:
        data[data_key] = test_collector.buffer.get(indices, key=key).reshape(new_shape)
    info = test_collector.buffer.get(indices, key="info")
    for key in ["constraint_input", "constraint"]:
        data[key] = info[key].reshape(new_shape)
    return data


def generate_demos(policy, cl, args, demo_len=1000, factor=5):
    total_demos = args.num_expert_trajs * factor
    data = collect_demos(policy, cl, args, total_demos, demo_len)

    data_keys = ["trajs", "acts", "rews", "constraint_input", "constraint"]
    good_demos = {k: [] for k in data_keys}

    def append_demo(i):
        for k in data_keys:
            good_demos[k].append(data[k][i])

    for i in range(total_demos):
        avg_constraint = data["constraint"][i, :, -1].mean()
        match args.constraint_type:
            case "Velocity":
                delta, vel_tol = avg_constraint - args.constraint_limit, 0.05
                if delta < 0 and delta > -vel_tol:
                    append_demo(i)
            case "Position":
                pos_tol = 0.1
                if avg_constraint < 0 and avg_constraint > -pos_tol:
                    append_demo(i)
            case _:
                raise ValueError(f"Invalid constraint type: {args.constraint_type}")

        print(
            f"Traj {i}: constraint value: {avg_constraint}, "
            f"num good demos: {len(good_demos['trajs'])}"
        )
        if len(good_demos["trajs"]) == args.num_expert_trajs:
            break

    assert len(good_demos["trajs"]) == args.num_expert_trajs
    for k in data_keys:
        data[k] = np.concatenate(good_demos[k], axis=0)
        print(f"Final shape of {k}: {data[k].shape}")

    np.savez(
        osp.join("demos/", f"{args.suffix}_demos"),
        **data,
    )


@pyrallis.wrap()
def evaluate(args: CPOConfig):
    policy, cl = restore_policy(args)
    generate_demos(policy, cl, args)
    if args.eval_render:
        visualize(
            setup_env(args, render_mode="human"),
            policy.actor,
            cl.constraint,
            args.num_expert_trajs,
            args.task == "AntBulletEnv-v0",
        )


if __name__ == "__main__":
    evaluate()
