import os.path as osp

import numpy as np
import pyrallis
from utils import CPOConfig, restore_policy, setup_env
from utils.render import visualize_region


def collect_noisy_demos(policy, constraint, args, total_demos):
    data_keys = ["trajs", "acts", "rews", "constraint_input", "constraint"]
    data = {k: [] for k in data_keys}
    for _ in range(total_demos):
        env = setup_env(args)
        obs, _ = env.reset()
        visualize_region()

        trajs, acts, rews, constraint_input, constraints = [], [], [], [], []
        terminated, truncated = False, False
        while not terminated and not truncated:
            action = policy.actor(obs[None, :])[0][0].detach().numpy().flatten()
            action += np.random.normal(
                0, 0.7 if args.constraint_type == "Velocity" else 0.5, action.shape
            )
            obs, reward, terminated, truncated, info = env.step(action)

            ci = np.concatenate(
                [obs[:-1], info["constraint_input"].reshape(-1)], axis=0
            )
            obs[-1] = constraint.eval_trajs(ci[None, :], act=False)

            trajs.append(obs)
            acts.append(action)
            rews.append(reward)
            constraint_input.append(info["constraint_input"])
            constraints.append([info["constraint"]])

        print("Trajectory Constraints: ", np.sum(np.array(constraints)))
        print("Trajectory Reward: ", np.sum(np.array(rews)))

        for k, v in zip(data_keys, [trajs, acts, rews, constraint_input, constraints]):
            data[k].append(np.array(v))

    return {k: np.array(v) for k, v in data.items()}


def generate_demos(policy, cl, args, demo_len=1000, factor=1):
    total_demos = args.num_expert_trajs * factor
    data = collect_noisy_demos(policy, cl.constraint, args, total_demos)

    data_keys = ["trajs", "acts", "rews", "constraint_input", "constraint"]
    good_demos = {k: [] for k in data_keys}

    def append_demo(i):
        for k in data_keys:
            good_demos[k].append(data[k][i])

    for i in range(total_demos):
        avg_constraint = data["constraint"][i, :, -1].mean()
        match args.constraint_type:
            case "Velocity":
                delta, vel_tol = avg_constraint - args.constraint_limit, 0.25
                if delta < 0 and delta > -vel_tol:
                    append_demo(i)
            case "Position":
                if avg_constraint < 0:
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
        osp.join("demos/", f"noisy_{args.suffix}_demos.npz"),
        **data,
    )


@pyrallis.wrap()
def evaluate(args: CPOConfig):
    policy, cl = restore_policy(args)
    generate_demos(policy, cl, args)


if __name__ == "__main__":
    evaluate()
