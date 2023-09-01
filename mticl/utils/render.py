import numpy as np
import pybullet


def visualize_region():
    constraint_color = [1, 0, 0]
    pybullet.addUserDebugLine(
        [0, 0, 0], [50, 0, 0], [0, 0, 0], lineWidth=25, lifeTime=0
    )
    pybullet.addUserDebugLine(
        [0, 0, 0], [0, 50, 0], [0, 0, 0], lineWidth=25, lifeTime=0
    )
    pybullet.addUserDebugLine(
        [-50, -25, 0], [50, 25, 0], constraint_color, lineWidth=25, lifeTime=0
    )
    for i in range(-50, 51):
        for j in range(10):
            x = i + j / 10
            pybullet.addUserDebugLine(
                [x, x / 2, 0],
                [x, x / 2 - 50, 0],
                constraint_color,
                lineWidth=25,
                lifeTime=0,
            )


def visualize(env, model, constraint, num_trajs, pybullet_env=False):
    rewards = []
    avg_constraints, avg_costs = [], []

    for _ in range(num_trajs):
        obs, _ = env.reset()
        if pybullet_env:
            visualize_region()

        terminated, truncated = False, False
        total_reward = 0
        constraints, costs = [], []

        while not terminated and not truncated:
            action = model(obs[None, :])[0][0].detach().numpy().flatten()
            obs, reward, terminated, truncated, info = env.step(action)

            ci = np.concatenate(
                [obs[:-1], info["constraint_input"].reshape(-1)], axis=0
            )
            cost = constraint.eval_trajs(ci[None, :])
            obs[-1] = constraint.eval_trajs(ci[None, :], act=False)

            total_reward += reward
            constraints.append(info["constraint"])
            if "cost" in info:
                costs.append(info["cost"])
            else:
                costs.append(cost)

        print("Trajectory Constraints: ", np.sum(np.array(constraints)))
        print("Trajectory Costs: ", np.sum(np.array(costs)))
        print("Trajectory Reward: ", total_reward)

        rewards.append(total_reward)
        avg_constraints.append(np.average(constraints))
        avg_costs.append(np.average(costs))

    print("Avg Reward:", np.mean(rewards))
    print("Avg Constraints:", np.average(avg_constraints))
    print("Avg Costs:", np.average(avg_costs))
