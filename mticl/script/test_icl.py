import pyrallis
from saferl.data import SRLCollector
from tianshou.data import VectorReplayBuffer
from utils import ICLConfig, restore_policy, setup_env, visualize


@pyrallis.wrap()
def evaluate(args: ICLConfig):
    policy, cl = restore_policy(args)
    policy.eval()

    if args.eval_render:
        visualize(
            setup_env(args, render_mode="human"),
            policy.actor,
            cl.constraint,
            args.num_expert_trajs,
            args.task == "AntBulletEnv-v0",
        )
    else:
        test_collector = SRLCollector(
            policy,
            setup_env(args),
            VectorReplayBuffer(args.buffer_size, 1),
            constraint=cl.constraint,
        )
        result = test_collector.collect(n_episode=5)
        rews, lens = result["rews"], result["lens"]
        print(f"Final eval reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    evaluate()
