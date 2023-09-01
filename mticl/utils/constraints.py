import numpy as np
import torch
from tianshou.data import ReplayBuffer
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


def to_torch(x: np.ndarray):
    return torch.from_numpy(x).type(torch.float32)


class Constraint(nn.Module):
    def __init__(self, in_dim=1):
        super(Constraint, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1),
        )
        self.in_dim = in_dim

    def raw_forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        net_inputs = inputs[:, -self.in_dim :]
        output = self.net(net_inputs)
        return output.view(-1)

    def forward(self, inputs):
        return torch.log(nn.functional.relu(self.raw_forward(inputs)) + 1.0)

    def eval_trajs(self, trajs: np.ndarray, act=True):
        inputs = to_torch(trajs)
        with torch.no_grad():
            output = self.forward(inputs) if act else self.raw_forward(inputs)
        return torch.Tensor.numpy(output, force=True)


class MazeConstraint(Constraint):
    def __init__(self, in_dim=2, dim=128, out_dim=1):
        super(MazeConstraint, self).__init__(in_dim=in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, out_dim),
        )


class ReducedVelocityConstraint(Constraint):
    def __init__(self, in_dim=1):
        super(ReducedVelocityConstraint, self).__init__(in_dim=in_dim)
        self.net = nn.Parameter(torch.randn(1))

    # inputs = (batch_size x [x, y])
    def raw_forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        net_inputs = inputs[:, -self.in_dim :]
        output = net_inputs - self.net
        return output.view(-1)


class ReducedPositionConstraint(Constraint):
    def __init__(self, in_dim=1):
        super(ReducedPositionConstraint, self).__init__(in_dim=in_dim)
        self.net = nn.Parameter(torch.randn(1))

    # inputs = (batch_size x [x, y])
    def raw_forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        net_inputs = inputs[:, -self.in_dim :]
        output = net_inputs[:, 0] * self.net - net_inputs[:, 1]
        return output.view(-1)


class ConstraintLearner:
    def __init__(self, args):
        self.args = args

        match (args.constraint_type, args.method):
            case ("Velocity", "cpo"):
                self.constraint = Constraint(in_dim=args.dim)
                constraint_weight = torch.tensor([[1.0]], dtype=torch.float32)
                constraint_bias = torch.tensor(
                    [-args.constraint_limit], dtype=torch.float32
                )
                state = {
                    "net.0.weight": constraint_weight,
                    "net.0.bias": constraint_bias,
                }
                self.constraint.load_state_dict(state)

            case ("Velocity", "icl"):
                if args.dim == 1:
                    self.constraint = ReducedVelocityConstraint(in_dim=args.dim)
                else:
                    self.constraint = Constraint(in_dim=args.dim)

            case ("Position", "cpo"):
                self.constraint = Constraint(in_dim=2)
                constraint_weight = torch.tensor(
                    [[args.constraint_limit, -1.0]], dtype=torch.float32
                )
                constraint_bias = torch.tensor([0.0], dtype=torch.float32)
                state = {
                    "net.0.weight": constraint_weight,
                    "net.0.bias": constraint_bias,
                }
                self.constraint.load_state_dict(state)

            case ("Position", "icl"):
                if args.dim == 2:
                    self.constraint = ReducedPositionConstraint(in_dim=args.dim)
                else:
                    self.constraint = Constraint(in_dim=args.dim)

            case _:
                raise NotImplementedError

        if args.method == "icl":
            self.demos = np.load(args.expert_traj_path, allow_pickle=True)
            self.expert_steps = self.demos["trajs"].shape[0]
            self.expert_trajs = np.concatenate(
                [
                    self.demos["trajs"].reshape(self.expert_steps, -1)[:, :-1],
                    self.demos["constraint_input"].reshape(self.expert_steps, -1),
                ],
                axis=1,
            )
            self.c_opt = Adam(self.constraint.parameters(), lr=args.constraint_lr)
            self.buffer = ReplayBuffer(size=(args.outer_epochs * self.expert_steps))

    def expert_cost(self):
        expert_costs = self.norm_constraint.eval_trajs(self.expert_trajs)
        return expert_costs.sum() / self.args.num_expert_trajs

    def collect_trajs(self, test_collector):
        test_collector.collect(n_episode=self.args.episode_per_collect)
        learner_sample, _ = test_collector.buffer.sample(self.expert_steps)
        for batch in learner_sample.split(1):
            self.buffer.add(batch)
        test_collector.reset()

    def sample_batch(self):
        batch_size = self.args.constraint_batch_size
        expert_indices = np.random.choice(self.expert_steps, batch_size)
        expert_batch = self.expert_trajs[expert_indices]
        learner_indices = self.buffer.sample_indices(batch_size)

        learner_obs = self.buffer.get(learner_indices, key="obs").reshape(
            batch_size, -1
        )[:, :-1]
        learner_ci = self.buffer.get(
            learner_indices, key="info"
        ).constraint_input.reshape(batch_size, -1)
        learner_batch = np.concatenate([learner_obs, learner_ci], axis=1)

        expert_data = to_torch(expert_batch)
        learner_data = to_torch(learner_batch)
        assert expert_data.shape == learner_data.shape
        return expert_data, learner_data

    def set_norm_constraint(self):
        self.norm_constraint = self.constraint
        if self.args.full_state:
            self.norm_constraint = Constraint(in_dim=self.args.dim)
            with torch.no_grad():
                constraint_weight, constraint_bias = list(self.constraint.parameters())
                l2_norm = torch.linalg.vector_norm(constraint_weight, ord=2)
                assert l2_norm != 0
                state = {
                    "net.0.weight": constraint_weight / l2_norm.item(),
                    "net.0.bias": constraint_bias / l2_norm.item(),
                }
                self.norm_constraint.load_state_dict(state)

    def update_constraint(self):
        self.constraint.train()
        self.c_opt = Adam(self.constraint.parameters(), lr=self.args.constraint_lr)
        for idx in (pbar := tqdm(range(self.args.constraint_steps))):
            self.c_opt.zero_grad()
            expert_batch, learner_batch = self.sample_batch()

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
        self.set_norm_constraint()
