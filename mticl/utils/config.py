import os.path as osp
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CPOConfig:
    # Task and constraint specification
    task: str = field(default="Ant-v4")  # !!!
    constraint_type: str = field(default="Velocity")  # !!!
    constraint_limit: float = field(default=1.5)  # !!!
    dim: int = field(default=1)  # !!!
    direction: Tuple[int, ...] = field(default=(1, 0))  # !!!
    baseline: bool = field(default=False)

    # Lagrangian params
    use_lagrangian: bool = field(default=True)
    reward_threshold: float = field(default=3000)  # !!!
    cost_limit: float = field(default=20)  # !!!
    lagrangian_pid: Tuple[float, ...] = field(default=(0.05, 0.0005, 0.1))

    # PPO params
    lr: float = field(default=0.0003)
    vf_coef: float = field(default=0.25)
    eps_clip: float = field(default=0.2)
    max_grad_norm: float = field(default=0.02)
    gae_lambda: float = field(default=0.97)
    gamma: float = field(default=0.99)
    batch_size: int = field(default=512)
    hidden_sizes: Tuple[int, ...] = field(default=(128, 128))
    rew_norm: float = field(default=0)
    recompute_adv: bool = field(default=False)
    norm_adv: bool = field(default=True)
    value_clip: bool = field(default=False)
    dual_clip: Optional[float] = field(default=None)

    # Collection params
    epoch: int = field(default=200)  # !!!
    episode_per_collect: int = field(default=16)
    step_per_epoch: int = field(default=20000)
    repeat_per_collect: int = field(default=4)
    buffer_size: int = field(default=100000)
    training_num: int = field(default=16)
    testing_num: int = field(default=4)
    num_expert_trajs: int = field(default=20)

    # General params
    log_dir: str = field(default="experts")
    log_path: str = field(default="")
    method: str = field(default="cpo")
    suffix: str = field(default="")  # !!!
    seed: int = field(default=100)  # !!!
    device: str = field(default="cpu")
    save_interval: int = field(default=4)
    render: bool = field(default=False)
    resume: bool = field(default=False)
    no_save: bool = field(default=False)
    eval_render: bool = field(default=False)

    def __post_init__(self):
        self.log_path = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "../..",
            self.log_dir,
            self.task,
            self.exp_name,
        )
        match (self.constraint_type, self.task):
            case ("Velocity", "AntBulletEnv-v0"):
                self.reward_threshold = 1800
                self.cost_limit = 20
                self.epoch = 50
                self.constraint_limit = 0.75
                self.dim = 1
            case ("Position", "AntBulletEnv-v0"):
                self.reward_threshold = 1800
                self.cost_limit = 100
                self.epoch = 100
                self.constraint_limit = 0.5
                self.dim = 2
            case _:
                print(
                    f"Warning: using default params for "
                    f"{self.constraint_type} on {self.task}"
                )

    @property
    def exp_name(self):
        exp_name = (
            f"{self.method}_{self.constraint_type}_{self.task}"
            f"_seed{self.seed}_{self.suffix}"
        )
        return exp_name


@dataclass
class ICLConfig(CPOConfig):
    # ICL params
    anneal_rate: float = field(default=10)
    constraint_batch_size: int = field(default=4096)
    outer_epochs: int = field(default=20)
    expert_traj_path: str = field(default="trajs/path.npz")
    constraint_lr: float = field(default=0.05)
    constraint_steps: int = field(default=250)
    traj_len: int = field(default=1000)
    constraint_name: str = field(default="Constraint")
    full_state: bool = field(default=False)
    use_noisy: bool = field(default=False)
    use_bc: bool = field(default=True)

    def __post_init__(self):
        self.method = "icl"
        self.log_dir = "learners"
        self.log_path = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "../..",
            self.log_dir,
            self.task,
            self.exp_name,
        )
        match (self.constraint_type, self.task):
            case ("Velocity", "AntBulletEnv-v0"):
                self.reward_threshold = 1800
                self.constraint_limit = 0.75  # for debugging purposes

                if self.full_state:
                    self.cost_limit = 0
                    self.outer_epochs = 100
                    self.epoch = 5
                    self.dim = 29
                else:
                    self.cost_limit = 20
                    self.outer_epochs = 20
                    self.epoch = 10
                    self.dim = 1

                if self.use_noisy:
                    self.expert_traj_path = "demos/noisy_aug_speed_0.75_demos.npz"
                else:
                    self.expert_traj_path = "demos/aug_speed_0.75_demos.npz"
                self.constraint_name = "Velocity"

                if self.baseline:
                    self.use_lagrangian = False
                    self.cost_limit = 0
                    self.epoch = 50
                    self.reward_threshold = 3000
                    self.step_per_epoch = 16000

            case ("Position", "AntBulletEnv-v0"):
                self.reward_threshold = 1800
                self.constraint_limit = 0.5  # for debugging purposes
                if self.full_state:
                    self.cost_limit = 0
                    self.outer_epochs = 50
                    self.epoch = 50
                    self.dim = 30
                else:
                    self.cost_limit = 100
                    self.outer_epochs = 10
                    self.epoch = 50
                    self.dim = 2

                if self.use_noisy:
                    self.expert_traj_path = "demos/noisy_aug_slope_0.5_demos.npz"
                else:
                    self.expert_traj_path = "demos/aug_slope_0.5_demos.npz"

                self.constraint_name = "Slope"

                if self.baseline:
                    self.use_lagrangian = False
                    self.cost_limit = 0
                    self.epoch = 100
                    self.reward_threshold = 3000
                    self.step_per_epoch = 16000

            case ("Maze", "AntMaze_UMazeDense-v3"):
                self.constraint_limit = 1.0
                self.constraint_name = "Maze IoU"
                self.constraint_lr = 3e-4
                self.constraint_steps = 5000
                self.sample_points = 100000
                self.expert_traj_path = "demos/maze_goal_0_demos.npz"
            case _:
                print(
                    f"Warning: using default params for "
                    f"{self.constraint_type} on {self.task}"
                )
