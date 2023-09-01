from typing import Tuple
import os.path as osp

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
from saferl.utils.logger.base_logger import BaseLogger


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir=None, log_txt=True, name=None) -> None:
        super().__init__(log_dir, log_txt, name)
        self.summary_writer = SummaryWriter(osp.join(self.log_dir, "tb"))

    def write(self, step, display=False, display_keys=None):
        self.write_without_reset(step)
        return super().write(step, display, display_keys)

    def write_without_reset(self, step):
        for k in self.logger_keys:
            self.summary_writer.add_scalar(k, self.get_mean(k), step)
        self.summary_writer.flush()

    def restore_data(self) -> Tuple[int, int, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            epoch = ea.scalars.Items("save/epoch")[-1].step
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = ea.scalars.Items("save/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step
