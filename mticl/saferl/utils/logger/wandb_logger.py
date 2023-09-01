from typing import Tuple
import uuid

import wandb
from saferl.utils.logger.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self, config, project, group, name, log_dir="log", log_txt=True
    ) -> None:
        super().__init__(log_dir, log_txt, name)
        self.wandb_run = (
            wandb.init(
                project=project,
                group=group,
                name=name,
                id=str(uuid.uuid4()),
                resume="allow",
                config=config,  # type: ignore
            )
            if not wandb.run
            else wandb.run
        )
        # wandb.run.save()

    def write(self, step, display=True, display_keys=None):
        self.write_without_reset(step)
        return super().write(step, display, display_keys)

    def write_without_reset(self, step):
        wandb.log(self.stats_mean, step=step)

    def restore_data(self) -> Tuple[int, int, int]:
        pass
