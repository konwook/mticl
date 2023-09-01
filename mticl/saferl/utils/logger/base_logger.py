import atexit
import json
import os
import os.path as osp
from abc import ABC
from collections import defaultdict
from typing import Callable, Optional, Union, Iterable

import numpy as np
import torch
import yaml
from saferl.utils.logger.logger_util import colorize, convert_json, RunningAverage


class DummyLogger(ABC):
    """A logger that does nothing. Used as the placeholder in trainer."""

    def __init__(self) -> None:
        super().__init__()

    def store(self, **kwarg):
        """Store the information"""

    def setup_checkpoint_fn(self, **kwarg):
        """Set up the function to obtain the model checkpoint"""

    def write(self, **kwarg):
        """The LazyLogger writes nothing."""

    def write_without_reset(self, **kwarg):
        """Writing data to somewhere without resetting the stats, for tb and wandb"""

    def save_checkpoint(self, **kwarg):
        """Save the model checkpoints"""

    def save_config(self, **kwarg):
        """Log an experiment configuration"""

    def restore_data(self, **kwarg):
        """Return the metadata from existing log"""

    def get_mean(self, **kwarg):
        """Return the mean value of a key"""
        return np.inf

    @property
    def stats_mean(self):
        """Return the dict of mean values of statistics"""


class BaseLogger(DummyLogger):
    """The base class for any logger which is compatible with trainer.

    Try to overwrite write() method to use your own writer.

    :param int train_interval: the log interval in log_train_data(). Default to 1000.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data(). Default to 1000.
    """

    def __init__(self, log_dir=None, log_txt=True, name=None) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.log_fname = "progress.txt"
        self.name = name
        if log_dir:
            if osp.exists(self.log_dir):
                warning_msg = colorize(
                    "Warning: Log dir %s already exists! Some logs may be overwritten."
                    % self.log_dir,
                    "magenta",
                    True,
                )
                print(warning_msg)
            else:
                os.makedirs(self.log_dir)
            if log_txt:
                self.output_file = open(osp.join(self.log_dir, self.log_fname), "w")
                atexit.register(self.output_file.close)
                print(
                    colorize(
                        "Logging data to %s" % self.output_file.name, "green", True
                    )
                )
        else:
            self.output_file = None
        self.first_row = True
        self.checkpoint_fn = None
        self.reset_data()

    def setup_checkpoint_fn(self, checkpoint_fn: Optional[Callable] = None):
        self.checkpoint_fn = checkpoint_fn

    def reset_data(self):
        self.log_data = defaultdict(RunningAverage)

    def store(self, tab=None, **kwargs):
        for k, v in kwargs.items():
            if tab is not None:
                k = tab + "/" + k
            self.log_data[k].add(np.mean(v))

    def write(self, step, display=False, display_keys=None):
        """Writing data to somewhere"""
        # if self.name:
        #     self.store(name=self.name)
        # save .txt file to the output logger
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.logger_keys) + "\n")
            vals = self.get_mean_list(self.logger_keys)
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
            self.first_row = False
        if display:
            self.display_tabular(display_keys=display_keys)
        self.reset_data()

    def write_without_reset(self, **kwarg):
        """Writing data to somewhere without resetting the stats, for tb and wandb"""

    def save_checkpoint(self, suffix: Optional[Union[int, str]] = None):
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        if self.checkpoint_fn and self.log_dir:
            fpath = osp.join(self.log_dir, "checkpoint")
            os.makedirs(fpath, exist_ok=True)
            suffix = "%d" % suffix if isinstance(suffix, int) else suffix
            suffix = "_" + suffix if suffix is not None else ""
            fname = "model" + suffix + ".pt"
            torch.save(self.checkpoint_fn(), osp.join(fpath, fname))

    def save_config(self, config: dict, verbose=True):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        if self.name is not None:
            config["name"] = self.name
        config_json = convert_json(config)
        if verbose:
            print(colorize("Saving config:\n", color="cyan", bold=True))
            output = json.dumps(
                config_json, separators=(",", ":\t"), indent=4, sort_keys=True
            )
            print(output)
        if self.log_dir:
            with open(osp.join(self.log_dir, "config.yaml"), "w") as out:
                yaml.dump(
                    config, out, default_flow_style=False, indent=4, sort_keys=False
                )

    def restore_data(self):
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """
        pass

    def get_std(self, key: str):
        return self.log_data[key].std

    def get_mean(self, key: str):
        return self.log_data[key].mean

    def get_mean_list(self, keys: Iterable[str]):
        return [self.get_mean(key) for key in keys]

    def get_mean_dict(self, keys: Iterable[str]):
        return {key: self.get_mean(key) for key in keys}

    @property
    def stats_mean(self):
        return self.get_mean_dict(self.logger_keys)

    @property
    def logger_keys(self):
        return self.log_data.keys()

    def display_tabular(self, display_keys=None):
        """Display the keys of interest in a tabular format."""
        if not display_keys:
            display_keys = sorted(self.logger_keys)
        key_lens = [len(key) for key in self.logger_keys]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in display_keys:
            val = self.log_data[key].mean
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
        print("-" * n_slashes, flush=True)

    def print(self, msg, color="green"):
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))
