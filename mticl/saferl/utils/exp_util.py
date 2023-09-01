import os
from typing import Optional, Union, List, Dict

import numpy as np
import random
import torch


def seed_all(seed=1029, others: Optional[list] = None):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if others is not None:
        if hasattr(others, "seed"):
            others.seed(seed)
            return True
        try:
            for item in others:
                if hasattr(item, "seed"):
                    item.seed(seed)
        except:
            pass


def get_cfg_value(config, key):
    if key in config:
        value = config[key]
        if isinstance(value, list):
            suffix = ""
            for i in value:
                suffix += str(i)
            return suffix
        return str(value)
    for k in config.keys():
        if isinstance(config[k], dict):
            res = get_cfg_value(config[k], key)
            if res is not None:
                return res
    return "None"


def get_name_string(keys: Union[List, Dict], config: Dict):
    name = ""
    if isinstance(keys, List):
        for k in keys:
            name += k + "_" + get_cfg_value(config, k) + "_"
        return name[:-1]
    elif isinstance(keys, Dict):
        for k in keys:
            name += keys[k] + "_" + get_cfg_value(config, k) + "_"
        return name[:-1]
