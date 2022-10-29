import logging
import random
import subprocess
from datetime import datetime
import copy

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_logging_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_logger(cfg, name=None, disable_console=False):
    if disable_console:
        # stop output to stdout; only output to log file
        if 'console' in cfg.job_logging_cfg.root.handlers:
            cfg = copy.deepcopy(cfg)
            cfg.job_logging_cfg.root.handlers.remove('console')
    # log_file_path is used when unit testing
    if is_logging_process():
        logging.config.dictConfig(
            OmegaConf.to_container(cfg.job_logging_cfg, resolve=True)
        )
        return logging.getLogger(name)

