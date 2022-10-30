import logging
import random
import subprocess
from datetime import datetime
from typing import Sequence
import copy
from  logging import Logger

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
import rich.syntax
import rich.tree


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

def print_config(
    config: DictConfig,
    logger: Logger,
    fields: Sequence[str] = (
        "name",
        "device",
        "working_dir",
        "random_seed",
        "model",
        "train",
        "test",
        "gen_dataset",
        "data",
        "dist",
        "log",
        "job_logging_cfg",
        "dist"
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    all_cfg_str = OmegaConf.to_yaml(config)
    logger.info("Config:\n" + all_cfg_str)

