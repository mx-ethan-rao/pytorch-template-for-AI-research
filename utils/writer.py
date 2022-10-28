import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
from omegaconf.dictconfig import DictConfig


class Writer(SummaryWriter):
    def __init__(self, cfg, logdir):
        self.cfg = cfg
        if cfg.log.use_tensorboard:
            self.tensorboard = SummaryWriter(logdir)
        if cfg.log.use_wandb:
            wandb_init_conf = cfg.log.wandb_init_conf
            wandb.init(config=self.wandb_formatter(cfg), **wandb_init_conf)

    def logging_with_step(self, value, step, logging_name):
        if self.cfg.log.use_tensorboard:
            self.tensorboard.add_scalar(logging_name, value, step)
        if self.cfg.log.use_wandb:
            wandb.log({logging_name: value}, step=step)

    def wandb_formatter(self, cfg):
        formatted_cfg = dict()
        return self.recursive_format('general/', formatted_cfg, cfg)

    def recursive_format(self, prefix, formatted_cfg, cfg):
        for key, val in dict(cfg).items():
            if not isinstance(val, DictConfig):
                formatted_cfg[prefix + str(key)] = val
            else:
                if prefix == 'general/':
                    self.recursive_format(str(key) + '/', formatted_cfg, val)
                else:
                    self.recursive_format(prefix + str(key) + '/', formatted_cfg, val)
        return formatted_cfg
