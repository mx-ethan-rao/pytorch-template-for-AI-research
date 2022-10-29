import os
import os.path as osp
from collections import OrderedDict
import copy
import math

import torch
import torch.nn
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from utils.utils import get_logger, is_logging_process


class Model_handler:
    def __init__(self, cfg, net_arch, loss_f, writer, rank=0):
        self.cfg = cfg
        self.device = self.cfg.device
        self.net = net_arch.to(self.device)
        self.writer = writer
        self.rank = rank
        if self.device != "cpu" and self.cfg.dist.gpus != 0:
            self.net = DDP(self.net, device_ids=[self.rank])
        self.step = 0
        self.epoch = -1
        self._logger = get_logger(cfg, os.path.basename(__file__))

        # init optimizer
        optimizer_mode = self.cfg.train.optimizer.mode
        if optimizer_mode == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), **(self.cfg.train.optimizer[optimizer_mode])
            )
        else:
            raise Exception("%s optimizer not supported" % optimizer_mode)

        # init loss
        self.loss_f = loss_f
        self.log = OmegaConf.create()
    
    def train_model(self, train_loader):
        logger = get_logger(self.cfg, os.path.basename(__file__), disable_console=True)
        
        self.net.train()
        for model_input, model_target in tqdm(train_loader, leave = False, desc="Training/Batch:", unit='batch'):
            self.optimize_parameters(model_input, model_target)
            loss = self.log.loss_v
            self.step += 1

            if is_logging_process() and (loss > 1e8 or math.isnan(loss)):
                logger.error("Loss exploded to %.02f at step %d!" % (loss, self.step))
                raise Exception("Loss exploded")

            if self.step % self.cfg.log.summary_interval == 0:
                if self.writer is not None:
                    self.writer.logging_with_step(loss, self.step, "train_loss")
                if is_logging_process():
                    logger.info("Train Loss %.04f at step %d" % (loss, self.step))

    def optimize_parameters(self, model_input, model_target):
        self.net.train()
        self.optimizer.zero_grad()
        output = self.run_network(model_input)
        loss_v = self.loss_f(output, model_target.to(self.device))
        loss_v.backward()
        self.optimizer.step()
        # set log
        self.log.loss_v = loss_v.item()

    def test_model(self, test_loader):
        logger = get_logger(self.cfg, os.path.basename(__file__), disable_console=True)
        self.net.eval()
        total_test_loss = 0
        test_loop_len = 0
        with torch.no_grad():
            for model_input, model_target in tqdm(test_loader, leave = False, desc="Testing/Batch:"):
                output = self.inference(model_input)
                loss_v = self.loss_f(output, model_target.to(self.cfg.device))
                if self.cfg.dist.gpus > 0:
                    # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                    torch.distributed.all_reduce(loss_v)
                    loss_v /= torch.tensor(float(self.cfg.dist.gpus))
                total_test_loss += loss_v.to("cpu").item()
                test_loop_len += 1

            total_test_loss /= test_loop_len

            if self.writer is not None:
                self.writer.logging_with_step(total_test_loss, self.step, "test_loss")
            if is_logging_process():
                logger.info("Test Loss %.04f at step %d" % (total_test_loss, self.step))

    def inference(self, model_input):
        self.net.eval()
        output = self.run_network(model_input)
        return output

    def run_network(self, model_input):
        model_input = model_input.to(self.device)
        output = self.net(model_input)
        return output

    def save_network(self, save_file=True):
        if is_logging_process():
            net = self.net.module if isinstance(self.net, DDP) else self.net
            state_dict = net.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.to("cpu")
            if save_file:
                save_filename = "%s_%d.pt" % (self.cfg.name, self.step)
                save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
                torch.save(state_dict, save_path)
                if self.cfg.log.use_wandb:
                    wandb.save(save_path)
                if is_logging_process():
                    self._logger.info("Saved network checkpoint to: %s" % save_path)
            return state_dict

    def load_network(self, loaded_net=None):
        add_log = False
        if loaded_net is None:
            add_log = True
            if self.cfg.load.wandb_load_path is not None:
                self.cfg.load.network_chkpt_path = wandb.restore(
                    self.cfg.load.network_chkpt_path,
                    run_path=self.cfg.load.wandb_load_path,
                ).name
            loaded_net = torch.load(
                self.cfg.load.network_chkpt_path,
                map_location=torch.device(self.device),
            )
        loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
        for k, v in loaded_net.items():
            if k.startswith("module."):
                loaded_clean_net[k[7:]] = v
            else:
                loaded_clean_net[k] = v

        self.net.load_state_dict(loaded_clean_net, strict=self.cfg.load.strict_load)
        if is_logging_process() and add_log:
            self._logger.info(
                "Checkpoint %s is loaded" % self.cfg.load.network_chkpt_path
            )

    def save_training_state(self):
        if is_logging_process():
            save_filename = "%s_%d.state" % (self.cfg.name, self.step)
            save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
            net_state_dict = self.save_network(False)
            state = {
                "model": net_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            }
            torch.save(state, save_path)
            if self.cfg.log.use_wandb:
                wandb.save(save_path)
            if is_logging_process():
                self._logger.info("Saved training state to: %s" % save_path)

    def load_training_state(self):
        if self.cfg.load.wandb_load_path is not None:
            self.cfg.load.resume_state_path = wandb.restore(
                self.cfg.load.resume_state_path,
                run_path=self.cfg.load.wandb_load_path,
            ).name
        resume_state = torch.load(
            self.cfg.load.resume_state_path,
            map_location=torch.device(self.device),
        )

        self.load_network(loaded_net=resume_state["model"])
        self.optimizer.load_state_dict(resume_state["optimizer"])
        self.step = resume_state["step"]
        self.epoch = resume_state["epoch"]
        if is_logging_process():
            self._logger.info(
                "Resuming from training state: %s" % self.cfg.load.resume_state_path
            )
