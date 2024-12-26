import os

import wandb


class Logger(object):
    def __init__(self, logging, use_wandb, run_name):
        self.logging = logging
        self.use_wandb = use_wandb

        self.wandb_log = {}

        self.step = 0

        if self.use_wandb:
            wandb_key = os.environ.get('WANDB_KEY', None)
            assert wandb_key is not None, "You have to setup WANDB_KEY environment variable"
            wandb.login(key=wandb_key.strip())
            wandb.init(name=run_name)
   
    def log_wandb(self, values_dict: dict) -> None:
        if not self.use_wandb:
            return
        self.wandb_log.update(values_dict)

    def flush_wandb(self) -> None:
        if self.use_wandb:
            wandb.log(self.wandb_log)
        self.wandb_log.clear()
        self.step += 1