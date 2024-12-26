import os

import wandb


class Logger(object):
    def __init__(self, config):
        self.config = config

        self.wandb_log = {}

        self.step = 0

        if self.config.logging_wandb:
            wandb_key = os.environ.get('WANDB_KEY', None)
            assert wandb_key is not None, "You have to setup WANDB_KEY environment variable"
            wandb.login(key=wandb_key.strip())
   
    def log_wand(self, values_dict: dict) -> None:
        if not self.config.logging_wandb:
            return
        self.wandb_log.update(values_dict)

    def flush_log(self) -> None:
        if self.config.logging_wandb:
            wandb.log(self.wandb_dict)
        self.wandb_log.clear()
        self.step += 1