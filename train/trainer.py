import torch
import torchvision
import torchvision.transforms as T
import wandb

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.segmentation import Segmentation
from utils.utils_vis import *


class InfiniteDataLoader(object):
    def __init__(self, split, shuffle, batch_size):
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        def tmp(x):
            x[x == 255] = 0
            return x
        t_transforms = T.Compose([
            T.Resize((224, 224)),
            T.PILToTensor(),
            tmp
        ])
        self.dataset = torchvision.datasets.VOCSegmentation(
            root="data", 
            download=True, 
            transform=transforms,
            target_transform=t_transforms,
            image_set=split
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


class Trainer(object):
    def __init__(self, config):
        self.config = config

        self.num_steps = config.train.num_steps
        self.batch_size = config.train.batch_size

        self.steps_per_val = config.val.steps_per_val
        self.val_batch_size = config.val.batch_size
        self.val_size = config.val.val_size

        self.train_dataloader = InfiniteDataLoader(
            "train", True, self.batch_size
        )
        self.val_dataloader = InfiniteDataLoader(
            "val", False, self.val_batch_size
        )

        self.model = Segmentation(
            config.backbone,
            config.head
        ).to("cuda")
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5) #TODO: add parametrization

        self.losses = [
            instantiate(cfg) for cfg in OmegaConf.to_container(config.train.losses).values()
        ]

        self.metrics = []
        # self.metrics = [
        #     instantiate(cfg) for cfg in OmegaConf.to_container(config.val.metrics).values()
        # ]

        self.logger  = instantiate(
            config.logger, run_name=f"{self.model.get_name()}"
        )


    def run_train(self):
        for step in tqdm(range(self.num_steps)):
            self._train_step()
            if step % self.steps_per_val == 0:
                self._validate()
            
            self.logger.flush_wandb()
            torch.cuda.empty_cache()

    def _train_step(self):
        self.model.train()
        batch = self.train_dataloader.__next__()
    
        pixel_values = batch[0].to("cuda")
        labels = batch[1].type(torch.LongTensor).to("cuda")

        logits = self.model(pixel_values)

        total_loss = 0
        loss_dict = {}
        for loss in self.losses:
            loss_val = loss(logits, labels)
            total_loss = total_loss + loss_val
            loss_dict[f"train/{loss.get_name()}"] = loss_val
        
        self.logger.log_wandb(loss_dict)
        total_loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def _validate(self):
        self.model.eval()

        metric_dict = {}
        for step in range(0, self.val_size, self.val_batch_size):
            batch = self.val_dataloader.__next__()
            pixel_values = batch[0].to("cuda")
            labels = batch[1].type(torch.LongTensor).to("cuda")

            logits = self.model(pixel_values)
    
            for metric in self.metrics:
                name = f"val/{metric.get_name()}"
                if name not in metric_dict.keys():
                    metric_dict[name] = []
                metric_dict[name].append(metric(logits, labels))

            if step == 0:
                color_map = logits.argmax(dim=1)
                pixel_values = (pixel_values * 255).cpu()
                color_map = visualize_map(pixel_values, color_map.squeeze().cpu())
                true_map = visualize_map(pixel_values, labels.squeeze().cpu())
                image_grid = generate_grid(pixel_values, true_map, color_map)
                image_grid = np.transpose(image_grid, (1, 2, 0))
                image_grid = wandb.Image(image_grid)
                metric_dict["Segmentation Map"] = image_grid
        
        for metric in self.metrics:
            name = f"val/{metric.get_name()}"
            metric_dict[name] = np.mean(metric_dict[name])
        
        self.logger.log_wandb(metric_dict)

    