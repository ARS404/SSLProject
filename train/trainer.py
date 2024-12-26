import torch
import torchvision
import torchvision.transforms as T

from hydra.utils import instantiate
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.segmentation import Segmentation


class InfiniteDataLoader(object):
    def __init__(self, split, shuffle, batch_size):
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        self.dataset = torchvision.datasets.VOCSegmentation(
            root="data", 
            download=True, 
            transform=transforms,
            target_transform=transforms,
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
        self.losses = [instantiate(cfg) for cfg in config.train.losses]

        self.logger  = instantiate(config.logger)



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
        labels = batch[1].to("cuda")

        predictions = self.model(pixel_values)

        total_loss = 0
        loss_dict = {}
        for loss in self.losses:
            loss_val = loss(predictions, labels)
            total_loss  = total_loss + loss_val
            loss_dict[loss.get_name()] = loss_val
        

        self.logger.log_wandb(loss_dict)
        total_loss.backward()
        self.optimizer.step()


        


    def _validate(self):
        pass

    