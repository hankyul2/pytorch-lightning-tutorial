from typing import Optional

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch.nn.functional as F
import pytorch_lightning as pl


class BaseImageClassificationSystem(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(1, 64, 3), nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(torch.flatten(self.backbone(x), 1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.fc(torch.flatten(self.backbone(x), 1))
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super(MNISTDataModule, self).__init__()
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)
        datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(), download=True)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            mnist_train = datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)
            self.train_ds, self.valid_ds = random_split(mnist_train, [55000, 5000])
        elif stage in (None, 'test'):
            mnist_test = datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(), download=True)
            self.test_ds = mnist_test

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size)


dl = MNISTDataModule(batch_size=128)
model = BaseImageClassificationSystem()
trainer = pl.Trainer(gpus=8, accelerator='ddp', max_epochs=100, profiler="simple")
trainer.fit(model, dl)
