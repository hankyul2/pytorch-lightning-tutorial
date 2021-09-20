import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
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
        self.log('train/loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

train_dl = DataLoader(datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True),
                      batch_size=128)
model = BaseImageClassificationSystem()
trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_dl)