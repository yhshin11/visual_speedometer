"""Models"""
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

@dataclass
class ModelConfig():
    """Model parameters"""
    in_channels = 2
    out_channels1 = 4
    out_channels2 = 8
    out_channels3 = 16


class Speedometer(nn.Module):  # inheriting from nn.Module!
    """nn.Module that takes input optical flows and predicts speed of car."""
    def __init__(self, ):
        super(Speedometer, self).__init__()
        # Define layers
        self.conv1 = nn.LazyConv2d(out_channels=32, kernel_size=3)
        self.conv2 = nn.LazyConv2d(out_channels=64, kernel_size=3)
        self.conv3 = nn.LazyConv2d(out_channels=128, kernel_size=3)
        self.conv4 = nn.LazyConv2d(out_channels=256, kernel_size=3)
        self.flat = nn.Flatten()
        self.fc1 = nn.LazyLinear(out_features=1024, )
        self.fc2 = nn.LazyLinear(out_features=16, )
        self.fc3 = nn.LazyLinear(out_features=4, )
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.head = nn.LazyLinear(out_features=1, )

    def forward(self, x):
        """Forward method"""
        x = torchvision.transforms.functional.resize(x, size=(30, 40))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.max_pool2d(x, 2)
        x = self.flat(x)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.head(x)
        x = x.flatten()
        # x = x * 10.0
        return x

class SpeedometerModule(pl.LightningModule):
    """lightning module"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = Speedometer(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # tqdm.write(f'{y[0].item():.3f}, {y_hat[0].item():.3f}')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        tqdm.write(f'{y[0].item():.3f}, {y_hat[0].item():.3f}')

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        return y_hat

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

class SpeedPrintingCallback(pl.callbacks.Callback):
    """Callback to print predictions to text file"""
    def __init__(self, file_path):
        self.file_path = file_path
        if Path(file_path).exists():
            print(f"Warning: File {str(file_path)} exists! Overwriting...")
        self.file = open(self.file_path, "w", encoding="utf-8")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Iterate through batch
        for idx in range(outputs.shape[0]):
            output = outputs[idx].item()
            text_output = f"{output:.3f}\n"
            self.file.write(text_output)

    def teardown(self, trainer, pl_module, stage):
        pass