"""Train on input data"""
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import clearml

from visual_speedometer import data_utils
from visual_speedometer import models


@dataclass
class Config:
    """Help string for this group of command-line arguments"""
    # Data root directory
    data_root: str = "./data"
    # Batch size
    batch_size: int = 64
    # Maximum epochs to train
    epochs: int = 1e9
    # Random seed
    seed: int = 0
    # Train split
    train_split: float = 0.8
    # Dataset type to train on. 'pt' to train on pytorch optical flow. 'cv' to train on OpenCV optical flow.
    dataset_type: str = 'pt'

def collect_statistics(dataset, logger, tag='train'):
    y_array = []
    for i, (x, y) in enumerate(dataset):
        y_array.append(y)
    y_array = np.array(y_array)
    logger.add_histogram(f"labels_{tag}", y_array)

def train_loop(dataloader, model, loss_fn, optimizer, device):
    """Train loop"""
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def test_loop(dataloader, model, loss_fn, optimizer, device):
#     """Test loop"""
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         q = np.quantile(X.numpy(), [0.0, 0.25, 0.5, 0.75, 1.0], axis=(2,3))
#         row = np.hstack(([y.numpy()], q))
#         # For first batch
#         if batch == 0:
#             output = row
#         else:
#             output = np.vstack((output, row))
#     np.savetxt('quantiles.txt', output)

def test_loop(dataset):
    for i, (X, y) in tqdm(enumerate(dataset)):
        q = np.quantile(X, [0.0, 0.25, 0.5, 0.75, 1.0], axis=(1,2))
        row = np.append(y, q)
        if i == 0:
            output = row
            print(X.shape, y.shape)
        else:
            output = np.vstack((output, row))
    np.savetxt('quantiles.txt', output)



def train():
    """Training function"""
    task = clearml.Task.init(project_name="visual_speedometer", task_name="Toy model training")
    task.set_tags(['raft'])
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(Config, dest="config")

    args = parser.parse_args()
    cfg = args.config
    print(f"args: {args}")  # DEBUG"
    print("config:", cfg)

    pl.seed_everything(cfg.seed)

    if cfg.dataset_type == 'cv':
        dataset = data_utils.SpeedometerDataset('./data/custom/train.txt', './data/custom/train')
    elif cfg.dataset_type == 'pt':
        dataset = data_utils.FlowDataset('./data/custom/train.txt', './data/custom/processed/train.flows.pt', )
    # Split train/valid set
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_length = int(dataset_size * cfg.train_split)
    indices_train = indices[:train_length]
    indices_valid = indices[train_length:]
    dataset_train = torch.utils.data.Subset(dataset, indices_train)
    dataset_valid = torch.utils.data.Subset(dataset, indices_valid)
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [0.8, 0.2])
    dataset_test = dataset
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=16,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=cfg.batch_size,
        num_workers=16,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.batch_size,
        num_workers=16,
    )
    model = models.SpeedometerModule()
    # Dummy batch to initialize parameters for layz pytorch layers
    for X, y in train_loader:
        model(X)
        break

    # Define callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    speed_printing = models.SpeedPrintingCallback(Path(dataset_test.label_file).with_suffix('.pred.txt'))
    callbacks = [
        lr_monitor,
        early_stopping,
        speed_printing,
    ]

    logger = TensorBoardLogger(save_dir='./')
    # collect_statistics(dataset_train, logger, 'train')
    # collect_statistics(dataset_valid, logger, 'valid')

    trainer = pl.Trainer(
        max_epochs=int(cfg.epochs),
        accelerator='gpu',
        devices=1,
        log_every_n_steps=50,
        # fast_dev_run=True,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # overfit_batches=1,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    print()

    trainer.test(dataloaders=test_loader)

    # test_loop(dataset)
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch+1}\n-------------------------------")
    #     train_loop(train_loader, model, loss_fn, optimizer, device)


if __name__ == "__main__":
    train()
