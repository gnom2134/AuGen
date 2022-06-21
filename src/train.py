from .dataset import prepare_dataset
from .models.cgan import CGANModel
from .models.cvae import CVAEModel
from .models.cvaegan import CVAEGANModel

import torch
import pytorch_lightning as pl
from typing import Dict, Any


def train_with_dataset(
    dataset: torch.utils.data.Dataset,
    model_name: str,
    model_params: Dict[str, Any],
    save_model_path: str = None,
    trainer_params: Dict[str, Any] = None,
    wandb_logger_params: Dict[str, Any] = None,
    batch_size: int = 8,
):
    if trainer_params is None:
        trainer_params = {}
    if wandb_logger_params is None:
        wandb_logger_params = {}

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    model = None
    if model_name == "CVAE":
        model = CVAEModel(**model_params)
    elif model_name == "CGAN":
        model = CGANModel(**model_params)
    elif model_name == "CVAEGAN":
        model = CVAEGANModel(**model_params)
    else:
        raise AttributeError(f"Unsupported model: {model_name}")

    wandb_logger = pl.loggers.WandbLogger(**wandb_logger_params)
    trainer_params["logger"] = wandb_logger
    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model, train_loader)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

    return model


def train(
    dataset_name: str,
    imgpath: str,
    csvpath: str,
    model_name: str,
    model_params: Dict[str, Any],
    save_model_path: str = None,
    trainer_params: Dict[str, Any] = None,
    wandb_logger_params: Dict[str, Any] = None,
    batch_size: int = 8,
):
    dataset = prepare_dataset(dataset_name, imgpath, csvpath)

    return train_with_dataset(
        dataset, model_name, model_params, save_model_path, trainer_params, wandb_logger_params, batch_size
    )
