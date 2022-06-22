import torch

from typing import Dict, Any, Tuple

from ..dataset import prepare_dataset, AugmentedDataset
from ..train import train_with_dataset
from ..models import CVAEGANModel



def accuracy(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader):
    good = 0
    all_ = 0
    for batch in test_loader:
        y = batch["lab"]
        x = batch["img"]
        all_ += y.shape[0]
        y_pred = model(x)
        good += torch.sum(y == torch.round(y_pred))

    return good / all_


def improvement_experiment(
    dataset_name: str,
    imgpath: str,
    csvpath: str,
    augmented_dataset_dir: str,
    gen_model_name: str,
    gen_model_params: Dict[str, Any],
    cls_model_name: str,
    cls_model_params: Dict[str, Any],
    gen_save_model_path: str = None,
    gen_trainer_params: Dict[str, Any] = None,
    gen_wandb_logger_params: Dict[str, Any] = None,
    cls_save_model_path: str = None,
    cls_trainer_params: Dict[str, Any] = None,
    cls_wandb_logger_params: Dict[str, Any] = None,
    batch_size: int = 8,
    test_ratio: float = 0.1,
) -> Tuple[torch.nn.Module, torch.nn.Module, float]:
    xrv_dataset = prepare_dataset(dataset_name, imgpath, csvpath)

    seed=42
    train_set, test_set = torch.utils.data.random_split(
        xrv_dataset, [11000, 1954],
        generator=torch.Generator().manual_seed(seed)
    )

    #gen_model = train_with_dataset(
    #    train_set,
    #    gen_model_name,
    #    gen_model_params,
    #    gen_save_model_path,
    #    gen_trainer_params,
    #    gen_wandb_logger_params,
    #    batch_size,
    #)
    gen_model = CVAEGANModel(**gen_model_params)
    gen_model.load_state_dict(torch.load("D:/programs/projects/AuGen/data/CVAEGAN_on_siim_dataset.state",
                                         map_location= 'cpu'))

    augmented_dataset = AugmentedDataset(
        torch.utils.data.DataLoader(train_set, batch_size=batch_size), gen_model, augmented_dataset_dir
    )

    cls_model = train_with_dataset(
        augmented_dataset,
        cls_model_name,
        cls_model_params,
        cls_save_model_path,
        cls_trainer_params,
        cls_wandb_logger_params,
        batch_size,
    )

    test_accuracy = accuracy(cls_model, torch.utils.data.DataLoader(test_set, batch_size=batch_size))

    return gen_model, cls_model, test_accuracy
