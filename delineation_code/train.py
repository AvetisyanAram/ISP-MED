from sklearn.model_selection import train_test_split
import torch
import hydra
import pandas as pd
from omegaconf import DictConfig

import models
from engine import get_model
from utils.data_utils import get_dataset_loader
from engine import train_epoch
from pred import predict


@hydra.main(version_base=None, config_path="configs", config_name="delineation_config")
def train(cfg: DictConfig):

    ecg_info = pd.read_csv(cfg.train_csv_path)
    if cfg.dataset == "isp":
        ecg_info = ecg_info[ecg_info.segmented]

    train_ecg_info, valid_ecg_info = train_test_split(
        ecg_info, test_size=0.2, random_state=42
    )
    valid_ecg_info, _ = train_test_split(valid_ecg_info, test_size=0.5, random_state=42)

    train_loader = get_dataset_loader(train_ecg_info, cfg, cfg.train_data_base_path)
    valid_loader = get_dataset_loader(
        valid_ecg_info, cfg, cfg.train_data_base_path, drop_last=False
    )

    device = cfg.training_params.device

    model = get_model(cfg)
    model.to(device)

    loss_args = cfg.training_params.loss_args
    criterion = getattr(models, cfg.training_params.loss_class)(**loss_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training_params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=cfg.training_params.factor,
        patience=cfg.training_params.scheduler_patience,
    )
    best_metrics = 1000
    epochs_no_improve = 0

    for ep in range(cfg.training_params.epochs):
        print("Epoch number:", ep)
        train_loss, val_loss, epochs_no_improve, best_metrics = train_epoch(
            model,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            scheduler,
            epochs_no_improve,
            best_metrics,
            device,
            cfg,
            epoch=ep,
        )
        print(
            f"\ntrain_loss:{train_loss}\n \
                \nval_loss: {val_loss}\n"
        )

        if epochs_no_improve >= cfg.training_params.early_stopping_patience:
            print("Early stopping")
            break

    cfg.predict.checkpoint_path = f"{cfg.save_dir}/{cfg.dataset}.pt"
    cfg.dataset = cfg.test_dataset
    predict(cfg)


if __name__ == "__main__":
    train()
