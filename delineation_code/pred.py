import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from engine import eval_fn, get_model
from utils.data_utils import get_dataset_loader
import models


@hydra.main(version_base=None, config_path="configs", config_name="delineation_config")
def predict(cfg: DictConfig):
    ecg_info = pd.read_csv(cfg.test_csv_path)
    if cfg.dataset == "isp":
        ecg_info = ecg_info[ecg_info.segmented]

    _, valid_ecg_info = train_test_split(ecg_info, test_size=0.2, random_state=42)
    _, test_ecg_info = train_test_split(valid_ecg_info, test_size=0.5, random_state=42)

    device = cfg.training_params.device
    print(f"Checkpoint path: {cfg.predict.checkpoint_path}")

    test_loader = get_dataset_loader(
        test_ecg_info, cfg, cfg.test_data_base_path, drop_last=False, shuffle=False
    )
    model_info = torch.load(
        cfg.predict.checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    state_dict = model_info["model"]
    model = get_model(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    loss_args = cfg.training_params.loss_args
    criterion = getattr(models, cfg.training_params.loss_class)(**loss_args)
    eval_fn(
        test_loader,
        model,
        criterion,
        device,
        cfg.training_params.window_tolerance,
        cfg,
    )


if __name__ == "__main__":
    predict()
