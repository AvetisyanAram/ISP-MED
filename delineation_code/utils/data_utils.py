import math

from torch.utils.data import DataLoader

from .dataloaders import PQRSTDataloader
from .unet_dataloader import UnetDataloader


def get_dataset_loader(
    ecg_info,
    cfg,
    data_base_path,
    drop_last=True,
    shuffle=True,
):
    set_to_zero_ends = True if cfg.test_dataset in ("ludb") else False

    if cfg.model_params.model_name == "ECGCode":
        intervals_count = math.ceil(
            cfg.ecg_record_params.resampled_frequency
            * cfg.ecg_record_params.observed_ecg_length
            / 2**5
        )

        ecg_dataset = PQRSTDataloader(
            ecg_info=ecg_info,
            frequency=cfg.ecg_record_params.resampled_frequency,
            ecg_length=cfg.ecg_record_params.observed_ecg_length,
            intervals_count=intervals_count,
            data_base_path=data_base_path,
            dataset=cfg.dataset,
            set_to_zero_ends=set_to_zero_ends,
        )
    else:
        ecg_dataset = UnetDataloader(
            ecg_info=ecg_info,
            cfg=cfg,
            training_frequency=cfg.ecg_record_params.resampled_frequency,
            ecg_length=cfg.ecg_record_params.observed_ecg_length,
            data_base_path=data_base_path,
            set_to_zero_ends=set_to_zero_ends,
        )

    ecg_segment_dataloader = DataLoader(
        ecg_dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training_params.num_workers,
        drop_last=drop_last,
    )
    return ecg_segment_dataloader
