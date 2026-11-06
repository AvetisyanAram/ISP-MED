import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from scipy import signal

from .preprocessing import preprocess_ecg_record


class UnetDataloader(Dataset):
    def __init__(
        self,
        ecg_info,
        cfg,
        training_frequency,
        ecg_length,
        data_base_path="",
        set_to_zero_ends=False,
    ):
        super().__init__()
        self.ecg_info = ecg_info
        self.cfg = cfg
        self.training_frequency = training_frequency
        self.ecg_length = ecg_length
        self.data_base_path = data_base_path
        self.set_to_zero_ends = set_to_zero_ends

    def __len__(self):
        return len(self.ecg_info)

    def __getitem__(self, index):
        row = self.ecg_info.iloc[index]
        file_path = self.data_base_path + str(row["ID"])

        ecg_record, file_frequency = self.read_ecg_record(file_path)
        wfdb_annotation = wfdb.rdann(file_path, "ii")
        mask = self.convert_wfdb_to_target(wfdb_annotation)
        mask = self.resample_segmentation_target(mask, file_frequency)
        if self.set_to_zero_ends and mask:
            ecg_record[:, : mask[0][1]] = ecg_record[:, mask[-1][2] :] = 0
        segment_annotation_target = self.create_annotation_mask(
            mask, ecg_record.shape[1]
        )

        return (
            torch.tensor(ecg_record, dtype=torch.float),
            segment_annotation_target,
            file_path,
        )

    def read_ecg_record(self, file_path, file_frequency=None):
        ecg_record = wfdb.rdrecord(file_path)
        if file_frequency is None:
            file_frequency = ecg_record.fs
        file_frequency = int(file_frequency)
        ecg_record = ecg_record.p_signal.T.astype("float64")

        ecg_record = np.array(
            [
                signal.resample(lead, int(self.ecg_length * self.training_frequency))
                for lead in ecg_record
            ]
        )

        ecg_record = preprocess_ecg_record(
            ecg_record,
            ecg_length=self.ecg_length,
            frequency=self.training_frequency,
        )

        return ecg_record, file_frequency

    def resample_segmentation_target(self, mask, file_frequency):
        if file_frequency == self.training_frequency:
            return mask

        scale = self.training_frequency / file_frequency
        return [
            (target, int(round(start * scale)), int(round(end * scale)))
            for target, start, end in mask
        ]

    def convert_wfdb_to_target(self, wfdb_annotation):
        symbol_to_index = {"p": 0, "N": 1, "t": 2}
        target = []
        symbols = wfdb_annotation.symbol
        samples = wfdb_annotation.sample
        for i in range(0, len(symbols), 3):
            triplet = symbols[i : i + 3]
            if len(triplet) == 3 and triplet[0] == "(" and triplet[2] == ")":
                label = triplet[1]
                if label in symbol_to_index:
                    target.append((symbol_to_index[label], samples[i], samples[i + 2]))
        return target

    def create_annotation_mask(self, mask, record_len):
        annotation_mask = torch.zeros((4, record_len), dtype=torch.float)

        for annotation in mask:
            target = int(annotation[0])
            start = max(0, int(annotation[1]))
            end = min(record_len, int(annotation[2]))
            if target in (0, 1, 2):
                annotation_mask[target, start:end] = 1

        fg = torch.max(annotation_mask[:3], dim=0).values
        annotation_mask[3] = (fg == 0).float()

        return annotation_mask
