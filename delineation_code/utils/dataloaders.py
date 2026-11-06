import numpy as np
from scipy import signal
import torch
from torch.utils.data import Dataset
import wfdb

from .preprocessing import preprocess_ecg_record


class PQRSTDataloader(Dataset):
    def __init__(
        self,
        ecg_info,
        frequency=1000,
        ecg_length=10,
        intervals_count=313,
        data_base_path="",
        dataset="isp",
        set_to_zero_ends=False,
    ):
        super().__init__()
        self.ecg_info = ecg_info
        self.training_frequency = frequency
        self.ecg_length = ecg_length
        self.intervals_count = intervals_count
        self.data_base_path = data_base_path
        self.dataset = dataset
        self.set_to_zero_ends = set_to_zero_ends

    def __len__(self):
        return self.ecg_info.shape[0]

    def __getitem__(self, index):
        row = self.ecg_info.iloc[index]
        file_path = self.data_base_path + str(row["ID"])

        ecg_record, wfdb_annotation, file_frequency = self.read_ecg_record(
            file_path=file_path,
        )
        segment_annotation_target = self.convert_wfdb_to_target(wfdb_annotation)
        segment_annotation_target = self.resample_segmentation_target(
            segment_annotation_target,
            file_frequency,
        )
        if self.set_to_zero_ends and segment_annotation_target:
            ecg_record[:, : segment_annotation_target[0][1]] = ecg_record[
                :, segment_annotation_target[-1][2] :
            ] = 0
        segment_annotation_target = self.create_annotation_mask(
            segment_annotation_target, ecg_record.shape[1], self.intervals_count
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
        annotation = wfdb.rdann(file_path, "ii")
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

        return ecg_record, annotation, file_frequency

    def resample_segmentation_target(self, segmentation_target, file_frequency):
        if file_frequency == self.training_frequency:
            return segmentation_target

        scale = self.training_frequency / file_frequency
        return [
            (label, int(round(start * scale)), int(round(end * scale)))
            for label, start, end in segmentation_target
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

    def create_annotation_mask(self, segmentation_target, record_len, intervals_count):
        num_labels = 3
        features_per_label = 3
        annotation_target = torch.zeros(
            (intervals_count, num_labels * features_per_label)
        )

        intervals = np.linspace(0, record_len, intervals_count + 1)
        intervals = list(zip(intervals[:-1], intervals[1:]))
        interval_len = intervals[0][1] - intervals[0][0]

        for interval_ind, (interval_start, interval_end) in enumerate(intervals):
            for label, start, end in segmentation_target:
                base_idx = label * features_per_label

                if interval_start <= start <= interval_end:
                    annotation_target[interval_ind][base_idx] = 1
                    annotation_target[interval_ind][base_idx + 1] = (
                        start - interval_start
                    ) / interval_len

                if interval_start <= end <= interval_end:
                    annotation_target[interval_ind][base_idx] = 1
                    annotation_target[interval_ind][base_idx + 2] = (
                        end - interval_start
                    ) / interval_len

                if start <= interval_end <= end:
                    annotation_target[interval_ind][base_idx + 2] = 1

                if start <= interval_start and end >= interval_end:
                    annotation_target[interval_ind][base_idx] = 1
                    annotation_target[interval_ind][base_idx + 1] = 0
                    annotation_target[interval_ind][base_idx + 2] = 1

        return annotation_target.T
