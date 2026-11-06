from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import build_target_matrix, parse_label_list
from .preprocess import ECGPreprocessor
from .readers import read_npz_signal, read_wfdb_signal


def load_ptbxl_dataframe(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_isp_dataframe(map_file: str | Path, data_dir: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(map_file).rename(columns={"ID": "record_id"})
    if "scp_codes" not in frame.columns:
        frame["scp_codes"] = frame["unified_decision"]
    frame["patient_id"] = frame["record_id"]
    frame["file_name"] = frame["record_id"].astype(str)
    frame["backend"] = "wfdb"
    frame["array_name"] = "arr_0"
    return frame


class ECGClassificationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_root: str | Path,
        backend: str,
        pathology_names: list[str],
        merge_map: dict[str, list[str]] | None,
        label_column: str,
        preprocessor: ECGPreprocessor,
    ):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.signal_root = Path(signal_root)
        self.backend = backend
        self.pathology_names = pathology_names
        self.merge_map = merge_map or {}
        self.label_column = label_column
        self.preprocessor = preprocessor
        self.target_labels = list(self.merge_map.keys()) if self.merge_map else list(self.pathology_names)
        self.targets = build_target_matrix(self.dataframe[self.label_column], self.pathology_names, self.merge_map)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _read_signal(self, row: dict):
        if self.backend == "npz":
            signal, _ = read_npz_signal(self.signal_root, row["file_name"], row.get("array_name", "arr_0"))
            sampling_rate = int(row["sampling_rate"])
            return signal, sampling_rate
        if self.backend == "wfdb":
            return read_wfdb_signal(self.signal_root, row["file_name"])
        raise ValueError(f"Unsupported backend '{self.backend}'")

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index].to_dict()
        signal, sampling_rate = self._read_signal(row)
        signal = self.preprocessor(signal, sampling_rate)
        return {
            "signal": torch.as_tensor(signal, dtype=torch.float32),
            "target": torch.as_tensor(self.targets[index], dtype=torch.float32),
            "record_id": str(row.get("record_id", index)),
            "raw_labels": parse_label_list(row.get(self.label_column, [])),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return {
            "signal": torch.stack([item["signal"] for item in batch], dim=0),
            "target": torch.stack([item["target"] for item in batch], dim=0),
            "record_id": [item["record_id"] for item in batch],
            "raw_labels": [item["raw_labels"] for item in batch],
        }
