from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import wfdb


def _resolve_signal_path(root_dir: str | Path, file_name: str) -> str:
    return os.path.join(str(root_dir), file_name)


def read_npz_signal(root_dir: str | Path, file_name: str, array_name: str = "arr_0") -> tuple[np.ndarray, int]:
    file_path = _resolve_signal_path(root_dir, file_name)
    signal = np.load(file_path)[array_name].astype(np.float32)
    return signal, -1


def read_wfdb_signal(root_dir: str | Path, file_name: str) -> tuple[np.ndarray, int]:
    file_path = _resolve_signal_path(root_dir, file_name)
    if file_path.endswith(".hea") or file_path.endswith(".dat"):
        file_path = file_path[:-4]
    record = wfdb.rdrecord(file_path)
    return record.p_signal.T.astype(np.float32), int(record.fs)
