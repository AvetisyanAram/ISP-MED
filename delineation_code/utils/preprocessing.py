import numpy as np


def preprocess_ecg_record(
    ecg_record: np.ndarray,
    ecg_length: float,
    frequency: int,
) -> np.ndarray:
    """Apply per-lead z-normalization and enforce the expected signal length."""

    ecg_record = _z_norm(ecg_record)
    return _pad_or_truncate(ecg_record, target_length=int(ecg_length * frequency))


def _z_norm(ecg_record: np.ndarray) -> np.ndarray:
    ecg_record = np.asarray(ecg_record, dtype=np.float32)
    mean = ecg_record.mean(axis=1, keepdims=True)
    std = ecg_record.std(axis=1, keepdims=True)

    normalized = np.zeros_like(ecg_record, dtype=np.float32)
    np.divide(ecg_record - mean, std, out=normalized, where=std != 0)
    return normalized


def _pad_or_truncate(ecg_record: np.ndarray, target_length: int) -> np.ndarray:
    current_length = ecg_record.shape[1]
    if current_length == target_length:
        return ecg_record

    if current_length > target_length:
        return ecg_record[:, :target_length]

    pad_width = target_length - current_length
    return np.pad(ecg_record, ((0, 0), (0, pad_width)), mode="constant")
