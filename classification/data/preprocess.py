from __future__ import annotations

from math import gcd

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from scipy.stats import zscore


def select_leads(signal_array: np.ndarray, leads: list[int]) -> np.ndarray:
    """Select ECG leads from a signal in `(channels, samples)` layout."""

    return signal_array[leads]


def pad_or_crop(signal_array: np.ndarray, length: int, mode: str = "constant", crop_mode: str = "first") -> np.ndarray:
    """Pad or crop a signal to a fixed number of samples."""

    if signal_array.shape[1] < length:
        pad_width = length - signal_array.shape[1]
        return np.pad(signal_array, ((0, 0), (0, pad_width)), mode=mode)
    if signal_array.shape[1] > length:
        if crop_mode == "first":
            return signal_array[:, :length]
        if crop_mode == "random":
            max_start = signal_array.shape[1] - length
            start = int(np.random.randint(0, max_start + 1))
            return signal_array[:, start:start + length]
        raise ValueError(f"Unsupported crop_mode '{crop_mode}'")
    return signal_array


def resample_signal(signal_array: np.ndarray, origin_sr: int, target_sr: int) -> np.ndarray:
    """Resample an ECG signal with polyphase filtering."""

    if origin_sr == target_sr:
        return signal_array
    divisor = gcd(int(origin_sr), int(target_sr))
    up = target_sr // divisor
    down = origin_sr // divisor
    return signal.resample_poly(signal_array, up=up, down=down, axis=1)


def z_norm(signal_array: np.ndarray) -> np.ndarray:
    """Apply per-lead z-score normalization and zero constant leads."""

    normalized = zscore(signal_array, axis=1, nan_policy="raise")
    same_values = np.all(signal_array == signal_array[:, 0][:, np.newaxis], axis=1)
    normalized[same_values] = 0
    return normalized.astype(np.float32)


def global_mean_std_norm(
    signal_array: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip_value: float | None = None,
) -> np.ndarray:
    """Normalize an ECG signal with precomputed per-lead statistics."""

    mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(-1, 1)
    safe_std = np.where(std == 0, 1.0, std)
    normalized = (signal_array.astype(np.float32) - mean) / safe_std
    if clip_value is not None:
        normalized = np.clip(normalized, -float(clip_value), float(clip_value))
    return normalized.astype(np.float32)


def global_scalar_std_norm(
    signal_array: np.ndarray,
    mean: float,
    std: float,
    clip_value: float | None = None,
) -> np.ndarray:
    """Normalize an ECG signal using scalar mean/std over all samples."""

    safe_std = std if std != 0 else 1.0
    normalized = (signal_array.astype(np.float32) - float(mean)) / float(safe_std)
    if clip_value is not None:
        normalized = np.clip(normalized, -float(clip_value), float(clip_value))
    return normalized.astype(np.float32)


def standardize(
    signal_array: np.ndarray,
    axis: tuple[int, ...] | list[int] | int = (-1, -2),
) -> np.ndarray:
    """Standardize an ECG signal over configurable axes and zero constant slices."""

    if isinstance(axis, list):
        axis = tuple(axis)
    mean = np.mean(signal_array, axis=axis, keepdims=True)
    std = np.std(signal_array, axis=axis, keepdims=True)
    normalized = np.divide(signal_array - mean, std, out=np.zeros_like(signal_array), where=std != 0)
    return normalized.astype(np.float32)


def minmax_norm(signal_array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply per-record min-max normalization."""

    min_value = float(np.min(signal_array))
    max_value = float(np.max(signal_array))
    denom = max(max_value - min_value, eps)
    normalized = (signal_array - min_value) / denom
    return normalized.astype(np.float32)


def minmax_scale_neg1_pos1(signal_array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale each lead to [-1, 1] using per-lead min/max values."""

    min_values = np.min(signal_array, axis=1, keepdims=True)
    max_values = np.max(signal_array, axis=1, keepdims=True)
    denom = np.maximum(max_values - min_values, eps)
    scaled = 2.0 * (signal_array - min_values) / denom - 1.0
    return scaled.astype(np.float32)


def swap_avl_avf(signal_array: np.ndarray) -> np.ndarray:
    """Swap aVL/aVF leads in-place for MERL compatibility."""

    if signal_array.shape[0] <= 5:
        return signal_array
    signal_array = signal_array.copy()
    signal_array[[4, 5]] = signal_array[[5, 4]]
    return signal_array


def bandpass_filter_ecgfounder(
    signal_array: np.ndarray,
    sampling_rate: int,
    notch_frequency: float = 50.0,
    notch_quality: float = 30.0,
    lowcut: float = 0.67,
    highcut: float = 40.0,
    baseline_window_seconds: float = 0.4,
    order: int = 4,
) -> np.ndarray:
    """Apply the public ECGFounder preprocessing filter stack."""

    filtered_signal = np.zeros_like(signal_array, dtype=np.float32)
    notch_b, notch_a = iirnotch(notch_frequency, notch_quality, sampling_rate)
    band_b, band_a = butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate,
    )

    for channel_index in range(signal_array.shape[0]):
        channel = filtfilt(notch_b, notch_a, signal_array[channel_index])
        channel = filtfilt(band_b, band_a, channel)
        filtered_signal[channel_index] = channel.astype(np.float32)

    baseline = np.zeros_like(filtered_signal, dtype=np.float32)
    kernel_size = int(baseline_window_seconds * sampling_rate) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    for channel_index in range(filtered_signal.shape[0]):
        baseline[channel_index] = medfilt(filtered_signal[channel_index], kernel_size=kernel_size)
    return (filtered_signal - baseline).astype(np.float32)


def bandpass_filter_ecgjepa(
    signal_array: np.ndarray,
    sampling_rate: int,
    lowcut: float = 0.67,
    highcut: float = 40.0,
    order: int = 5,
) -> np.ndarray:
    """Apply the public ECG-JEPA band-pass filter used before downstream evaluation."""

    band_b, band_a = butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate,
    )
    filtered_signal = np.zeros_like(signal_array, dtype=np.float32)
    for channel_index in range(signal_array.shape[0]):
        filtered_signal[channel_index] = filtfilt(band_b, band_a, signal_array[channel_index]).astype(np.float32)
    return filtered_signal


def bandpass_filter_stmem(
    signal_array: np.ndarray,
    sampling_rate: int,
    lowcut: float = 0.67,
    highcut: float = 40.0,
    order: int = 5,
) -> np.ndarray:
    """Apply the official ST-MEM downstream high-pass and low-pass filters."""

    highpass_b, highpass_a = butter(N=order, Wn=lowcut, btype="highpass", fs=sampling_rate)
    lowpass_b, lowpass_a = butter(N=order, Wn=highcut, btype="lowpass", fs=sampling_rate)
    filtered_signal = np.zeros_like(signal_array, dtype=np.float32)
    for channel_index in range(signal_array.shape[0]):
        channel = filtfilt(highpass_b, highpass_a, signal_array[channel_index])
        channel = filtfilt(lowpass_b, lowpass_a, channel)
        filtered_signal[channel_index] = channel.astype(np.float32)
    return filtered_signal


def bandpass_filter_hubert_ecg(
    signal_array: np.ndarray,
    sampling_rate: int,
    lowcut: float = 0.05,
    highcut: float = 47.0,
) -> np.ndarray:
    """Apply the HuBERT-ECG FIR band-pass filter."""

    order = max(int(0.3 * sampling_rate), 1)
    taps = signal.firwin(order, [lowcut, highcut], pass_zero=False, fs=sampling_rate)
    filtered_signal = np.zeros_like(signal_array, dtype=np.float32)
    for channel_index in range(signal_array.shape[0]):
        filtered_signal[channel_index] = filtfilt(taps, [1.0], signal_array[channel_index]).astype(np.float32)
    return filtered_signal


def apply_filter(signal_array: np.ndarray, sampling_rate: int, filter_cfg: dict | None = None) -> np.ndarray:
    """Apply an optional configured filter stage to the ECG signal."""

    if filter_cfg is None:
        return signal_array
    filter_name = filter_cfg.get("name", "none")
    if filter_name in (None, "none"):
        return signal_array
    filter_params = dict(filter_cfg.get("params", {}))
    if filter_name == "ecgfounder_bandpass":
        return bandpass_filter_ecgfounder(signal_array, sampling_rate=sampling_rate, **filter_params)
    if filter_name == "ecgjepa_bandpass":
        return bandpass_filter_ecgjepa(signal_array, sampling_rate=sampling_rate, **filter_params)
    if filter_name == "stmem_bandpass":
        return bandpass_filter_stmem(signal_array, sampling_rate=sampling_rate, **filter_params)
    if filter_name == "hubert_ecg_bandpass":
        return bandpass_filter_hubert_ecg(signal_array, sampling_rate=sampling_rate, **filter_params)
    raise ValueError(f"Unsupported filter '{filter_name}'")


class ECGPreprocessor:
    """Configurable ECG preprocessing pipeline used by train and predict flows."""

    def __init__(
        self,
        leads: list[int],
        resampled_frequency: int,
        observed_length_seconds: float,
        normalization: str = "z_norm",
        padding_mode: str = "constant",
        crop_mode: str = "first",
        filter_cfg: dict | None = None,
        normalization_params: dict | None = None,
        normalization_stats: dict | None = None,
        lead_swap: str | None = None,
    ):
        self.leads = leads
        self.resampled_frequency = resampled_frequency
        self.observed_length_seconds = observed_length_seconds
        self.normalization = normalization
        self.padding_mode = padding_mode
        self.crop_mode = crop_mode
        self.filter_cfg = filter_cfg or {"name": "none", "params": {}}
        self.normalization_params = normalization_params or {}
        self.normalization_stats = normalization_stats or {}
        self.lead_swap = lead_swap or "none"

    def __call__(self, signal_array: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Apply lead selection, resampling, resizing, filtering, and normalization."""

        signal_array = select_leads(signal_array, self.leads)
        signal_array = resample_signal(signal_array, int(sampling_rate), int(self.resampled_frequency))
        target_length = int(self.observed_length_seconds * self.resampled_frequency)
        signal_array = pad_or_crop(signal_array, target_length, mode=self.padding_mode, crop_mode=self.crop_mode)
        signal_array = apply_filter(signal_array, sampling_rate=int(self.resampled_frequency), filter_cfg=self.filter_cfg)
        if self.normalization == "z_norm":
            signal_array = z_norm(signal_array)
        elif self.normalization == "standardize":
            signal_array = standardize(
                signal_array,
                axis=self.normalization_params.get("axis", (-1, -2)),
            )
        elif self.normalization == "merl_minmax":
            signal_array = minmax_norm(signal_array, eps=self.normalization_params.get("eps", 1e-8))
        elif self.normalization == "global_mean_std":
            if "mean" not in self.normalization_stats or "std" not in self.normalization_stats:
                raise ValueError("global_mean_std normalization requires precomputed mean/std statistics.")
            signal_array = global_mean_std_norm(
                signal_array,
                mean=self.normalization_stats["mean"],
                std=self.normalization_stats["std"],
                clip_value=self.normalization_params.get("clip_value"),
            )
        elif self.normalization == "global_scalar_std":
            if "mean" not in self.normalization_stats or "std" not in self.normalization_stats:
                raise ValueError("global_scalar_std normalization requires precomputed mean/std statistics.")
            signal_array = global_scalar_std_norm(
                signal_array,
                mean=float(self.normalization_stats["mean"]),
                std=float(self.normalization_stats["std"]),
                clip_value=self.normalization_params.get("clip_value"),
            )
        elif self.normalization == "hubert_minmax":
            signal_array = minmax_scale_neg1_pos1(
                signal_array,
                eps=self.normalization_params.get("eps", 1e-8),
            )
        elif self.normalization not in (None, "none"):
            raise ValueError(f"Unsupported normalization '{self.normalization}'")
        if self.lead_swap == "avl_avf":
            signal_array = swap_avl_avf(signal_array)
        elif self.lead_swap not in (None, "none"):
            raise ValueError(f"Unsupported lead_swap '{self.lead_swap}'")
        return signal_array.astype(np.float32)
