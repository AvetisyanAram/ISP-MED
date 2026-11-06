from __future__ import annotations

from math import isfinite

import numpy as np
import torch

from classification.data.preprocess import (
    apply_filter,
    global_mean_std_norm,
    global_scalar_std_norm,
    minmax_norm,
    minmax_scale_neg1_pos1,
    standardize,
    z_norm,
)


def sliding_window_enabled(cfg: dict, stage: str) -> bool:
    """Return whether sliding-window inference is enabled for the given stage."""

    inference_cfg = dict(cfg.get("inference", {}))
    if inference_cfg.get("strategy", "single_crop") != "sliding_window":
        return False
    if stage == "validation":
        return bool(inference_cfg.get("enabled_on_validation", False))
    if stage == "prediction":
        return bool(inference_cfg.get("enabled_on_prediction", False))
    raise ValueError(f"Unsupported sliding-window stage '{stage}'")


def get_full_signal_length_seconds(cfg: dict) -> float:
    """Return the target full-record length used before window extraction."""

    value = cfg.get("inference", {}).get("full_signal_length_seconds")
    if value is None:
        raise ValueError("inference.full_signal_length_seconds must be set when sliding-window inference is enabled.")
    return float(value)


def _compute_window_starts(signal_length: int, window_size: int, num_windows: int) -> np.ndarray:
    max_start = max(signal_length - window_size, 0)
    if num_windows <= 1 or max_start == 0:
        return np.zeros(1, dtype=int)
    starts = np.rint(np.linspace(0, max_start, num=num_windows)).astype(int)
    starts[-1] = max_start
    return starts


def build_sliding_windows(signal_batch: torch.Tensor, window_size: int, num_windows: int) -> torch.Tensor:
    """Extract evenly spaced overlapping windows from a batched ECG tensor."""

    if signal_batch.ndim != 3:
        raise ValueError(f"Expected signal batch of shape (batch, channels, samples), got {tuple(signal_batch.shape)}")
    batch_size, channels, signal_length = signal_batch.shape
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if num_windows <= 0:
        raise ValueError("num_windows must be positive.")

    if signal_length < window_size:
        pad_width = window_size - signal_length
        signal_batch = torch.nn.functional.pad(signal_batch, (0, pad_width))
        signal_length = window_size

    starts = _compute_window_starts(signal_length, window_size, num_windows)
    windows = [signal_batch[:, :, start:start + window_size] for start in starts]
    return torch.stack(windows, dim=1).reshape(batch_size * len(starts), channels, window_size)


def build_strided_windows(signal_batch: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """Extract fixed-stride overlapping windows from a batched ECG tensor."""

    if signal_batch.ndim != 3:
        raise ValueError(f"Expected signal batch of shape (batch, channels, samples), got {tuple(signal_batch.shape)}")
    batch_size, channels, signal_length = signal_batch.shape
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    if signal_length < window_size:
        pad_width = window_size - signal_length
        signal_batch = torch.nn.functional.pad(signal_batch, (0, pad_width))
        signal_length = window_size

    starts = list(range(0, signal_length - window_size + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != signal_length - window_size:
        starts.append(signal_length - window_size)
    windows = [signal_batch[:, :, start:start + window_size] for start in starts]
    return torch.stack(windows, dim=1).reshape(batch_size * len(starts), channels, window_size)


def preprocess_windows(
    windows: torch.Tensor,
    filter_cfg: dict | None,
    sampling_rate: int,
    normalization: str | None,
    normalization_params: dict | None = None,
    normalization_stats: dict | None = None,
) -> torch.Tensor:
    """Apply per-window filtering and normalization after window extraction."""

    normalization_params = normalization_params or {}
    normalization_stats = normalization_stats or {}
    processed = []
    for window in windows:
        window_array = window.detach().cpu().numpy()
        window_array = apply_filter(window_array, sampling_rate=sampling_rate, filter_cfg=filter_cfg)
        if normalization in (None, "none"):
            processed_array = window_array
        elif normalization == "z_norm":
            processed_array = z_norm(window_array)
        elif normalization == "standardize":
            processed_array = standardize(window_array, axis=normalization_params.get("axis", (-1, -2)))
        elif normalization == "merl_minmax":
            processed_array = minmax_norm(window_array, eps=normalization_params.get("eps", 1e-8))
        elif normalization == "global_mean_std":
            if "mean" not in normalization_stats or "std" not in normalization_stats:
                raise ValueError("global_mean_std normalization requires precomputed mean/std statistics.")
            processed_array = global_mean_std_norm(
                window_array,
                mean=normalization_stats["mean"],
                std=normalization_stats["std"],
                clip_value=normalization_params.get("clip_value"),
            )
        elif normalization == "global_scalar_std":
            if "mean" not in normalization_stats or "std" not in normalization_stats:
                raise ValueError("global_scalar_std normalization requires precomputed mean/std statistics.")
            processed_array = global_scalar_std_norm(
                window_array,
                mean=float(normalization_stats["mean"]),
                std=float(normalization_stats["std"]),
                clip_value=normalization_params.get("clip_value"),
            )
        elif normalization == "hubert_minmax":
            processed_array = minmax_scale_neg1_pos1(
                window_array,
                eps=normalization_params.get("eps", 1e-8),
            )
        else:
            raise ValueError(f"Unsupported normalization '{normalization}' for sliding-window inference.")
        processed.append(torch.as_tensor(processed_array, dtype=windows.dtype))
    return torch.stack(processed, dim=0).to(windows.device)


def aggregate_probabilities(window_probabilities: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Aggregate per-window probabilities into a record-level probability."""

    if aggregation == "mean":
        return window_probabilities.mean(dim=1)
    if aggregation == "max":
        return window_probabilities.max(dim=1).values
    raise ValueError(f"Unsupported inference.sliding_window.aggregation '{aggregation}'")


def probabilities_to_logits(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert aggregated probabilities back to stable logits for BCEWithLogitsLoss."""

    clipped = probabilities.clamp(min=eps, max=1.0 - eps)
    return torch.log(clipped / (1.0 - clipped))


def run_model_with_optional_sliding_windows(
    model,
    signal_batch: torch.Tensor,
    cfg: dict,
    stage: str,
    preprocess_metadata: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a model on a batch, optionally using sliding-window inference."""

    if not sliding_window_enabled(cfg, stage):
        logits = model(signal_batch)
        probabilities = torch.sigmoid(logits)
        return logits, probabilities

    inference_cfg = dict(cfg["inference"])
    aggregation = inference_cfg["aggregation"]
    window_size = int(float(cfg["ecg"]["observed_length_seconds"]) * int(cfg["ecg"]["resampled_frequency"]))
    stride_seconds = inference_cfg.get("window_stride_seconds")
    if stride_seconds is not None:
        stride = int(float(stride_seconds) * int(cfg["ecg"]["resampled_frequency"]))
        windows = build_strided_windows(signal_batch, window_size=window_size, stride=stride)
    else:
        num_windows = int(inference_cfg["num_windows"])
        windows = build_sliding_windows(signal_batch, window_size=window_size, num_windows=num_windows)
    windows = preprocess_windows(
        windows,
        filter_cfg=dict(cfg["ecg"].get("filter", {})),
        sampling_rate=int(cfg["ecg"]["resampled_frequency"]),
        normalization=cfg["ecg"].get("normalization"),
        normalization_params=dict(cfg["ecg"].get("normalization_params", {})),
        normalization_stats=dict((preprocess_metadata or {}).get("normalization_stats", {})),
    )
    window_logits = model(windows)
    batch_size = signal_batch.shape[0]
    num_classes = window_logits.shape[1]
    num_windows = windows.shape[0] // batch_size
    window_probabilities = torch.sigmoid(window_logits).reshape(batch_size, num_windows, num_classes)
    aggregated_probabilities = aggregate_probabilities(window_probabilities, aggregation=aggregation)
    aggregated_logits = probabilities_to_logits(aggregated_probabilities)
    return aggregated_logits, aggregated_probabilities
