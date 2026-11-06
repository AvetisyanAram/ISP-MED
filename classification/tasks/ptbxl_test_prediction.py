from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification.data.datasets import ECGClassificationDataset, load_ptbxl_dataframe
from classification.data.preprocess import ECGPreprocessor
from classification.models import create_model
from .checkpoint import load_checkpoint
from .metrics import build_classification_report, compute_metrics, format_prediction_rows, sigmoid
from .windowing import get_full_signal_length_seconds, run_model_with_optional_sliding_windows, sliding_window_enabled


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
    candidates.append(Path.cwd().parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def _clone_state_dict_tensors(state_dict: dict[str, object]) -> dict[str, object]:
    cloned_state_dict: dict[str, object] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned_state_dict[key] = value.clone()
        else:
            cloned_state_dict[key] = value
    return cloned_state_dict


def _build_ptbxl_test_dataset(cfg: dict, checkpoint: dict) -> ECGClassificationDataset:
    checkpoint_cfg = dict(checkpoint["config"])
    preprocess_metadata = dict(checkpoint.get("preprocess", {}))
    dataset_cfg = dict(checkpoint_cfg["dataset"])

    override_dataset_cfg = cfg.get("dataset", {})
    if "signal_root" in override_dataset_cfg:
        dataset_cfg["signal_root"] = override_dataset_cfg["signal_root"]
    if "backend" in override_dataset_cfg:
        dataset_cfg["backend"] = override_dataset_cfg["backend"]
    if "label_column" in override_dataset_cfg:
        dataset_cfg["label_column"] = override_dataset_cfg["label_column"]
    if "split_files" in override_dataset_cfg and "test" in override_dataset_cfg["split_files"]:
        dataset_cfg["split_files"] = dict(dataset_cfg.get("split_files", {}))
        dataset_cfg["split_files"]["test"] = override_dataset_cfg["split_files"]["test"]

    test_frame = load_ptbxl_dataframe(_resolve_path(dataset_cfg["split_files"]["test"]))
    observed_length_seconds = float(checkpoint_cfg["ecg"]["observed_length_seconds"])
    normalization = checkpoint_cfg["ecg"]["normalization"]
    crop_mode = checkpoint_cfg["ecg"].get("crop_mode", "first")
    filter_cfg = dict(checkpoint_cfg["ecg"].get("filter", {}))
    if sliding_window_enabled(checkpoint_cfg, stage="prediction"):
        observed_length_seconds = get_full_signal_length_seconds(checkpoint_cfg)
        normalization = "none"
        crop_mode = "first"
        filter_cfg = {"name": "none", "params": {}}
    preprocessor = ECGPreprocessor(
        leads=list(checkpoint_cfg["ecg"]["leads"]),
        resampled_frequency=int(checkpoint_cfg["ecg"]["resampled_frequency"]),
        observed_length_seconds=observed_length_seconds,
        normalization=normalization,
        normalization_params=dict(checkpoint_cfg["ecg"].get("normalization_params", {})),
        crop_mode=crop_mode,
        filter_cfg=filter_cfg,
        normalization_stats=dict(preprocess_metadata.get("normalization_stats", {})),
        lead_swap=checkpoint_cfg["ecg"].get("lead_swap"),
    )
    return ECGClassificationDataset(
        dataframe=test_frame,
        signal_root=_resolve_path(dataset_cfg["signal_root"]),
        backend=dataset_cfg["backend"],
        pathology_names=list(checkpoint["pathology_names"]),
        merge_map={},
        label_column=dataset_cfg["label_column"],
        preprocessor=preprocessor,
    )


def run_ptbxl_test_prediction(cfg: dict) -> dict:
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Hydra runtime dir: {output_dir}")

    checkpoint = load_checkpoint(cfg["predict"]["checkpoint_path"], map_location="cpu")
    checkpoint_cfg = dict(checkpoint["config"])
    preprocess_metadata = dict(checkpoint.get("preprocess", {}))
    dataset = _build_ptbxl_test_dataset(cfg, checkpoint)
    dataframe = dataset.dataframe.copy()

    device_name = cfg["predict"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = create_model(
        model_name=checkpoint["model_name"],
        model_params=dict(checkpoint.get("model_params", {})),
        num_classes=len(checkpoint["pathology_names"]),
        input_channels=int(checkpoint["input_channels"]),
    )
    model.load_state_dict(_clone_state_dict_tensors(checkpoint["state_dict"]), assign=True)
    model = model.to(device)
    model.eval()

    num_workers = int(cfg["predict"]["num_workers"])
    pin_memory = bool(cfg["predict"].get("pin_memory", False))
    persistent_workers = bool(cfg["predict"].get("persistent_workers", False)) if num_workers > 0 else False
    prefetch_factor = cfg["predict"].get("prefetch_factor", 2)
    if num_workers == 0:
        prefetch_factor = None

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["predict"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    logits_batches = []
    record_ids = []
    raw_labels = []

    with torch.inference_mode():
        progress = tqdm(loader, total=len(loader), desc="Predict PTB-XL test", leave=False)
        for batch in progress:
            signal = batch["signal"].to(device, non_blocking=pin_memory and device.type == "cuda")
            logits, _ = run_model_with_optional_sliding_windows(
                model=model,
                signal_batch=signal,
                cfg=checkpoint_cfg,
                stage="prediction",
                preprocess_metadata=preprocess_metadata,
            )
            logits_batches.append(logits.detach().cpu().numpy())
            record_ids.extend(batch["record_id"])
            raw_labels.extend(batch["raw_labels"])

    logits = np.concatenate(logits_batches, axis=0)
    probabilities = sigmoid(logits)
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)
    predictions = (probabilities > thresholds).astype(np.float32)

    metrics, confusion = compute_metrics(
        dataset.targets,
        predictions,
        probabilities,
        dataset.target_labels,
    )
    summary = metrics.mean(axis=1).to_frame(name="value")
    summary.index.name = "metric"

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(metrics_dir / "test_per_pathology.csv")
    confusion.to_csv(metrics_dir / "test_confusion_matrix.csv")
    summary.to_csv(metrics_dir / "test_summary.csv")
    (reports_dir / "test_classification_report.txt").write_text(
        build_classification_report(dataset.targets, predictions)
    )

    prediction_rows = format_prediction_rows(
        record_ids=record_ids,
        true_targets=dataset.targets,
        predicted_targets=predictions,
        probabilities=probabilities,
        label_names=dataset.target_labels,
        raw_labels=raw_labels,
    )
    metadata_by_id = dataframe.copy()
    metadata_by_id["record_id"] = metadata_by_id["record_id"].astype(str)
    metadata_by_id = metadata_by_id.set_index("record_id")
    prediction_rows["patient_id"] = [
        metadata_by_id.loc[str(record_id), "patient_id"] if str(record_id) in metadata_by_id.index else ""
        for record_id in record_ids
    ]
    prediction_rows["checkpoint_raw_labels"] = [json.dumps(labels) for labels in raw_labels]
    predictions_path = output_dir / "ptbxl_test_predictions.csv"
    prediction_rows.to_csv(predictions_path, index=False)

    summary_payload = {
        "checkpoint_path": str(cfg["predict"]["checkpoint_path"]),
        "target_labels": dataset.target_labels,
        "metrics_dir": str(metrics_dir),
        "predictions_path": str(predictions_path),
    }
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary_payload, indent=2))
    return summary_payload
