from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification.data.datasets import ECGClassificationDataset, load_isp_dataframe
from classification.data.labels import build_target_matrix, get_pathology_preset, parse_label_list
from classification.data.preprocess import ECGPreprocessor
from classification.models import create_model
from .checkpoint import load_checkpoint
from .metrics import build_classification_report, compute_metrics, format_prediction_rows, sigmoid
from .windowing import get_full_signal_length_seconds, run_model_with_optional_sliding_windows, sliding_window_enabled


DOCTOR_COLUMNS = [f"doctor{i}" for i in range(1, 8)]
SUMMARY_METRICS = [
    "Specificity",
    "Sensitivity",
    "ROC-AUC",
    "Precision_PPV_",
    "NPV",
    "G-mean",
    "fbeta2-score",
    "f1-score",
    "AP",
]


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
    """Clone tensor values before loading a checkpoint state dict into a fresh model."""

    cloned_state_dict: dict[str, object] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned_state_dict[key] = value.clone()
        else:
            cloned_state_dict[key] = value
    return cloned_state_dict


def _build_isp_dataset(cfg: dict, checkpoint: dict) -> ECGClassificationDataset:
    dataset_cfg = cfg["dataset"]
    map_file = _resolve_path(dataset_cfg["map_file"])
    signal_root = _resolve_path(dataset_cfg["signal_root"])
    frame = load_isp_dataframe(map_file, signal_root)
    checkpoint_cfg = dict(checkpoint["config"])
    preprocess_metadata = dict(checkpoint.get("preprocess", {}))
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
        dataframe=frame,
        signal_root=signal_root,
        backend=dataset_cfg["backend"],
        pathology_names=list(checkpoint["pathology_names"]),
        merge_map={},
        label_column=dataset_cfg["label_column"],
        preprocessor=preprocessor,
    )


def _resolve_target_labels(cfg: dict, checkpoint: dict) -> list[str]:
    explicit_labels = list(cfg.get("target", {}).get("pathology_names", []))
    preset_cfg = cfg.get("pathology_preset")

    if explicit_labels and preset_cfg is not None:
        raise ValueError("Use either target.pathology_names or pathology_preset, not both.")

    if explicit_labels:
        target_labels = explicit_labels
    elif preset_cfg is not None:
        target_labels = get_pathology_preset(preset_cfg["name"])["target_labels"]
    else:
        target_labels = list(checkpoint["pathology_names"])

    checkpoint_labels = list(checkpoint["pathology_names"])
    missing_in_checkpoint = [label for label in target_labels if label not in checkpoint_labels]
    if missing_in_checkpoint:
        raise ValueError(
            "Target labels are not present in checkpoint outputs: "
            + ", ".join(missing_in_checkpoint)
        )
    return target_labels


def _resolve_annotation_sources(cfg: dict, dataframe: pd.DataFrame) -> list[str]:
    requested_sources = list(cfg.get("target", {}).get("annotation_sources", []))
    if not requested_sources:
        requested_sources = ["unified_decision"]

    missing_sources = [column for column in requested_sources if column not in dataframe.columns]
    if missing_sources:
        raise ValueError(
            "Requested annotation sources are missing from ISP map_file.csv: "
            + ", ".join(missing_sources)
        )
    return requested_sources


def _evaluate_source(
    dataframe: pd.DataFrame,
    source_column: str,
    target_labels: list[str],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    row_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if row_mask is None:
        row_mask = np.ones(len(dataframe), dtype=bool)
    source_frame = dataframe.loc[row_mask]
    source_targets = build_target_matrix(
        source_frame[source_column],
        pathology_names=target_labels,
        merge_map={},
    )
    metrics, confusion = compute_metrics(
        source_targets,
        predictions[row_mask],
        probabilities[row_mask],
        target_labels,
    )
    summary = metrics.mean(axis=1, skipna=True).to_frame(name="value")
    summary.index.name = "metric"
    return metrics, confusion, summary


def _labels_to_matrix(label_series: pd.Series, label_names: list[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(label_names)}
    targets = np.zeros((len(label_series), len(label_names)), dtype=np.float32)
    for row_index, value in enumerate(label_series):
        for label in parse_label_list(value):
            column_index = label_to_index.get(label)
            if column_index is not None:
                targets[row_index, column_index] = 1.0
    return targets


def _summary_from_targets(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> pd.Series:
    metrics, _ = compute_metrics(targets, predictions, probabilities, label_names)
    summary = metrics.mean(axis=1, skipna=True)
    return summary.reindex(SUMMARY_METRICS)


def _doctor_available_mask(dataframe: pd.DataFrame, doctor: str) -> np.ndarray:
    return dataframe[doctor].map(lambda value: len(parse_label_list(value)) > 0).to_numpy(dtype=bool)


def _available_doctors_for_row(row: pd.Series, doctor_sources: list[str]) -> tuple[str, ...]:
    return tuple(doctor for doctor in doctor_sources if len(parse_label_list(row[doctor])) > 0)


def _finite_weighted_mean(values: list[float], weights: list[int]) -> float:
    finite_pairs = [
        (float(value), int(weight))
        for value, weight in zip(values, weights)
        if np.isfinite(float(value)) and int(weight) > 0
    ]
    if not finite_pairs:
        return float("nan")
    numerator = sum(value * weight for value, weight in finite_pairs)
    denominator = sum(weight for _, weight in finite_pairs)
    return numerator / denominator


def build_doctors_vs_consensus_summary(
    dataframe: pd.DataFrame,
    target_labels: list[str],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    doctor_sources: list[str],
    row_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    if row_mask is not None:
        dataframe = dataframe.loc[row_mask].reset_index(drop=True)
        predictions = predictions[row_mask]
        probabilities = probabilities[row_mask]
    rows: list[dict[str, object]] = []
    consensus_targets = build_target_matrix(
        dataframe["unified_decision"],
        pathology_names=target_labels,
        merge_map={},
    )

    for doctor in doctor_sources:
        mask = _doctor_available_mask(dataframe, doctor)
        if not np.any(mask):
            continue

        doctor_targets = build_target_matrix(
            dataframe.loc[mask, doctor],
            pathology_names=target_labels,
            merge_map={},
        )
        doctor_summary = _summary_from_targets(
            doctor_targets,
            predictions[mask],
            probabilities[mask],
            target_labels,
        )
        matched_consensus_summary = _summary_from_targets(
            consensus_targets[mask],
            predictions[mask],
            probabilities[mask],
            target_labels,
        )

        for metric_name in SUMMARY_METRICS:
            doctor_value = float(doctor_summary.loc[metric_name])
            consensus_value = float(matched_consensus_summary.loc[metric_name])
            rows.append(
                {
                    "doctor": doctor,
                    "n_records": int(mask.sum()),
                    "metric": metric_name,
                    "Doctor": doctor_value,
                    "Matched Consensus": consensus_value,
                    "Doctor - Consensus": doctor_value - consensus_value,
                }
            )

    return pd.DataFrame(rows)


def build_panel_weighted_summary(
    dataframe: pd.DataFrame,
    target_labels: list[str],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    doctor_sources: list[str],
    row_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if row_mask is not None:
        dataframe = dataframe.loc[row_mask].reset_index(drop=True)
        predictions = predictions[row_mask]
        probabilities = probabilities[row_mask]
    consensus_targets = build_target_matrix(
        dataframe["unified_decision"],
        pathology_names=target_labels,
        merge_map={},
    )
    panel_keys = dataframe.apply(lambda row: _available_doctors_for_row(row, doctor_sources), axis=1)
    detail_rows: list[dict[str, object]] = []

    for panel, indices in sorted(panel_keys.groupby(panel_keys).groups.items(), key=lambda item: (len(item[0]), item[0])):
        if not panel:
            continue
        row_indices = np.asarray(list(indices), dtype=int)
        n_records = int(row_indices.size)
        consensus_summary = _summary_from_targets(
            consensus_targets[row_indices],
            predictions[row_indices],
            probabilities[row_indices],
            target_labels,
        )
        doctor_summaries = []
        for doctor in panel:
            doctor_targets = build_target_matrix(
                dataframe.iloc[row_indices][doctor],
                pathology_names=target_labels,
                merge_map={},
            )
            doctor_summaries.append(
                _summary_from_targets(
                    doctor_targets,
                    predictions[row_indices],
                    probabilities[row_indices],
                    target_labels,
                )
            )

        for metric_name in SUMMARY_METRICS:
            doctor_values = [float(summary.loc[metric_name]) for summary in doctor_summaries]
            finite_doctor_values = [value for value in doctor_values if np.isfinite(value)]
            doctor_mean = float(np.mean(finite_doctor_values)) if finite_doctor_values else float("nan")
            highest_expert = float(np.max(finite_doctor_values)) if finite_doctor_values else float("nan")
            detail_rows.append(
                {
                    "panel": "+".join(panel),
                    "n_records": n_records,
                    "metric": metric_name,
                    "Consensus": float(consensus_summary.loc[metric_name]),
                    "Doctor Mean": doctor_mean,
                    "Highest Expert": highest_expert,
                }
            )

    details = pd.DataFrame(detail_rows)
    if details.empty:
        summary = pd.DataFrame(
            [
                {
                    "metric": metric_name,
                    "Consensus": float("nan"),
                    "Doctor Mean": float("nan"),
                    "Highest Expert": float("nan"),
                    "Panel Count": 0,
                    "Record Count": 0,
                }
                for metric_name in SUMMARY_METRICS
            ]
        ).set_index("metric")
        return summary, details

    summary_rows = []
    for metric_name in SUMMARY_METRICS:
        metric_details = details.loc[details["metric"] == metric_name]
        weights = metric_details["n_records"].astype(int).tolist()
        summary_rows.append(
            {
                "metric": metric_name,
                "Consensus": _finite_weighted_mean(metric_details["Consensus"].tolist(), weights),
                "Doctor Mean": _finite_weighted_mean(metric_details["Doctor Mean"].tolist(), weights),
                "Highest Expert": _finite_weighted_mean(metric_details["Highest Expert"].tolist(), weights),
                "Panel Count": int(metric_details["panel"].nunique()),
                "Record Count": int(metric_details.drop_duplicates("panel")["n_records"].sum()),
            }
        )
    return pd.DataFrame(summary_rows).set_index("metric"), details


def _mean_metrics_by_doctor(per_source_metrics: dict[str, pd.DataFrame], doctor_sources: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not doctor_sources:
        return pd.DataFrame(), pd.DataFrame()

    stacked_metrics = np.stack([per_source_metrics[source].to_numpy(dtype=float) for source in doctor_sources], axis=0)
    mean_per_pathology = pd.DataFrame(
        np.nanmean(stacked_metrics, axis=0),
        index=per_source_metrics[doctor_sources[0]].index,
        columns=per_source_metrics[doctor_sources[0]].columns,
    ).round(4)
    mean_summary = mean_per_pathology.mean(axis=1, skipna=True).to_frame(name="value")
    mean_summary.index.name = "metric"
    return mean_per_pathology, mean_summary


def _build_prediction_rows(
    dataframe: pd.DataFrame,
    record_ids: list[str],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    checkpoint_label_names: list[str],
    target_labels: list[str],
    annotation_sources: list[str],
) -> pd.DataFrame:
    base_rows = format_prediction_rows(
        record_ids=record_ids,
        true_targets=np.zeros((len(record_ids), len(checkpoint_label_names)), dtype=np.float32),
        predicted_targets=predictions,
        probabilities=probabilities,
        label_names=checkpoint_label_names,
        raw_labels=[[] for _ in record_ids],
    )
    frame_by_id = dataframe.copy()
    frame_by_id["record_id"] = frame_by_id["record_id"].astype(str)
    frame_by_id = frame_by_id.set_index("record_id")

    for source in annotation_sources:
        raw_column = f"{source}_labels"
        target_column = f"{source}_target_labels"
        source_lists = [parse_label_list(frame_by_id.loc[str(record_id), source]) for record_id in record_ids]
        base_rows[raw_column] = [json.dumps(labels) for labels in source_lists]
        base_rows[target_column] = [
            json.dumps([label for label in target_labels if label in labels])
            for labels in source_lists
        ]

    return base_rows


def run_prediction(cfg: dict) -> dict:
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Hydra runtime dir: {output_dir}")

    checkpoint = load_checkpoint(cfg["predict"]["checkpoint_path"], map_location="cpu")
    checkpoint_cfg = dict(checkpoint["config"])
    preprocess_metadata = dict(checkpoint.get("preprocess", {}))
    dataset = _build_isp_dataset(cfg, checkpoint)
    dataframe = dataset.dataframe.copy()

    target_labels = _resolve_target_labels(cfg, checkpoint)
    checkpoint_label_names = list(checkpoint["pathology_names"])
    target_indices = [checkpoint_label_names.index(label) for label in target_labels]
    annotation_sources = _resolve_annotation_sources(cfg, dataframe)
    doctor_sources = [source for source in annotation_sources if source in DOCTOR_COLUMNS]

    print("Prediction target labels:")
    for label in target_labels:
        print(f"- {label}")
    print("Annotation sources:")
    for source in annotation_sources:
        print(f"- {source}")

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
        num_classes=len(checkpoint_label_names),
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
        progress = tqdm(loader, total=len(loader), desc="Predict", leave=False)
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

    target_probabilities = probabilities[:, target_indices]
    target_predictions = predictions[:, target_indices]

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    per_source_metrics: dict[str, pd.DataFrame] = {}
    per_source_metric_rows = []
    per_source_summary_rows = []

    for source in annotation_sources:
        row_mask = _doctor_available_mask(dataframe, source) if source in DOCTOR_COLUMNS else None
        metrics, confusion, summary = _evaluate_source(
            dataframe=dataframe,
            source_column=source,
            target_labels=target_labels,
            probabilities=target_probabilities,
            predictions=target_predictions,
            row_mask=row_mask,
        )
        source_frame = dataframe if row_mask is None else dataframe.loc[row_mask]
        source_targets = build_target_matrix(source_frame[source], pathology_names=target_labels, merge_map={})
        source_predictions = target_predictions if row_mask is None else target_predictions[row_mask]
        per_source_metrics[source] = metrics

        metrics.to_csv(metrics_dir / f"{source}_per_pathology.csv")
        confusion.to_csv(metrics_dir / f"{source}_confusion_matrix.csv")
        summary.to_csv(metrics_dir / f"{source}_summary.csv")
        (reports_dir / f"{source}_classification_report.txt").write_text(
            build_classification_report(source_targets, source_predictions)
        )

        for metric_name, value in summary["value"].items():
            per_source_summary_rows.append(
                {
                    "annotation_source": source,
                    "metric": metric_name,
                    "value": float(value),
                }
            )
        for metric_name, row in metrics.iterrows():
            for pathology_name, value in row.items():
                per_source_metric_rows.append(
                    {
                        "annotation_source": source,
                        "metric": metric_name,
                        "pathology": pathology_name,
                        "value": float(value),
                    }
                )

    all_sources_summary = pd.DataFrame(per_source_summary_rows)
    if not all_sources_summary.empty:
        all_sources_summary.to_csv(metrics_dir / "all_sources_summary.csv", index=False)
    all_sources_per_pathology = pd.DataFrame(per_source_metric_rows)
    if not all_sources_per_pathology.empty:
        all_sources_per_pathology.to_csv(metrics_dir / "all_sources_per_pathology.csv", index=False)

    mean_doctor_metrics, mean_doctor_summary = _mean_metrics_by_doctor(per_source_metrics, doctor_sources)
    if not mean_doctor_metrics.empty:
        mean_doctor_metrics.to_csv(metrics_dir / "doctors_mean_per_pathology.csv")
        mean_doctor_summary.to_csv(metrics_dir / "doctors_mean_summary.csv")

    if doctor_sources:
        doctors_vs_consensus = build_doctors_vs_consensus_summary(
            dataframe=dataframe,
            target_labels=target_labels,
            predictions=target_predictions,
            probabilities=target_probabilities,
            doctor_sources=doctor_sources,
        )
        if not doctors_vs_consensus.empty:
            doctors_vs_consensus.to_csv(metrics_dir / "doctors_vs_consensus_summary.csv", index=False)

        panel_weighted_summary, panel_weighted_details = build_panel_weighted_summary(
            dataframe=dataframe,
            target_labels=target_labels,
            predictions=target_predictions,
            probabilities=target_probabilities,
            doctor_sources=doctor_sources,
        )
        panel_weighted_summary.to_csv(metrics_dir / "panel_weighted_summary.csv")
        panel_weighted_details.to_csv(metrics_dir / "panel_weighted_details.csv", index=False)

    prediction_rows = _build_prediction_rows(
        dataframe=dataframe,
        record_ids=record_ids,
        probabilities=probabilities,
        predictions=predictions,
        checkpoint_label_names=checkpoint_label_names,
        target_labels=target_labels,
        annotation_sources=annotation_sources,
    )
    prediction_rows["checkpoint_raw_labels"] = [json.dumps(labels) for labels in raw_labels]
    prediction_rows.to_csv(output_dir / "isp_predictions.csv", index=False)

    summary_payload = {
        "checkpoint_path": str(cfg["predict"]["checkpoint_path"]),
        "target_labels": target_labels,
        "annotation_sources": annotation_sources,
        "doctor_sources": doctor_sources,
        "metrics_dir": str(metrics_dir),
        "predictions_path": str(output_dir / "isp_predictions.csv"),
    }
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary_payload, indent=2))

    return summary_payload
