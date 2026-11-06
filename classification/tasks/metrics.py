from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, f1_score, fbeta_score, roc_auc_score


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def get_confusion_matrix(target: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int, int]:
    target = np.asarray(target, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    tp = int(np.sum((target == 1) & (prediction == 1)))
    tn = int(np.sum((target == 0) & (prediction == 0)))
    fp = int(np.sum((target == 0) & (prediction == 1)))
    fn = int(np.sum((target == 1) & (prediction == 0)))
    return tn, fp, fn, tp


def compute_metrics(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, label_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame(
        columns=label_names,
        index=["Specificity", "Sensitivity", "G-mean", "f1-score", "fbeta2-score", "ROC-AUC", "AP", "Precision_PPV_", "NPV"],
    )
    confusion = pd.DataFrame(columns=["TN", "FP", "FN", "TP"], index=label_names)
    for index, label_name in enumerate(label_names):
        tn, fp, fn, tp = get_confusion_matrix(targets[:, index], predictions[:, index])
        target_column = targets[:, index]
        probability_column = probabilities[:, index]
        with np.errstate(all="ignore"):
            specificity = np.divide(tn, tn + fp)
            sensitivity = np.divide(tp, tp + fn)
            gmean = np.sqrt(specificity * sensitivity)
            f1 = f1_score(target_column, predictions[:, index], zero_division=0)
            fbeta = fbeta_score(target_column, predictions[:, index], beta=2, zero_division=0)
            if len(np.unique(target_column)) == 2:
                roc_auc = roc_auc_score(target_column, probability_column)
            else:
                roc_auc = np.nan
            if np.any(target_column == 1):
                ap = average_precision_score(target_column, probability_column)
            else:
                ap = np.nan
            precision = np.divide(tp, tp + fp)
            npv = np.divide(tn, tn + fn)
        metrics.loc[:, label_name] = [specificity, sensitivity, gmean, f1, fbeta, roc_auc, ap, precision, npv]
        confusion.loc[label_name] = [tn, fp, fn, tp]
    return metrics.astype(float).round(4), confusion.astype(int)


def _safe_nanmean(values: np.ndarray) -> float:
    """Return the mean of finite values or `nan` when none are available."""

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(finite_values.mean())


def compute_aggregated_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    """Compute macro and micro validation metrics for multilabel ECG classification."""

    metrics, confusion = compute_metrics(targets, predictions, probabilities, label_names)

    macro_metrics = {
        "Specificity": float(metrics.loc["Specificity"].mean()),
        "Sensitivity": float(metrics.loc["Sensitivity"].mean()),
        "G-mean": float(metrics.loc["G-mean"].mean()),
        "ROC-AUC": _safe_nanmean(metrics.loc["ROC-AUC"].to_numpy(dtype=float)),
        "f1-score": float(metrics.loc["f1-score"].mean()),
    }

    total_tn = int(confusion["TN"].sum())
    total_fp = int(confusion["FP"].sum())
    total_fn = int(confusion["FN"].sum())
    total_tp = int(confusion["TP"].sum())

    micro_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) else np.nan
    micro_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else np.nan
    micro_gmean = np.sqrt(micro_specificity * micro_sensitivity)
    flat_targets = np.asarray(targets, dtype=int).reshape(-1)
    flat_predictions = np.asarray(predictions, dtype=int).reshape(-1)
    flat_probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(np.unique(flat_targets)) == 2:
        micro_roc_auc = float(roc_auc_score(flat_targets, flat_probabilities))
    else:
        micro_roc_auc = np.nan
    micro_f1 = float(f1_score(flat_targets, flat_predictions, zero_division=0))

    micro_metrics = {
        "Specificity": float(micro_specificity),
        "Sensitivity": float(micro_sensitivity),
        "G-mean": float(micro_gmean),
        "ROC-AUC": float(micro_roc_auc),
        "f1-score": float(micro_f1),
    }

    return pd.DataFrame({"macro": macro_metrics, "micro": micro_metrics}).round(4)


def select_best_validation_threshold(targets: np.ndarray, logits: np.ndarray, metric_weights: dict[str, float]) -> np.ndarray:
    probabilities = sigmoid(logits)
    thresholds = np.arange(-0.01, 0.9, 0.06)
    best_thresholds = []
    for class_index in range(probabilities.shape[1]):
        class_targets = targets[:, class_index].astype(float)
        class_probabilities = probabilities[:, class_index]
        candidates = []
        for threshold in thresholds:
            class_predictions = (class_probabilities > threshold).astype(float)
            tn, fp, fn, tp = get_confusion_matrix(class_targets, class_predictions)
            specificity = np.divide(tn, tn + fp) if (tn + fp) else 0.0
            sensitivity = np.divide(tp, tp + fn) if (tp + fn) else 0.0
            precision = np.divide(tp, tp + fp) if (tp + fp) else 0.0
            components = {
                "gmean": np.sqrt(specificity * sensitivity),
                "f1-score": (2 * sensitivity * precision / (sensitivity + precision)) if (sensitivity + precision) else 0.0,
                "ROC-AUC": (sensitivity + specificity) / 2.0,
                "specificity": specificity,
                "sensitivity": sensitivity,
            }
            score = sum(metric_weights[name] * components[name] for name in metric_weights)
            candidates.append((score, threshold))
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_thresholds.append(candidates[0][1])
    return np.asarray(best_thresholds, dtype=np.float32)


def format_prediction_rows(
    record_ids: list[str],
    true_targets: np.ndarray,
    predicted_targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    raw_labels: list[list[str]],
) -> pd.DataFrame:
    rows = []
    for record_id, target_row, prediction_row, probability_row, raw_label_row in zip(
        record_ids,
        true_targets,
        predicted_targets,
        probabilities,
        raw_labels,
    ):
        target_labels = [label for label, value in zip(label_names, target_row) if value >= 0.5]
        predicted_labels = [label for label, value in zip(label_names, prediction_row) if value >= 0.5]
        probability_map = {label: float(probability) for label, probability in zip(label_names, probability_row)}
        rows.append(
            {
                "record_id": record_id,
                "raw_labels": json.dumps(raw_label_row),
                "true_labels": json.dumps(target_labels),
                "predicted_labels": json.dumps(predicted_labels),
                "probabilities": json.dumps(probability_map),
            }
        )
    return pd.DataFrame(rows)


def build_classification_report(targets: np.ndarray, predictions: np.ndarray) -> str:
    return classification_report(targets, predictions, zero_division=0)
