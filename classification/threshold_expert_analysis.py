from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classification.data.labels import build_target_matrix, parse_label_list
from classification.tasks.metrics import compute_metrics


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_SOURCE_ROOT = PACKAGE_ROOT / "outputs" / "classification_log"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "threshold_log"
DEFAULT_FINAL_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "threshold_analysis"
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
DEFAULT_FINAL_METRICS = ["G-mean", "f1-score", "fbeta2-score"]
THRESHOLD_OPTIMIZATION_METRICS = {
    "Specificity",
    "Sensitivity",
    "G-mean",
    "f1-score",
    "fbeta2-score",
    "Precision_PPV_",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune doctor-specific pathology thresholds from final_classification_log ISP predictions "
            "and compare doctor-vs-consensus metrics on held-out test splits."
        )
    )
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--experts", nargs="+", default=DOCTOR_COLUMNS)
    parser.add_argument("--runs", nargs="*", default=None, help="Optional run names, e.g. run_01 run_02")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--val-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--optimization-metric",
        default="G-mean",
        choices=sorted(THRESHOLD_OPTIMIZATION_METRICS),
        help="Threshold tuning objective on the doctor-specific validation subset.",
    )
    parser.add_argument(
        "--threshold-grid-step",
        type=float,
        default=0.01,
        help="Grid step for threshold search over [0, 1].",
    )
    parser.add_argument(
        "--min-test-fraction",
        type=float,
        default=0.6,
        help="Safety check for held-out test size. Must be satisfied by 1 - val_fraction.",
    )
    parser.add_argument(
        "--final-output-root",
        default=None,
        help=(
            "Output directory for aggregated ranked threshold-analysis tables. "
            "Defaults to a sibling final_threshold_analysis directory next to --output-root."
        ),
    )
    parser.add_argument(
        "--final-metrics",
        nargs="+",
        default=DEFAULT_FINAL_METRICS,
        help="Metrics exported into the aggregated ranked threshold-analysis tables.",
    )
    parser.add_argument(
        "--final-sort-by",
        default="tradeoff_score",
        choices=[
            "tradeoff_score",
            "doctor_gain",
            "consensus_delta",
            "abs_consensus_delta",
        ],
        help="Ranking key for the aggregated final threshold-analysis tables.",
    )
    parser.add_argument(
        "--final-min-positive-support",
        type=float,
        default=0.0,
        help="Optional filter on both doctor and consensus mean positive support in final aggregated tables.",
    )
    parser.add_argument(
        "--final-doctor-gain-min",
        type=float,
        default=None,
        help="Optional lower bound on doctor_gain in final aggregated tables.",
    )
    parser.add_argument(
        "--final-consensus-delta-max",
        type=float,
        default=None,
        help="Optional upper bound on consensus_delta in final aggregated tables.",
    )
    return parser.parse_args()


def _parse_probability_map(value: object) -> dict[str, float]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected probability JSON object, got {value!r}")
    return {str(key): float(item) for key, item in parsed.items()}


def _load_prediction_payload(predictions_path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    frame = pd.read_csv(predictions_path)
    if frame.empty:
        raise RuntimeError(f"Empty predictions file: {predictions_path}")
    label_names = list(_parse_probability_map(frame.loc[0, "probabilities"]))
    probabilities = np.asarray(
        [[_parse_probability_map(value)[label] for label in label_names] for value in frame["probabilities"]],
        dtype=np.float32,
    )
    frame = frame.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    return frame, label_names, probabilities


def _available_mask(frame: pd.DataFrame, expert_column: str) -> np.ndarray:
    return frame[expert_column].map(lambda value: len(parse_label_list(value)) > 0).to_numpy(dtype=bool)


def split_doctor_records(
    frame: pd.DataFrame,
    expert_column: str,
    label_names: list[str],
    seed: int,
    val_fraction: float,
) -> dict[str, list[str]]:
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("Expected val_fraction in (0, 1).")

    available_frame = frame.loc[_available_mask(frame, expert_column), ["record_id", expert_column]].copy()
    available_frame["record_id"] = available_frame["record_id"].astype(str)
    if len(available_frame) < 2:
        raise RuntimeError(f"Not enough records with labels for {expert_column}: {len(available_frame)}")

    target_matrix = build_target_matrix(available_frame[expert_column], pathology_names=label_names, merge_map={})
    zero_target_mask = target_matrix.sum(axis=1) == 0
    if np.any(zero_target_mask):
        target_matrix = np.concatenate([target_matrix, zero_target_mask[:, None].astype(np.float32)], axis=1)

    rng = np.random.default_rng(seed)
    val_size = int(round(val_fraction * len(available_frame)))
    val_size = min(max(1, val_size), len(available_frame) - 1)
    test_size = len(available_frame) - val_size

    desired_sizes = np.asarray([val_size, test_size], dtype=int)
    label_totals = target_matrix.sum(axis=0).astype(np.float64)
    desired_label_counts = np.outer(desired_sizes / float(len(available_frame)), label_totals)
    assigned_indices: list[list[int]] = [[], []]
    remaining_capacity = desired_sizes.copy()
    unassigned = set(range(len(available_frame)))

    def _choose_split(sample_targets: np.ndarray, focus_label: int | None = None) -> int:
        candidate_splits = [index for index, capacity in enumerate(remaining_capacity.tolist()) if capacity > 0]
        if not candidate_splits:
            raise RuntimeError("No remaining split capacity while building val/test split.")
        best_split = candidate_splits[0]
        best_score: tuple[float, float, float] | None = None
        for split_index in candidate_splits:
            if focus_label is not None:
                label_need = float(desired_label_counts[split_index, focus_label])
            else:
                label_need = float(np.dot(sample_targets, np.maximum(desired_label_counts[split_index], 0.0)))
            relative_fill = float(remaining_capacity[split_index] / desired_sizes[split_index])
            score = (label_need, relative_fill, float(remaining_capacity[split_index]))
            if best_score is None or score > best_score:
                best_score = score
                best_split = split_index
        return best_split

    while unassigned:
        remaining_label_counts = target_matrix[list(unassigned)].sum(axis=0)
        positive_labels = np.flatnonzero(remaining_label_counts > 0)
        if positive_labels.size == 0:
            break
        min_count = float(np.min(remaining_label_counts[positive_labels]))
        candidate_labels = [int(index) for index in positive_labels if np.isclose(remaining_label_counts[index], min_count)]
        focus_label = candidate_labels[int(rng.integers(len(candidate_labels)))]
        candidate_samples = [sample_index for sample_index in unassigned if target_matrix[sample_index, focus_label] > 0]
        rng.shuffle(candidate_samples)
        candidate_samples.sort(key=lambda sample_index: -int(target_matrix[sample_index].sum()))
        for sample_index in candidate_samples:
            if sample_index not in unassigned:
                continue
            sample_targets = target_matrix[sample_index]
            split_index = _choose_split(sample_targets, focus_label=focus_label)
            assigned_indices[split_index].append(sample_index)
            remaining_capacity[split_index] -= 1
            desired_label_counts[split_index] -= sample_targets
            unassigned.remove(sample_index)

    if unassigned:
        leftover = list(unassigned)
        rng.shuffle(leftover)
        leftover.sort(key=lambda sample_index: -int(target_matrix[sample_index].sum()))
        for sample_index in leftover:
            sample_targets = target_matrix[sample_index]
            split_index = _choose_split(sample_targets, focus_label=None)
            assigned_indices[split_index].append(sample_index)
            remaining_capacity[split_index] -= 1
            desired_label_counts[split_index] -= sample_targets
            unassigned.remove(sample_index)

    record_ids = available_frame["record_id"].to_numpy()
    val_ids = sorted(record_ids[assigned_indices[0]].tolist())
    test_ids = sorted(record_ids[assigned_indices[1]].tolist())
    return {"val": val_ids, "test": test_ids}


def _subset_by_record_ids(frame: pd.DataFrame, record_ids: list[str]) -> pd.DataFrame:
    record_id_set = set(str(record_id) for record_id in record_ids)
    subset = frame.loc[frame["record_id"].astype(str).isin(record_id_set)].copy()
    subset["record_id"] = subset["record_id"].astype(str)
    return subset.sort_values("record_id").reset_index(drop=True)


def _subset_probabilities(frame: pd.DataFrame, probabilities: np.ndarray, record_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    subset = _subset_by_record_ids(frame, record_ids)
    frame_by_id = frame.copy().set_index("record_id")
    row_indices = [frame_by_id.index.get_loc(record_id) for record_id in subset["record_id"].tolist()]
    return subset, probabilities[np.asarray(row_indices, dtype=int)]


def _confusion_components(targets: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    target_array = np.asarray(targets, dtype=int)
    prediction_array = np.asarray(predictions, dtype=int)
    tp = int(np.sum((target_array == 1) & (prediction_array == 1)))
    tn = int(np.sum((target_array == 0) & (prediction_array == 0)))
    fp = int(np.sum((target_array == 0) & (prediction_array == 1)))
    fn = int(np.sum((target_array == 1) & (prediction_array == 0)))
    return tn, fp, fn, tp


def _threshold_objective(metric_name: str, targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    predictions = (probabilities > threshold).astype(int)
    tn, fp, fn, tp = _confusion_components(targets, predictions)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    if metric_name == "Specificity":
        return float(specificity)
    if metric_name == "Sensitivity":
        return float(sensitivity)
    if metric_name == "G-mean":
        return float(np.sqrt(specificity * sensitivity))
    if metric_name == "f1-score":
        return float((2 * sensitivity * precision / (sensitivity + precision)) if (sensitivity + precision) else 0.0)
    if metric_name == "fbeta2-score":
        beta_sq = 4.0
        denominator = beta_sq * precision + sensitivity
        return float(((1.0 + beta_sq) * precision * sensitivity / denominator) if denominator else 0.0)
    if metric_name == "Precision_PPV_":
        return float(precision)
    raise ValueError(f"Unsupported threshold optimization metric '{metric_name}'")


def select_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    optimization_metric: str,
    grid_step: float,
    fallback_thresholds: np.ndarray,
) -> np.ndarray:
    thresholds = np.arange(0.0, 1.0 + 1e-9, grid_step, dtype=np.float32)
    best_thresholds: list[float] = []
    for class_index in range(probabilities.shape[1]):
        class_targets = targets[:, class_index].astype(int)
        class_probabilities = probabilities[:, class_index].astype(float)
        unique_targets = np.unique(class_targets)
        if unique_targets.size < 2:
            best_thresholds.append(float(fallback_thresholds[class_index]))
            continue
        best_score = float("-inf")
        best_threshold = float(fallback_thresholds[class_index])
        for threshold in thresholds.tolist():
            score = _threshold_objective(optimization_metric, class_targets, class_probabilities, float(threshold))
            if score > best_score or (
                np.isclose(score, best_score) and abs(float(threshold) - float(fallback_thresholds[class_index])) < abs(best_threshold - float(fallback_thresholds[class_index]))
            ):
                best_score = score
                best_threshold = float(threshold)
        best_thresholds.append(best_threshold)
    return np.asarray(best_thresholds, dtype=np.float32)


def _compute_pathology_metric_table(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    label_names: list[str],
    expert_column: str,
    run_name: str,
    threshold_source: str,
) -> pd.DataFrame:
    doctor_targets = build_target_matrix(dataframe[expert_column], pathology_names=label_names, merge_map={})
    consensus_targets = build_target_matrix(dataframe["unified_decision_target_labels"], pathology_names=label_names, merge_map={})
    doctor_metrics, doctor_confusion = compute_metrics(doctor_targets, predictions, probabilities, label_names)
    consensus_metrics, consensus_confusion = compute_metrics(consensus_targets, predictions, probabilities, label_names)
    doctor_support = doctor_targets.sum(axis=0).astype(int)
    consensus_support = consensus_targets.sum(axis=0).astype(int)

    rows: list[dict[str, object]] = []
    for label_index, pathology in enumerate(label_names):
        for target_name, metrics_table, confusion_table, support in [
            ("doctor", doctor_metrics, doctor_confusion, int(doctor_support[label_index])),
            ("consensus", consensus_metrics, consensus_confusion, int(consensus_support[label_index])),
        ]:
            rows.append(
                {
                    "run": run_name,
                    "pathology": pathology,
                    "target": target_name,
                    "threshold_source": threshold_source,
                    "n_records": int(len(dataframe)),
                    "positive_support": support,
                    "negative_support": int(len(dataframe) - support),
                    "threshold": float("nan"),
                    "Specificity": float(metrics_table.loc["Specificity", pathology]),
                    "Sensitivity": float(metrics_table.loc["Sensitivity", pathology]),
                    "ROC-AUC": float(metrics_table.loc["ROC-AUC", pathology]),
                    "Precision_PPV_": float(metrics_table.loc["Precision_PPV_", pathology]),
                    "NPV": float(metrics_table.loc["NPV", pathology]),
                    "G-mean": float(metrics_table.loc["G-mean", pathology]),
                    "fbeta2-score": float(metrics_table.loc["fbeta2-score", pathology]),
                    "f1-score": float(metrics_table.loc["f1-score", pathology]),
                    "AP": float(metrics_table.loc["AP", pathology]),
                    "TN": int(confusion_table.loc[pathology, "TN"]),
                    "FP": int(confusion_table.loc[pathology, "FP"]),
                    "FN": int(confusion_table.loc[pathology, "FN"]),
                    "TP": int(confusion_table.loc[pathology, "TP"]),
                }
            )
    return pd.DataFrame(rows)


def _insert_thresholds(table: pd.DataFrame, thresholds: np.ndarray, label_names: list[str]) -> pd.DataFrame:
    threshold_by_pathology = {pathology: float(value) for pathology, value in zip(label_names, thresholds)}
    output = table.copy()
    output["threshold"] = output["pathology"].map(threshold_by_pathology).astype(float)
    return output


def _aggregate_pathologies(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (run_name, target_name, threshold_source), group in table.groupby(["run", "target", "threshold_source"], sort=True):
        row: dict[str, object] = {
            "run": run_name,
            "pathology": "__macro__",
            "target": target_name,
            "threshold_source": threshold_source,
            "n_records": int(group["n_records"].max()),
            "positive_support": float(group["positive_support"].mean()),
            "negative_support": float(group["negative_support"].mean()),
            "threshold": float("nan"),
        }
        for metric_name in SUMMARY_METRICS:
            row[metric_name] = float(group[metric_name].mean(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows)


def _sample_std(values: pd.Series) -> float:
    finite_values = values[np.isfinite(values.to_numpy(dtype=float))].astype(float)
    if len(finite_values) <= 1:
        return 0.0
    return float(finite_values.std(ddof=1))


def _effect_class(doctor_gain: float, consensus_delta: float, eps: float = 1e-12) -> str:
    if doctor_gain > eps and consensus_delta < -eps:
        return "doctor_specific_tradeoff"
    if doctor_gain > eps and consensus_delta > eps:
        return "aligned_gain"
    if doctor_gain < -eps and consensus_delta > eps:
        return "consensus_preferred"
    if abs(doctor_gain) <= eps and abs(consensus_delta) <= eps:
        return "neutral"
    return "mixed"


def _cross_run_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (pathology, target_name, threshold_source), group in table.groupby(["pathology", "target", "threshold_source"], sort=True):
        row: dict[str, object] = {
            "pathology": pathology,
            "target": target_name,
            "threshold_source": threshold_source,
            "run_count": int(group["run"].nunique()),
            "n_records_mean": float(group["n_records"].mean()),
            "positive_support_mean": float(group["positive_support"].mean()),
            "negative_support_mean": float(group["negative_support"].mean()),
            "threshold_mean": float(group["threshold"].mean(skipna=True)) if np.isfinite(group["threshold"]).any() else float("nan"),
            "threshold_std": _sample_std(group["threshold"]),
        }
        for metric_name in SUMMARY_METRICS:
            row[f"{metric_name} Mean"] = float(group[metric_name].mean(skipna=True))
            row[f"{metric_name} STD"] = _sample_std(group[metric_name])
        rows.append(row)
    return pd.DataFrame(rows)


def _load_rank_rows(threshold_root: Path, baselines: list[str], experts: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for baseline in baselines:
        for expert in experts:
            summary_path = threshold_root / baseline / expert / "cross_run_per_pathology_summary.csv"
            if not summary_path.exists():
                continue
            frame = pd.read_csv(summary_path)
            for metric in metrics:
                metric_mean = f"{metric} Mean"
                metric_std = f"{metric} STD"
                if metric_mean not in frame.columns:
                    raise RuntimeError(f"Missing metric column '{metric_mean}' in {summary_path}")
                for pathology in sorted(frame["pathology"].unique()):
                    pathology_frame = frame.loc[frame["pathology"] == pathology].copy()
                    lookup = {
                        (row["target"], row["threshold_source"]): row
                        for _, row in pathology_frame.iterrows()
                    }
                    required_keys = [
                        ("doctor", "doctor_tuned"),
                        ("doctor", "consensus_tuned"),
                        ("consensus", "doctor_tuned"),
                        ("consensus", "consensus_tuned"),
                    ]
                    missing = [key for key in required_keys if key not in lookup]
                    if missing:
                        raise RuntimeError(f"Missing rows {missing} in {summary_path} for pathology={pathology}")

                    doctor_tuned_doctor = lookup[("doctor", "doctor_tuned")]
                    consensus_tuned_doctor = lookup[("doctor", "consensus_tuned")]
                    doctor_tuned_consensus = lookup[("consensus", "doctor_tuned")]
                    consensus_tuned_consensus = lookup[("consensus", "consensus_tuned")]

                    doctor_gain = float(doctor_tuned_doctor[metric_mean] - consensus_tuned_doctor[metric_mean])
                    consensus_delta = float(doctor_tuned_consensus[metric_mean] - consensus_tuned_consensus[metric_mean])
                    rows.append(
                        {
                            "baseline": baseline,
                            "expert": expert,
                            "pathology": pathology,
                            "metric": metric,
                            "run_count": int(doctor_tuned_doctor["run_count"]),
                            "doctor_gain": doctor_gain,
                            "consensus_delta": consensus_delta,
                            "tradeoff_score": float(doctor_gain - consensus_delta),
                            "abs_consensus_delta": abs(consensus_delta),
                            "effect_class": _effect_class(doctor_gain, consensus_delta),
                            "doctor_metric_doctor_tuned_mean": float(doctor_tuned_doctor[metric_mean]),
                            "doctor_metric_doctor_tuned_std": float(doctor_tuned_doctor[metric_std]),
                            "doctor_metric_consensus_tuned_mean": float(consensus_tuned_doctor[metric_mean]),
                            "doctor_metric_consensus_tuned_std": float(consensus_tuned_doctor[metric_std]),
                            "consensus_metric_doctor_tuned_mean": float(doctor_tuned_consensus[metric_mean]),
                            "consensus_metric_doctor_tuned_std": float(doctor_tuned_consensus[metric_std]),
                            "consensus_metric_consensus_tuned_mean": float(consensus_tuned_consensus[metric_mean]),
                            "consensus_metric_consensus_tuned_std": float(consensus_tuned_consensus[metric_std]),
                            "doctor_positive_support_mean": float(doctor_tuned_doctor["positive_support_mean"]),
                            "consensus_positive_support_mean": float(doctor_tuned_consensus["positive_support_mean"]),
                            "doctor_threshold_mean": float(doctor_tuned_doctor["threshold_mean"]),
                            "consensus_threshold_mean": float(consensus_tuned_doctor["threshold_mean"]),
                        }
                    )
    return pd.DataFrame(rows)


def _sort_rank_table(frame: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "tradeoff_score":
        return frame.sort_values(
            by=["tradeoff_score", "doctor_gain", "consensus_delta"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    if sort_by == "doctor_gain":
        return frame.sort_values(
            by=["doctor_gain", "consensus_delta"],
            ascending=[False, True],
        ).reset_index(drop=True)
    if sort_by == "consensus_delta":
        return frame.sort_values(
            by=["consensus_delta", "doctor_gain"],
            ascending=[True, False],
        ).reset_index(drop=True)
    if sort_by == "abs_consensus_delta":
        return frame.sort_values(
            by=["abs_consensus_delta", "doctor_gain"],
            ascending=[False, False],
        ).reset_index(drop=True)
    raise ValueError(f"Unsupported sort mode: {sort_by}")


def _resolve_final_output_root(threshold_root: Path, explicit_output_root: str | None) -> Path:
    if explicit_output_root:
        return Path(explicit_output_root)
    if threshold_root == DEFAULT_OUTPUT_ROOT:
        return DEFAULT_FINAL_OUTPUT_ROOT
    return threshold_root.parent / "final_threshold_analysis"


def finalize_threshold_results(
    *,
    threshold_root: Path,
    output_root: Path,
    baselines: list[str] | None = None,
    experts: list[str] | None = None,
    metrics: list[str] | None = None,
    sort_by: str = "tradeoff_score",
    min_positive_support: float = 0.0,
    doctor_gain_min: float | None = None,
    consensus_delta_max: float | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_baselines = sorted(baselines) if baselines else sorted(path.name for path in threshold_root.iterdir() if path.is_dir())
    resolved_experts = list(experts) if experts is not None else list(DOCTOR_COLUMNS)
    resolved_metrics = list(metrics) if metrics is not None else list(DEFAULT_FINAL_METRICS)

    ranked = _load_rank_rows(
        threshold_root=threshold_root,
        baselines=resolved_baselines,
        experts=resolved_experts,
        metrics=resolved_metrics,
    )
    if ranked.empty:
        raise RuntimeError(f"No threshold summaries found under {threshold_root}")

    if float(min_positive_support) > 0.0:
        ranked = ranked.loc[
            (ranked["doctor_positive_support_mean"] >= float(min_positive_support))
            & (ranked["consensus_positive_support_mean"] >= float(min_positive_support))
        ].copy()

    if doctor_gain_min is not None:
        ranked = ranked.loc[ranked["doctor_gain"] >= float(doctor_gain_min)].copy()

    if consensus_delta_max is not None:
        ranked = ranked.loc[ranked["consensus_delta"] <= float(consensus_delta_max)].copy()

    ranked = _sort_rank_table(ranked, str(sort_by))
    ranked.to_csv(output_root / "ranked_pathology_results.csv", index=False)

    for metric in resolved_metrics:
        metric_frame = ranked.loc[ranked["metric"] == metric].copy()
        metric_frame = _sort_rank_table(metric_frame, str(sort_by))
        metric_frame.to_csv(output_root / f"ranked_pathology_results_{metric.replace('-', '_')}.csv", index=False)

    tradeoffs = ranked.loc[ranked["effect_class"] == "doctor_specific_tradeoff"].copy()
    tradeoffs = _sort_rank_table(tradeoffs, "tradeoff_score")
    tradeoffs.to_csv(output_root / "ranked_pathology_tradeoffs_only.csv", index=False)

    aligned = ranked.loc[ranked["effect_class"] == "aligned_gain"].copy()
    aligned = _sort_rank_table(aligned, "doctor_gain")
    aligned.to_csv(output_root / "ranked_pathology_aligned_only.csv", index=False)


def _write_split_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def analyze_run(
    *,
    baseline: str,
    expert: str,
    run_name: str,
    source_root: Path,
    val_fraction: float,
    split_seed: int,
    optimization_metric: str,
    threshold_grid_step: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    predictions_path = source_root / baseline / run_name / "isp" / "isp_predictions.csv"
    if not predictions_path.exists():
        raise RuntimeError(f"Missing ISP predictions file: {predictions_path}")

    prediction_frame, label_names, probabilities = _load_prediction_payload(predictions_path)
    merged = prediction_frame.copy()

    expert_column = f"{expert}_target_labels"
    if expert_column not in merged.columns:
        raise RuntimeError(f"Missing expert target column '{expert_column}' in {predictions_path}")
    if "unified_decision_target_labels" not in merged.columns:
        raise RuntimeError(f"Missing consensus target column 'unified_decision_target_labels' in {predictions_path}")

    split = split_doctor_records(
        frame=merged,
        expert_column=expert_column,
        label_names=label_names,
        seed=split_seed,
        val_fraction=val_fraction,
    )
    val_frame, val_probabilities = _subset_probabilities(merged, probabilities, split["val"])
    test_frame, test_probabilities = _subset_probabilities(merged, probabilities, split["test"])

    doctor_val_targets = build_target_matrix(val_frame[expert_column], pathology_names=label_names, merge_map={})
    consensus_val_targets = build_target_matrix(
        val_frame["unified_decision_target_labels"],
        pathology_names=label_names,
        merge_map={},
    )

    base_thresholds = np.full(len(label_names), 0.5, dtype=np.float32)
    checkpoint_path = source_root / baseline / run_name / "train" / "best_model.pt"
    if checkpoint_path.exists():
        import torch  # local import to avoid requiring torch for purely CSV-driven analysis at module import time

        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        base_thresholds = np.asarray(checkpoint_payload["thresholds"], dtype=np.float32)
        if len(base_thresholds) != len(label_names):
            raise RuntimeError(
                f"Threshold count mismatch for {checkpoint_path}: "
                f"{len(base_thresholds)} thresholds vs {len(label_names)} labels."
            )

    doctor_thresholds = select_thresholds(
        targets=doctor_val_targets,
        probabilities=val_probabilities,
        optimization_metric=optimization_metric,
        grid_step=threshold_grid_step,
        fallback_thresholds=base_thresholds,
    )
    consensus_thresholds = select_thresholds(
        targets=consensus_val_targets,
        probabilities=val_probabilities,
        optimization_metric=optimization_metric,
        grid_step=threshold_grid_step,
        fallback_thresholds=base_thresholds,
    )

    per_pathology_tables: list[pd.DataFrame] = []
    for threshold_source, thresholds in [
        ("checkpoint", base_thresholds),
        ("doctor_tuned", doctor_thresholds),
        ("consensus_tuned", consensus_thresholds),
    ]:
        test_predictions = (test_probabilities > thresholds).astype(np.float32)
        table = _compute_pathology_metric_table(
            dataframe=test_frame,
            probabilities=test_probabilities,
            predictions=test_predictions,
            label_names=label_names,
            expert_column=expert_column,
            run_name=run_name,
            threshold_source=threshold_source,
        )
        per_pathology_tables.append(_insert_thresholds(table, thresholds, label_names))

    per_pathology = pd.concat(per_pathology_tables, ignore_index=True)
    run_summary = _aggregate_pathologies(per_pathology)
    split_manifest = {
        "baseline": baseline,
        "expert": expert,
        "run": run_name,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "test_fraction": 1.0 - val_fraction,
        "optimization_metric": optimization_metric,
        "threshold_grid_step": threshold_grid_step,
        "label_names": label_names,
        "val_record_ids": split["val"],
        "test_record_ids": split["test"],
        "val_size": len(split["val"]),
        "test_size": len(split["test"]),
    }
    return per_pathology, run_summary, split_manifest


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    final_output_root = _resolve_final_output_root(output_root, args.final_output_root)

    if 1.0 - float(args.val_fraction) < float(args.min_test_fraction):
        raise ValueError(
            f"Requested val_fraction={args.val_fraction} leaves test_fraction={1.0 - float(args.val_fraction):.3f}, "
            f"below min_test_fraction={args.min_test_fraction}."
        )

    run_filters = set(args.runs) if args.runs else None
    for baseline in args.baselines:
        baseline_dir = source_root / baseline
        if not baseline_dir.exists():
            raise RuntimeError(f"Missing baseline directory: {baseline_dir}")
        run_dirs = sorted(path for path in baseline_dir.glob("run_*") if path.is_dir())
        if run_filters is not None:
            run_dirs = [path for path in run_dirs if path.name in run_filters]
        if not run_dirs:
            raise RuntimeError(f"No runs selected for baseline={baseline}")

        for expert in args.experts:
            expert_output_dir = output_root / baseline / expert
            expert_output_dir.mkdir(parents=True, exist_ok=True)

            per_pathology_tables: list[pd.DataFrame] = []
            run_summary_tables: list[pd.DataFrame] = []
            split_manifests: list[dict[str, object]] = []

            for run_dir in run_dirs:
                run_name = run_dir.name
                split_seed = int(args.seed) + int(run_name.split("_")[-1])
                per_pathology, run_summary, split_manifest = analyze_run(
                    baseline=baseline,
                    expert=expert,
                    run_name=run_name,
                    source_root=source_root,
                    val_fraction=float(args.val_fraction),
                    split_seed=split_seed,
                    optimization_metric=str(args.optimization_metric),
                    threshold_grid_step=float(args.threshold_grid_step),
                )
                run_output_dir = expert_output_dir / run_name
                run_output_dir.mkdir(parents=True, exist_ok=True)
                per_pathology.to_csv(run_output_dir / "per_pathology_metrics.csv", index=False)
                run_summary.to_csv(run_output_dir / "macro_summary.csv", index=False)
                _write_split_manifest(run_output_dir / "split_manifest.json", split_manifest)

                per_pathology_tables.append(per_pathology)
                run_summary_tables.append(run_summary)
                split_manifests.append(split_manifest)

            all_per_pathology = pd.concat(per_pathology_tables, ignore_index=True)
            all_run_summary = pd.concat(run_summary_tables, ignore_index=True)
            cross_run_per_pathology = _cross_run_summary(all_per_pathology)
            cross_run_macro = _cross_run_summary(all_run_summary)

            all_per_pathology.to_csv(expert_output_dir / "all_runs_per_pathology_metrics.csv", index=False)
            all_run_summary.to_csv(expert_output_dir / "all_runs_macro_summary.csv", index=False)
            cross_run_per_pathology.to_csv(expert_output_dir / "cross_run_per_pathology_summary.csv", index=False)
            cross_run_macro.to_csv(expert_output_dir / "cross_run_macro_summary.csv", index=False)
            _write_split_manifest(
                expert_output_dir / "analysis_manifest.json",
                {
                    "baseline": baseline,
                    "expert": expert,
                    "runs": [manifest["run"] for manifest in split_manifests],
                    "optimization_metric": str(args.optimization_metric),
                    "val_fraction": float(args.val_fraction),
                    "test_fraction": 1.0 - float(args.val_fraction),
                    "threshold_grid_step": float(args.threshold_grid_step),
                    "source_root": str(source_root),
                },
            )

    finalize_threshold_results(
        threshold_root=output_root,
        output_root=final_output_root,
        baselines=list(args.baselines),
        experts=list(args.experts),
        metrics=list(args.final_metrics),
        sort_by=str(args.final_sort_by),
        min_positive_support=float(args.final_min_positive_support),
        doctor_gain_min=args.final_doctor_gain_min,
        consensus_delta_max=args.final_consensus_delta_max,
    )


if __name__ == "__main__":
    main()
