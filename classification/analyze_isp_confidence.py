from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classification.data.labels import parse_label_list
from classification.tasks.checkpoint import load_checkpoint
from classification.tasks.metrics import compute_metrics


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "classification_log"
GLOBAL_DIRNAME = "confidence_analysis"
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
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
CONSENSUS_DIFFICULTIES = ["easy", "medium"]
HARD_DIFFICULTY = "hard"
SUMMARY_METRICS_OF_INTEREST = ("ROC-AUC", "G-mean")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze confidence and task metrics on ISP disagreement-driven difficulty slices "
            "using exact active panels."
        )
    )
    parser.add_argument(
        "baselines",
        nargs="+",
        help="Baseline directory names inside the selected output root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory containing baseline run folders.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional subset of run directory names, for example run_01 run_02.",
    )
    parser.add_argument(
        "--pathologies",
        nargs="*",
        default=None,
        help="Optional subset of pathology names to analyze.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=30,
        help="Minimum Decision Count required for pathology-level top/min and macro summaries.",
    )
    return parser.parse_args()


def _normalized_confidence(probability: float, threshold: float) -> float:
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    probability = float(np.clip(probability, 0.0, 1.0))
    if probability >= threshold:
        return float((probability - threshold) / (1.0 - threshold))
    return float((threshold - probability) / threshold)


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


def _summary_from_targets(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> pd.Series:
    metrics, _ = compute_metrics(targets, predictions, probabilities, label_names)
    summary = metrics.mean(axis=1, skipna=True)
    return summary.reindex(SUMMARY_METRICS)


def _difficulty_label(positive_votes: int, negative_votes: int) -> str:
    dissent_count = min(positive_votes, negative_votes)
    if dissent_count == 0:
        return "easy"
    if positive_votes == negative_votes or dissent_count >= 2:
        return "hard"
    return "medium"


def _resolve_run_dirs(baseline_dir: Path, run_filters: set[str] | None) -> list[Path]:
    run_dirs = sorted(path for path in baseline_dir.glob("run_*") if path.is_dir())
    if run_filters is not None:
        run_dirs = [path for path in run_dirs if path.name in run_filters]
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {baseline_dir}")
    return run_dirs


def _panel_key_from_row(row: pd.Series, doctor_sources: list[str]) -> tuple[str, ...]:
    return tuple(
        doctor
        for doctor in doctor_sources
        if len(parse_label_list(row[f"{doctor}_labels"])) > 0
    )


def _binary_targets_from_consensus(dataframe: pd.DataFrame) -> np.ndarray:
    return dataframe["unified_label"].to_numpy(dtype=np.float32).reshape(-1, 1)


def _binary_targets_from_doctor(dataframe: pd.DataFrame, doctor: str) -> np.ndarray:
    targets = [
        float(row["pathology"] in set(parse_label_list(row[f"{doctor}_target_labels"])))
        for _, row in dataframe.iterrows()
    ]
    return np.asarray(targets, dtype=np.float32).reshape(-1, 1)


def _build_panel_weighted_summary_for_slice(
    dataframe: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    doctor_sources: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_labels = ["positive"]
    consensus_targets = _binary_targets_from_consensus(dataframe)
    panel_keys = dataframe.apply(lambda row: _panel_key_from_row(row, doctor_sources), axis=1)
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
            doctor_targets = _binary_targets_from_doctor(dataframe.iloc[row_indices], doctor)
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
            detail_rows.append(
                {
                    "panel": "+".join(panel),
                    "n_records": n_records,
                    "metric": metric_name,
                    "Consensus": float(consensus_summary.loc[metric_name]),
                    "Doctor Mean": float(np.mean(finite_doctor_values)) if finite_doctor_values else float("nan"),
                    "Highest Expert": float(np.max(finite_doctor_values)) if finite_doctor_values else float("nan"),
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


def _build_long_table(
    *,
    baseline_name: str,
    run_dir: Path,
    pathology_filter: set[str] | None,
) -> pd.DataFrame:
    checkpoint_path = run_dir / "train" / "best_model.pt"
    predictions_path = run_dir / "isp" / "isp_predictions.csv"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")
    if not predictions_path.exists():
        raise RuntimeError(f"Missing ISP predictions: {predictions_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    pathology_names = list(checkpoint["pathology_names"])
    thresholds = np.asarray(checkpoint["thresholds"], dtype=float)
    threshold_by_pathology = {name: float(value) for name, value in zip(pathology_names, thresholds)}
    selected_pathologies = pathology_names
    if pathology_filter is not None:
        selected_pathologies = [pathology for pathology in pathology_names if pathology in pathology_filter]
        if not selected_pathologies:
            raise RuntimeError(
                f"Requested pathologies are not present in checkpoint labels for {baseline_name}/{run_dir.name}"
            )

    predictions = pd.read_csv(predictions_path)
    run_index = int(run_dir.name.split("_")[-1])
    rows: list[dict[str, object]] = []

    for _, record in predictions.iterrows():
        probability_map = json.loads(record["probabilities"])
        unified_labels = set(parse_label_list(record["unified_decision_target_labels"]))
        active_panel = [
            doctor
            for doctor in DOCTOR_COLUMNS
            if len(parse_label_list(record[f"{doctor}_labels"])) > 0
        ]
        if not active_panel:
            continue
        doctor_target_sets = {
            doctor: set(parse_label_list(record[f"{doctor}_target_labels"]))
            for doctor in active_panel
        }

        for pathology in selected_pathologies:
            if pathology not in probability_map:
                raise RuntimeError(
                    f"Pathology '{pathology}' is missing from probability map in {predictions_path}"
                )
            positive_votes = int(sum(pathology in doctor_target_sets[doctor] for doctor in active_panel))
            negative_votes = int(len(active_panel) - positive_votes)
            difficulty = _difficulty_label(positive_votes, negative_votes)
            threshold = threshold_by_pathology[pathology]
            probability = float(probability_map[pathology])
            predicted_label = int(probability >= threshold)
            majority_label = 0 if positive_votes == negative_votes else int(positive_votes > negative_votes)
            rows.append(
                {
                    "baseline": baseline_name,
                    "run": run_index,
                    "run_name": run_dir.name,
                    "record_id": str(record["record_id"]),
                    "pathology": pathology,
                    "difficulty": difficulty,
                    "panel_key": "+".join(active_panel),
                    "panel_size": len(active_panel),
                    "positive_votes": positive_votes,
                    "negative_votes": negative_votes,
                    "majority_label": majority_label,
                    "unified_label": int(pathology in unified_labels),
                    "threshold": threshold,
                    "probability": probability,
                    "predicted_label": predicted_label,
                    "normalized_confidence": _normalized_confidence(probability, threshold),
                    "unified_decision_target_labels": record["unified_decision_target_labels"],
                    **{f"{doctor}_labels": record[f"{doctor}_labels"] for doctor in DOCTOR_COLUMNS},
                    **{f"{doctor}_target_labels": record[f"{doctor}_target_labels"] for doctor in DOCTOR_COLUMNS},
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No confidence-analysis rows were generated for {baseline_name}/{run_dir.name}")
    return frame


def _run_level_panel_weighted(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for (baseline, run, run_name, pathology, difficulty), slice_df in long_df.groupby(
        ["baseline", "run", "run_name", "pathology", "difficulty"],
        sort=False,
    ):
        predictions = slice_df["predicted_label"].to_numpy(dtype=np.float32).reshape(-1, 1)
        probabilities = slice_df["probability"].to_numpy(dtype=np.float32).reshape(-1, 1)
        summary, details = _build_panel_weighted_summary_for_slice(
            dataframe=slice_df.reset_index(drop=True),
            predictions=predictions,
            probabilities=probabilities,
            doctor_sources=DOCTOR_COLUMNS,
        )
        for metric_name, metric_row in summary.reset_index().iterrows():
            run_rows.append(
                {
                    "baseline": baseline,
                    "run": run,
                    "run_name": run_name,
                    "pathology": pathology,
                    "difficulty": difficulty,
                    "metric": metric_row["metric"],
                    "Consensus": float(metric_row["Consensus"]),
                    "Doctor Mean": float(metric_row["Doctor Mean"]),
                    "Highest Expert": float(metric_row["Highest Expert"]),
                    "Panel Count": int(metric_row["Panel Count"]),
                    "Record Count": int(metric_row["Record Count"]),
                }
            )
        if not details.empty:
            details = details.copy()
            details.insert(0, "pathology", pathology)
            details.insert(0, "difficulty", difficulty)
            details.insert(0, "run_name", run_name)
            details.insert(0, "run", run)
            details.insert(0, "baseline", baseline)
            detail_frames.append(details)
    return pd.DataFrame(run_rows), pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()


def _run_level_confidence(long_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_df.groupby(["baseline", "run", "run_name", "pathology", "difficulty"], as_index=False)
        .agg(
            mean_confidence=("normalized_confidence", "mean"),
            std_confidence=("normalized_confidence", "std"),
            decision_count=("record_id", "size"),
            unique_record_count=("record_id", "nunique"),
            panel_count=("panel_key", "nunique"),
        )
    )
    summary = summary.rename(
        columns={
            "decision_count": "Decision Count",
            "unique_record_count": "Unique Record Count",
            "panel_count": "Panel Count",
        }
    )
    summary["difficulty"] = pd.Categorical(summary["difficulty"], DIFFICULTY_ORDER, ordered=True)
    return summary.sort_values(["baseline", "run", "pathology", "difficulty"], ignore_index=True)


def _aggregate_panel_weighted(run_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (baseline, pathology, difficulty, metric), group in run_level_df.groupby(
        ["baseline", "pathology", "difficulty", "metric"],
        sort=False,
        observed=False,
    ):
        rows.append(
            {
                "baseline": baseline,
                "pathology": pathology,
                "difficulty": difficulty,
                "metric": metric,
                "run_count": int(group["run"].nunique()),
                "Consensus": float(group["Consensus"].mean()),
                "Consensus STD": float(group["Consensus"].std(ddof=1)) if len(group) > 1 else 0.0,
                "Doctor Mean": float(group["Doctor Mean"].mean()),
                "Doctor Mean STD": float(group["Doctor Mean"].std(ddof=1)) if len(group) > 1 else 0.0,
                "Highest Expert": float(group["Highest Expert"].mean()),
                "Highest Expert STD": float(group["Highest Expert"].std(ddof=1)) if len(group) > 1 else 0.0,
                "Panel Count": float(group["Panel Count"].mean()),
                "Panel Count STD": float(group["Panel Count"].std(ddof=1)) if len(group) > 1 else 0.0,
                "Decision Count": float(group["Record Count"].mean()),
                "Decision Count STD": float(group["Record Count"].std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_confidence(run_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (baseline, pathology, difficulty), group in run_level_df.groupby(
        ["baseline", "pathology", "difficulty"],
        sort=False,
        observed=False,
    ):
        rows.append(
            {
                "baseline": baseline,
                "pathology": pathology,
                "difficulty": difficulty,
                "run_count": int(group["run"].nunique()),
                "mean_confidence": float(group["mean_confidence"].mean()),
                "std_confidence": float(group["mean_confidence"].std(ddof=1)) if len(group) > 1 else 0.0,
                "Decision Count": float(group["Decision Count"].mean()),
                "Decision Count STD": float(group["Decision Count"].std(ddof=1)) if len(group) > 1 else 0.0,
                "Unique Record Count": float(group["Unique Record Count"].mean()),
                "Unique Record Count STD": float(group["Unique Record Count"].std(ddof=1)) if len(group) > 1 else 0.0,
                "Panel Count": float(group["Panel Count"].mean()),
                "Panel Count STD": float(group["Panel Count"].std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_record_counts(run_level_df: pd.DataFrame) -> pd.DataFrame:
    return _aggregate_confidence(run_level_df)[
        [
            "baseline",
            "pathology",
            "difficulty",
            "run_count",
            "Decision Count",
            "Decision Count STD",
            "Unique Record Count",
            "Unique Record Count STD",
            "Panel Count",
            "Panel Count STD",
        ]
    ].copy()


def _filtered_rows(frame: pd.DataFrame, *, difficulty: str, min_count: int) -> pd.DataFrame:
    subset = frame.loc[frame["difficulty"] == difficulty].copy()
    if subset.empty:
        return subset
    return subset.loc[subset["Decision Count"] >= int(min_count)].copy()


def _macro_mean(values: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.size == 0:
        return float("nan")
    return float(finite_values.mean())


def _macro_std(values: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.size <= 1:
        return 0.0
    return float(finite_values.std(ddof=1))


def _run_level_macro_confidence_summary(
    run_level_confidence_df: pd.DataFrame,
    *,
    min_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for baseline, baseline_df in run_level_confidence_df.groupby("baseline", sort=True):
        baseline_row: dict[str, object] = {"baseline": baseline}
        for difficulty in DIFFICULTY_ORDER:
            run_rows = []
            for run, run_df in baseline_df.loc[baseline_df["difficulty"] == difficulty].groupby("run", sort=True):
                filtered = run_df.loc[run_df["Decision Count"] >= int(min_count)].copy()
                run_rows.append(
                    {
                        "run": run,
                        "macro_mean": _macro_mean(filtered["mean_confidence"]) if not filtered.empty else float("nan"),
                        "pathology_count": int(filtered["pathology"].nunique()) if not filtered.empty else 0,
                    }
                )
            run_summary = pd.DataFrame(run_rows)
            baseline_row[f"confidence_{difficulty}_macro_mean"] = _macro_mean(run_summary["macro_mean"])
            baseline_row[f"confidence_{difficulty}_macro_std"] = _macro_std(run_summary["macro_mean"])
            baseline_row[f"confidence_{difficulty}_pathology_count_mean"] = _macro_mean(run_summary["pathology_count"])
        rows.append(baseline_row)
    return rows


def _run_level_metric_summary(
    run_level_panel_df: pd.DataFrame,
    *,
    min_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_key_map = {
        "ROC-AUC": "roc_auc",
        "G-mean": "g_mean",
    }
    source_key_map = {
        "Consensus": "consensus",
        "Highest Expert": "highest",
    }

    for baseline, baseline_df in run_level_panel_df.groupby("baseline", sort=True):
        baseline_row: dict[str, object] = {"baseline": baseline}

        # Easy / medium: Consensus and Highest Expert
        subset = baseline_df.loc[
            baseline_df["difficulty"].isin(CONSENSUS_DIFFICULTIES)
            & baseline_df["metric"].isin(SUMMARY_METRICS_OF_INTEREST)
        ].copy()
        for difficulty in CONSENSUS_DIFFICULTIES:
            for metric in SUMMARY_METRICS_OF_INTEREST:
                metric_subset = subset.loc[
                    (subset["difficulty"] == difficulty)
                    & (subset["metric"] == metric)
                ].copy()
                for source_column, source_key in source_key_map.items():
                    run_rows = []
                    for run, run_df in metric_subset.groupby("run", sort=True):
                        filtered = run_df.loc[run_df["Record Count"] >= int(min_count)].copy()
                        run_rows.append(
                            {
                                "run": run,
                                "macro_mean": _macro_mean(filtered[source_column]) if not filtered.empty else float("nan"),
                                "pathology_count": int(filtered["pathology"].nunique()) if not filtered.empty else 0,
                            }
                        )
                    run_summary = pd.DataFrame(run_rows)
                    prefix = f"{metric_key_map[metric]}_{difficulty}_{source_key}"
                    baseline_row[f"{prefix}_macro_mean"] = _macro_mean(run_summary["macro_mean"])
                    baseline_row[f"{prefix}_macro_std"] = _macro_std(run_summary["macro_mean"])
                    baseline_row[f"{prefix}_pathology_count_mean"] = _macro_mean(run_summary["pathology_count"])

        # Hard: Highest Expert only
        hard_subset = baseline_df.loc[
            (baseline_df["difficulty"] == HARD_DIFFICULTY)
            & baseline_df["metric"].isin(SUMMARY_METRICS_OF_INTEREST)
        ].copy()
        for metric in SUMMARY_METRICS_OF_INTEREST:
            metric_subset = hard_subset.loc[hard_subset["metric"] == metric].copy()
            run_rows = []
            for run, run_df in metric_subset.groupby("run", sort=True):
                filtered = run_df.loc[run_df["Record Count"] >= int(min_count)].copy()
                run_rows.append(
                    {
                        "run": run,
                        "macro_mean": _macro_mean(filtered["Highest Expert"]) if not filtered.empty else float("nan"),
                        "pathology_count": int(filtered["pathology"].nunique()) if not filtered.empty else 0,
                    }
                )
            run_summary = pd.DataFrame(run_rows)
            prefix = f"hard_highest_expert_{metric_key_map[metric]}"
            baseline_row[f"{prefix}_macro_mean"] = _macro_mean(run_summary["macro_mean"])
            baseline_row[f"{prefix}_macro_std"] = _macro_std(run_summary["macro_mean"])
            baseline_row[f"{prefix}_pathology_count_mean"] = _macro_mean(run_summary["pathology_count"])
        rows.append(baseline_row)

    return rows


def _build_confidence_summary(
    confidence_summary_df: pd.DataFrame,
    *,
    min_count: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    baseline_records: list[dict[str, object]] = []

    for baseline, baseline_df in confidence_summary_df.groupby("baseline", sort=True):
        baseline_row: dict[str, object] = {"baseline": baseline}
        for difficulty in DIFFICULTY_ORDER:
            filtered = _filtered_rows(baseline_df, difficulty=difficulty, min_count=min_count)
            if filtered.empty:
                baseline_row[f"confidence_{difficulty}_macro_mean"] = float("nan")
                baseline_row[f"confidence_{difficulty}_macro_std"] = float("nan")
                baseline_row[f"confidence_{difficulty}_pathology_count"] = 0
                continue
            baseline_row[f"confidence_{difficulty}_macro_mean"] = _macro_mean(filtered["mean_confidence"])
            baseline_row[f"confidence_{difficulty}_macro_std"] = float(filtered["std_confidence"].mean())
            baseline_row[f"confidence_{difficulty}_pathology_count"] = int(filtered["pathology"].nunique())

            for direction, ascending in (("top", False), ("min", True)):
                ranked = filtered.sort_values(
                    ["mean_confidence", "Decision Count", "pathology"],
                    ascending=[ascending, False, True],
                ).head(5)
                for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                    rows.append(
                        {
                            "baseline": baseline,
                            "analysis": "confidence",
                            "difficulty": difficulty,
                            "rank_direction": direction,
                            "rank": rank,
                            "pathology": row["pathology"],
                            "value": float(row["mean_confidence"]),
                            "Decision Count": float(row["Decision Count"]),
                            "Unique Record Count": float(row["Unique Record Count"]),
                        }
                    )
        baseline_records.append(baseline_row)

    return pd.DataFrame(rows), baseline_records


def _build_consensus_metric_summary(
    panel_summary_df: pd.DataFrame,
    *,
    min_count: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    baseline_records: dict[str, dict[str, object]] = {}

    subset = panel_summary_df.loc[
        panel_summary_df["difficulty"].isin(CONSENSUS_DIFFICULTIES)
        & panel_summary_df["metric"].isin(SUMMARY_METRICS_OF_INTEREST)
    ].copy()

    for (baseline, difficulty, metric), group in subset.groupby(
        ["baseline", "difficulty", "metric"],
        sort=True,
        observed=False,
    ):
        filtered = group.loc[group["Decision Count"] >= int(min_count)].copy()
        baseline_row = baseline_records.setdefault(baseline, {"baseline": baseline})
        key_prefix = f"{metric.lower().replace('-', '_')}_{difficulty}_consensus"
        if filtered.empty:
            baseline_row[f"{key_prefix}_macro_mean"] = float("nan")
            baseline_row[f"{key_prefix}_macro_std"] = float("nan")
            baseline_row[f"{key_prefix}_pathology_count"] = 0
            continue
        baseline_row[f"{key_prefix}_macro_mean"] = _macro_mean(filtered["Consensus"])
        baseline_row[f"{key_prefix}_macro_std"] = float(filtered["Consensus STD"].mean())
        baseline_row[f"{key_prefix}_pathology_count"] = int(filtered["pathology"].nunique())

        for direction, ascending in (("top", False), ("min", True)):
            ranked = filtered.sort_values(
                ["Consensus", "Decision Count", "pathology"],
                ascending=[ascending, False, True],
            ).head(5)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "baseline": baseline,
                        "analysis": "consensus",
                        "difficulty": difficulty,
                        "metric": metric,
                        "rank_direction": direction,
                        "rank": rank,
                        "pathology": row["pathology"],
                        "value": float(row["Consensus"]),
                        "Highest Expert": float(row["Highest Expert"]),
                        "Decision Count": float(row["Decision Count"]),
                    }
                )

    return pd.DataFrame(rows), list(baseline_records.values())


def _build_hard_highest_expert_summary(
    panel_summary_df: pd.DataFrame,
    *,
    min_count: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    baseline_records: dict[str, dict[str, object]] = {}

    subset = panel_summary_df.loc[
        (panel_summary_df["difficulty"] == HARD_DIFFICULTY)
        & panel_summary_df["metric"].isin(SUMMARY_METRICS_OF_INTEREST)
    ].copy()

    for (baseline, metric), group in subset.groupby(
        ["baseline", "metric"],
        sort=True,
        observed=False,
    ):
        filtered = group.loc[group["Decision Count"] >= int(min_count)].copy()
        baseline_row = baseline_records.setdefault(baseline, {"baseline": baseline})
        key_prefix = f"hard_highest_expert_{metric.lower().replace('-', '_')}"
        if filtered.empty:
            baseline_row[f"{key_prefix}_macro_mean"] = float("nan")
            baseline_row[f"{key_prefix}_macro_std"] = float("nan")
            baseline_row[f"{key_prefix}_pathology_count"] = 0
            continue
        baseline_row[f"{key_prefix}_macro_mean"] = _macro_mean(filtered["Highest Expert"])
        baseline_row[f"{key_prefix}_macro_std"] = float(filtered["Highest Expert STD"].mean())
        baseline_row[f"{key_prefix}_pathology_count"] = int(filtered["pathology"].nunique())

        for direction, ascending in (("top", False), ("min", True)):
            ranked = filtered.sort_values(
                ["Highest Expert", "Decision Count", "pathology"],
                ascending=[ascending, False, True],
            ).head(5)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "baseline": baseline,
                        "analysis": "hard_highest_expert",
                        "difficulty": HARD_DIFFICULTY,
                        "metric": metric,
                        "rank_direction": direction,
                        "rank": rank,
                        "pathology": row["pathology"],
                        "value": float(row["Highest Expert"]),
                        "Doctor Mean": float(row["Doctor Mean"]),
                        "Decision Count": float(row["Decision Count"]),
                    }
                )

    return pd.DataFrame(rows), list(baseline_records.values())


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    global_dir = output_root / GLOBAL_DIRNAME
    global_dir.mkdir(parents=True, exist_ok=True)

    run_filters = set(args.runs) if args.runs else None
    pathology_filter = set(args.pathologies) if args.pathologies else None
    min_count = int(args.min_count)

    global_summary_rows: list[dict[str, object]] = []

    for baseline in args.baselines:
        baseline_dir = output_root / baseline
        if not baseline_dir.exists():
            raise RuntimeError(f"Missing baseline directory: {baseline_dir}")
        run_dirs = _resolve_run_dirs(baseline_dir, run_filters)
        baseline_output_dir = baseline_dir / GLOBAL_DIRNAME
        baseline_output_dir.mkdir(parents=True, exist_ok=True)

        baseline_run_level_panel_frames: list[pd.DataFrame] = []
        baseline_run_level_confidence_frames: list[pd.DataFrame] = []
        for run_dir in run_dirs:
            long_df = _build_long_table(
                baseline_name=baseline,
                run_dir=run_dir,
                pathology_filter=pathology_filter,
            )
            run_level_panel_df, detail_df = _run_level_panel_weighted(long_df)
            run_level_confidence_df = _run_level_confidence(long_df)
            baseline_run_level_panel_frames.append(run_level_panel_df)
            baseline_run_level_confidence_frames.append(run_level_confidence_df)

        baseline_panel_df = pd.concat(baseline_run_level_panel_frames, ignore_index=True)
        baseline_confidence_df = pd.concat(baseline_run_level_confidence_frames, ignore_index=True)
        baseline_panel_summary = _aggregate_panel_weighted(baseline_panel_df).sort_values(
            ["pathology", "difficulty", "metric"],
            ignore_index=True,
        )
        baseline_confidence_summary = _aggregate_confidence(baseline_confidence_df).sort_values(
            ["pathology", "difficulty"],
            ignore_index=True,
        )
        confidence_highlights_df, confidence_summary_rows = _build_confidence_summary(
            baseline_confidence_summary,
            min_count=min_count,
        )
        consensus_highlights_df, consensus_summary_rows = _build_consensus_metric_summary(
            baseline_panel_summary,
            min_count=min_count,
        )
        hard_highlights_df, hard_summary_rows = _build_hard_highest_expert_summary(
            baseline_panel_summary,
            min_count=min_count,
        )

        run_level_confidence_summary_rows = _run_level_macro_confidence_summary(
            baseline_confidence_df,
            min_count=min_count,
        )
        run_level_metric_summary_rows = _run_level_metric_summary(
            baseline_panel_df,
            min_count=min_count,
        )

        baseline_summary: dict[str, object] = {"baseline": baseline}
        for row in confidence_summary_rows + consensus_summary_rows + hard_summary_rows:
            baseline_summary.update(row)
        for row in run_level_confidence_summary_rows + run_level_metric_summary_rows:
            baseline_summary.update(row)
        global_summary_rows.append(baseline_summary)

        baseline_confidence_summary.to_csv(baseline_output_dir / "confidence_pathology_summary.csv", index=False)
        baseline_panel_summary.to_csv(baseline_output_dir / "performance_pathology_summary.csv", index=False)
        confidence_highlights_df.to_csv(baseline_output_dir / "confidence_pathology_highlights.csv", index=False)
        consensus_highlights_df.to_csv(baseline_output_dir / "consensus_easy_medium_pathology_highlights.csv", index=False)
        hard_highlights_df.to_csv(baseline_output_dir / "hard_highest_expert_pathology_highlights.csv", index=False)
        pd.DataFrame([baseline_summary]).to_csv(baseline_output_dir / "baseline_final_summary.csv", index=False)

    confidence_metadata = {
        "confidence_metric": "normalized_confidence",
        "formula": "if p >= t: (p - t) / (1 - t); else: (t - p) / t",
        "threshold_source": "checkpoint pathology-specific thresholds",
        "difficulty_levels": DIFFICULTY_ORDER,
        "consensus_difficulties": CONSENSUS_DIFFICULTIES,
        "hard_difficulty_source": "Highest Expert only",
        "tie_rule": "ties are retained, mapped to majority label 0, and assigned to hard",
        "active_panel_rule": "active panel is defined by non-empty doctor*_labels columns",
        "negative_vote_rule": "active doctors with empty doctor*_target_labels are treated as negative for the pathology",
        "min_count_for_highlights_and_macro": min_count,
    }
    (global_dir / "metadata.json").write_text(json.dumps(confidence_metadata, indent=2))
    pd.DataFrame(global_summary_rows).sort_values("baseline", ignore_index=True).to_csv(
        global_dir / "all_baselines_final_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
