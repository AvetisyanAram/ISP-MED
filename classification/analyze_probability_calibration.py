from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classification.data.labels import parse_label_list
from classification.tasks.checkpoint import load_checkpoint


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "classification_log"
GLOBAL_DIRNAME = "probability_calibration_summary"
DOCTOR_COLUMNS = [f"doctor{i}_target_labels" for i in range(1, 8)]
TARGET_SOURCES = ("consensus", "doctor_single", "doctor_vote_fraction", "ptbxl_test")
BINNING_STRATEGIES = ("uniform", "quantile")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze probability calibration for completed classification runs "
            "on ISP consensus labels, individual doctor labels, pooled doctor vote fractions, "
            "and PTB-XL test labels."
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
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for reliability summaries.",
    )
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=1,
        help="Minimum sample count required to keep a bin in exported bin tables.",
    )
    parser.add_argument(
        "--merge-global",
        action="store_true",
        help=(
            "Merge selected baselines into existing global summary CSVs instead of "
            "replacing the whole probability_calibration_summary directory contents."
        ),
    )
    return parser.parse_args()


def _resolve_run_dirs(baseline_dir: Path, run_filters: set[str] | None) -> list[Path]:
    run_dirs = sorted(path for path in baseline_dir.glob("run_*") if path.is_dir())
    if run_filters is not None:
        run_dirs = [path for path in run_dirs if path.name in run_filters]
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {baseline_dir}")
    return run_dirs


def _safe_probabilities(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(values.astype(float), eps, 1.0 - eps)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / weight_sum)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean_value = _weighted_mean(values, weights)
    if not np.isfinite(mean_value):
        return float("nan")
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return float("nan")
    variance = float(np.sum(weights * np.square(values - mean_value)) / weight_sum)
    return float(np.sqrt(max(variance, 0.0)))


def _safe_logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    safe_probabilities = _safe_probabilities(probabilities, eps=eps)
    return np.log(safe_probabilities / (1.0 - safe_probabilities))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive_mask = values >= 0.0
    result = np.empty_like(values, dtype=float)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-values[positive_mask]))
    exp_values = np.exp(values[~positive_mask])
    result[~positive_mask] = exp_values / (1.0 + exp_values)
    return result


def _calibration_intercept_and_slope(
    probabilities: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, float]:
    if probabilities.size == 0:
        return float("nan"), float("nan")
    finite_targets = targets[np.isfinite(targets)]
    if finite_targets.size == 0 or np.allclose(finite_targets, finite_targets[0]):
        return float("nan"), float("nan")

    x = _safe_logit(probabilities)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.asarray([0.0, 1.0], dtype=float)

    for _ in range(max_iter):
        eta = design @ beta
        mu = _sigmoid(eta)
        variance = np.clip(mu * (1.0 - mu), 1e-12, None)
        residual = targets - mu
        score = design.T @ (weights * residual)
        weighted_design = design * (weights * variance)[:, None]
        hessian = -(design.T @ weighted_design)
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta_next = beta - step
        if np.max(np.abs(beta_next - beta)) < tol:
            beta = beta_next
            break
        beta = beta_next

    return float(beta[0]), float(beta[1])


def _build_bin_edges(probabilities: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
    if strategy == "quantile":
        raw_edges = np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1))
        raw_edges[0] = 0.0
        raw_edges[-1] = 1.0
        unique_edges = np.unique(raw_edges)
        if unique_edges.size < 2:
            return np.asarray([0.0, 1.0], dtype=float)
        return unique_edges.astype(float)
    raise ValueError(f"Unsupported binning strategy: {strategy}")


def _assign_bins(probabilities: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size < 2:
        return np.zeros(probabilities.shape[0], dtype=int)
    bin_ids = np.searchsorted(edges[1:-1], probabilities, side="right")
    return bin_ids.astype(int)


def _bin_summary(
    frame: pd.DataFrame,
    *,
    n_bins: int,
    strategy: str,
    min_bin_count: int,
) -> pd.DataFrame:
    probabilities = frame["probability"].to_numpy(dtype=float)
    targets = frame["target"].to_numpy(dtype=float)
    weights = frame["sample_weight"].to_numpy(dtype=float)
    edges = _build_bin_edges(probabilities, n_bins=n_bins, strategy=strategy)
    bin_ids = _assign_bins(probabilities, edges)

    rows: list[dict[str, object]] = []
    for bin_index in range(max(edges.size - 1, 1)):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        count = int(mask.sum())
        if count < min_bin_count:
            continue
        prob_bin = probabilities[mask]
        target_bin = targets[mask]
        weight_bin = weights[mask]
        mean_probability = _weighted_mean(prob_bin, weight_bin)
        mean_target = _weighted_mean(target_bin, weight_bin)
        gap = float(mean_probability - mean_target)
        rows.append(
            {
                "bin_index": bin_index,
                "bin_lower": float(edges[bin_index]),
                "bin_upper": float(edges[bin_index + 1]) if edges.size > 1 else 1.0,
                "count": count,
                "weight_sum": float(weight_bin.sum()),
                "mean_probability": mean_probability,
                "mean_target": mean_target,
                "calibration_gap": gap,
                "absolute_gap": float(abs(gap)),
            }
        )
    return pd.DataFrame(rows)


def _scalar_metrics(frame: pd.DataFrame, *, n_bins: int) -> dict[str, float]:
    probabilities = frame["probability"].to_numpy(dtype=float)
    targets = frame["target"].to_numpy(dtype=float)
    weights = frame["sample_weight"].to_numpy(dtype=float)
    safe_probabilities = _safe_probabilities(probabilities)

    brier = float(np.average(np.square(probabilities - targets), weights=weights))
    log_loss = float(
        np.average(
            -(targets * np.log(safe_probabilities) + (1.0 - targets) * np.log(1.0 - safe_probabilities)),
            weights=weights,
        )
    )

    scalar_rows: dict[str, float] = {
        "sample_count": float(len(frame)),
        "weight_sum": float(weights.sum()),
        "positive_rate": _weighted_mean(targets, weights),
        "mean_probability": _weighted_mean(probabilities, weights),
        "std_probability": _weighted_std(probabilities, weights),
        "brier_score": brier,
        "log_loss": log_loss,
    }
    calibration_intercept, calibration_slope = _calibration_intercept_and_slope(
        probabilities=probabilities,
        targets=targets,
        weights=weights,
    )
    scalar_rows["calibration_intercept"] = calibration_intercept
    scalar_rows["calibration_slope"] = calibration_slope

    for strategy in BINNING_STRATEGIES:
        bin_df = _bin_summary(frame, n_bins=n_bins, strategy=strategy, min_bin_count=1)
        if bin_df.empty:
            scalar_rows[f"ece_{strategy}"] = float("nan")
            scalar_rows[f"rmsce_{strategy}"] = float("nan")
            scalar_rows[f"mce_{strategy}"] = float("nan")
            continue
        bin_weights = bin_df["weight_sum"].to_numpy(dtype=float)
        total_weight = float(bin_weights.sum())
        abs_gaps = bin_df["absolute_gap"].to_numpy(dtype=float)
        gaps = bin_df["calibration_gap"].to_numpy(dtype=float)
        scalar_rows[f"ece_{strategy}"] = float(np.sum((bin_weights / total_weight) * abs_gaps))
        scalar_rows[f"rmsce_{strategy}"] = float(np.sqrt(np.sum((bin_weights / total_weight) * np.square(gaps))))
        scalar_rows[f"mce_{strategy}"] = float(np.max(abs_gaps))

    return scalar_rows


def _load_isp_rows(
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
    selected_pathologies = pathology_names
    if pathology_filter is not None:
        selected_pathologies = [pathology for pathology in pathology_names if pathology in pathology_filter]
        if not selected_pathologies:
            raise RuntimeError(
                f"Requested pathologies are not present in checkpoint labels for {baseline_name}/{run_dir.name}"
            )

    predictions = pd.read_csv(predictions_path)
    rows: list[dict[str, object]] = []
    run_index = int(run_dir.name.split("_")[-1])

    for _, record in predictions.iterrows():
        probability_map = json.loads(record["probabilities"])
        consensus_labels = set(parse_label_list(record.get("unified_decision_target_labels", [])))
        doctor_label_sets = {
            doctor_column.replace("_target_labels", ""): set(parse_label_list(record[doctor_column]))
            for doctor_column in DOCTOR_COLUMNS
        }
        available_doctors = {doctor_name: labels for doctor_name, labels in doctor_label_sets.items() if labels}

        for pathology in selected_pathologies:
            if pathology not in probability_map:
                raise RuntimeError(
                    f"Pathology '{pathology}' is missing from probability map in {predictions_path}"
                )
            probability = float(probability_map[pathology])
            rows.append(
                {
                    "baseline": baseline_name,
                    "run": run_index,
                    "run_name": run_dir.name,
                    "domain": "isp",
                    "target_source": "consensus",
                    "target_id": "consensus",
                    "record_id": str(record["record_id"]),
                    "pathology": pathology,
                    "probability": probability,
                    "target": float(pathology in consensus_labels),
                    "sample_weight": 1.0,
                }
            )

            if available_doctors:
                positive_votes = sum(pathology in labels for labels in available_doctors.values())
                rows.append(
                    {
                        "baseline": baseline_name,
                        "run": run_index,
                        "run_name": run_dir.name,
                        "domain": "isp",
                        "target_source": "doctor_vote_fraction",
                        "target_id": "doctor_vote_fraction",
                        "record_id": str(record["record_id"]),
                        "pathology": pathology,
                        "probability": probability,
                        "target": float(positive_votes / len(available_doctors)),
                        "sample_weight": float(len(available_doctors)),
                    }
                )

            for doctor_name, labels in available_doctors.items():
                rows.append(
                    {
                        "baseline": baseline_name,
                        "run": run_index,
                        "run_name": run_dir.name,
                        "domain": "isp",
                        "target_source": "doctor_single",
                        "target_id": doctor_name,
                        "record_id": str(record["record_id"]),
                        "pathology": pathology,
                        "probability": probability,
                        "target": float(pathology in labels),
                        "sample_weight": 1.0,
                    }
                )

    if not rows:
        raise RuntimeError(f"No ISP calibration rows were generated for {baseline_name}/{run_dir.name}")
    return pd.DataFrame(rows)


def _load_ptbxl_rows(
    *,
    baseline_name: str,
    run_dir: Path,
    pathology_filter: set[str] | None,
) -> pd.DataFrame:
    checkpoint_path = run_dir / "train" / "best_model.pt"
    predictions_path = run_dir / "ptbxl_test" / "ptbxl_test_predictions.csv"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")
    if not predictions_path.exists():
        raise RuntimeError(f"Missing PTB-XL test predictions: {predictions_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    pathology_names = list(checkpoint["pathology_names"])
    selected_pathologies = pathology_names
    if pathology_filter is not None:
        selected_pathologies = [pathology for pathology in pathology_names if pathology in pathology_filter]
        if not selected_pathologies:
            raise RuntimeError(
                f"Requested pathologies are not present in checkpoint labels for {baseline_name}/{run_dir.name}"
            )

    predictions = pd.read_csv(predictions_path)
    rows: list[dict[str, object]] = []
    run_index = int(run_dir.name.split("_")[-1])

    for _, record in predictions.iterrows():
        probability_map = json.loads(record["probabilities"])
        true_labels = set(parse_label_list(record["true_labels"]))
        for pathology in selected_pathologies:
            if pathology not in probability_map:
                raise RuntimeError(
                    f"Pathology '{pathology}' is missing from probability map in {predictions_path}"
                )
            rows.append(
                {
                    "baseline": baseline_name,
                    "run": run_index,
                    "run_name": run_dir.name,
                    "domain": "ptbxl_test",
                    "target_source": "ptbxl_test",
                    "target_id": "ptbxl_test",
                    "record_id": str(record["record_id"]),
                    "pathology": pathology,
                    "probability": float(probability_map[pathology]),
                    "target": float(pathology in true_labels),
                    "sample_weight": 1.0,
                }
            )

    if not rows:
        raise RuntimeError(f"No PTB-XL calibration rows were generated for {baseline_name}/{run_dir.name}")
    return pd.DataFrame(rows)


def _pathology_summary(long_df: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["baseline", "run", "run_name", "domain", "target_source", "target_id", "pathology"]
    for group_keys, group_df in long_df.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, group_keys))
        row.update(_scalar_metrics(group_df, n_bins=n_bins))
        rows.append(row)
    return pd.DataFrame(rows)


def _bin_summary_table(long_df: pd.DataFrame, *, n_bins: int, min_bin_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["baseline", "run", "run_name", "domain", "target_source", "target_id", "pathology"]
    for group_keys, group_df in long_df.groupby(group_columns, sort=True):
        group_prefix = dict(zip(group_columns, group_keys))
        for strategy in BINNING_STRATEGIES:
            bin_df = _bin_summary(group_df, n_bins=n_bins, strategy=strategy, min_bin_count=min_bin_count)
            if bin_df.empty:
                continue
            for _, bin_row in bin_df.iterrows():
                row = dict(group_prefix)
                row["binning_strategy"] = strategy
                row.update(bin_row.to_dict())
                rows.append(row)
    return pd.DataFrame(rows)


def _calibration_curve_table(long_df: pd.DataFrame, *, n_bins: int, min_bin_count: int) -> pd.DataFrame:
    curve_df = _bin_summary_table(long_df, n_bins=n_bins, min_bin_count=min_bin_count)
    if curve_df.empty:
        return curve_df
    return curve_df.rename(
        columns={
            "mean_probability": "prob_pred",
            "mean_target": "prob_true",
        }
    )


def _baseline_summary(pathology_summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "positive_rate",
        "mean_probability",
        "std_probability",
        "brier_score",
        "log_loss",
        "calibration_intercept",
        "calibration_slope",
        "ece_uniform",
        "rmsce_uniform",
        "mce_uniform",
        "ece_quantile",
        "rmsce_quantile",
        "mce_quantile",
    ]
    rows: list[dict[str, object]] = []
    for group_keys, group_df in pathology_summary_df.groupby(["baseline", "domain", "target_source"], sort=True):
        baseline_name, domain, target_source = group_keys
        row: dict[str, object] = {
            "baseline": baseline_name,
            "domain": domain,
            "target_source": target_source,
            "run_count": int(group_df["run"].nunique()),
            "pathology_group_count": int(len(group_df)),
            "target_id_count": int(group_df["target_id"].nunique()),
            "mean_sample_count": float(group_df["sample_count"].mean()),
            "std_sample_count": float(group_df["sample_count"].std(ddof=1)) if len(group_df) > 1 else 0.0,
        }
        for metric_name in metric_columns:
            values = group_df[metric_name].to_numpy(dtype=float)
            row[f"mean_{metric_name}"] = float(np.nanmean(values))
            row[f"std_{metric_name}"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["domain", "target_source", "baseline"], ignore_index=True)


def _plot_reliability_diagram(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    n_bins: int,
    strategy: str,
) -> None:
    bin_df = _bin_summary(frame, n_bins=n_bins, strategy=strategy, min_bin_count=1)
    if bin_df.empty:
        return

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.2, color="#777777", label="Perfect calibration")
    ax.plot(
        bin_df["mean_probability"].to_numpy(dtype=float),
        bin_df["mean_target"].to_numpy(dtype=float),
        marker="o",
        linewidth=2.0,
        color="#1f4e79",
        label="Observed",
    )
    sizes = np.maximum(bin_df["count"].to_numpy(dtype=float), 1.0)
    size_scale = 20.0 + 180.0 * (sizes / sizes.max())
    ax.scatter(
        bin_df["mean_probability"].to_numpy(dtype=float),
        bin_df["mean_target"].to_numpy(dtype=float),
        s=size_scale,
        color="#7aa6c2",
        alpha=0.7,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_summary_scatter(summary_df: pd.DataFrame, *, output_path: Path, metric_x: str, metric_y: str, title: str) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    x = summary_df[metric_x].to_numpy(dtype=float)
    y = summary_df[metric_y].to_numpy(dtype=float)
    ax.scatter(x, y, color="#0b6e4f")
    for _, row in summary_df.iterrows():
        label = f"{row['baseline']} [{row['domain']}:{row['target_source']}]"
        ax.annotate(label, (row[metric_x], row[metric_y]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _merge_by_baseline(
    existing_path: Path,
    new_df: pd.DataFrame,
    *,
    baseline_names: list[str],
) -> pd.DataFrame:
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        existing_df = existing_df[~existing_df["baseline"].isin(baseline_names)].copy()
        return pd.concat([existing_df, new_df], ignore_index=True)
    return new_df.copy()


def analyze_baselines(
    *,
    baseline_names: list[str],
    output_root: Path,
    run_filters: set[str] | None = None,
    pathology_filter: set[str] | None = None,
    n_bins: int = 10,
    min_bin_count: int = 1,
    merge_global: bool = False,
) -> dict[str, str]:
    output_root = output_root.resolve()
    global_dir = output_root / GLOBAL_DIRNAME
    global_dir.mkdir(parents=True, exist_ok=True)

    all_pathology_summaries: list[pd.DataFrame] = []
    all_baseline_summaries: list[pd.DataFrame] = []
    all_calibration_curves: list[pd.DataFrame] = []

    for baseline_name in baseline_names:
        baseline_dir = output_root / baseline_name
        if not baseline_dir.exists():
            raise RuntimeError(f"Missing baseline directory: {baseline_dir}")

        run_dirs = _resolve_run_dirs(baseline_dir, run_filters=run_filters)
        baseline_long = pd.concat(
            [
                pd.concat(
                    [
                        _load_isp_rows(
                            baseline_name=baseline_name,
                            run_dir=run_dir,
                            pathology_filter=pathology_filter,
                        ),
                        _load_ptbxl_rows(
                            baseline_name=baseline_name,
                            run_dir=run_dir,
                            pathology_filter=pathology_filter,
                        ),
                    ],
                    ignore_index=True,
                )
                for run_dir in run_dirs
            ],
            ignore_index=True,
        )

        pathology_summary_df = _pathology_summary(baseline_long, n_bins=n_bins)
        bin_summary_df = _bin_summary_table(baseline_long, n_bins=n_bins, min_bin_count=min_bin_count)
        calibration_curve_df = _calibration_curve_table(baseline_long, n_bins=n_bins, min_bin_count=min_bin_count)
        baseline_summary_df = _baseline_summary(pathology_summary_df)

        calibration_dir = baseline_dir / "probability_calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        baseline_long.to_csv(calibration_dir / "calibration_long.csv", index=False)
        pathology_summary_df.to_csv(calibration_dir / "pathology_calibration_summary.csv", index=False)
        bin_summary_df.to_csv(calibration_dir / "bin_calibration_summary.csv", index=False)
        calibration_curve_df.to_csv(calibration_dir / "calibration_curve_summary.csv", index=False)
        baseline_summary_df.to_csv(calibration_dir / "baseline_calibration_summary.csv", index=False)

        for (domain, target_source), plot_df in baseline_long.groupby(["domain", "target_source"], sort=True):
            plot_path = calibration_dir / f"reliability_{domain}_{target_source}.png"
            _plot_reliability_diagram(
                plot_df,
                output_path=plot_path,
                title=f"{baseline_name}: {domain} / {target_source}",
                n_bins=n_bins,
                strategy="quantile",
            )

        all_pathology_summaries.append(pathology_summary_df)
        all_baseline_summaries.append(baseline_summary_df)
        all_calibration_curves.append(calibration_curve_df)

    global_pathology_summary = pd.concat(all_pathology_summaries, ignore_index=True)
    global_baseline_summary = pd.concat(all_baseline_summaries, ignore_index=True)
    global_calibration_curve_summary = (
        pd.concat(all_calibration_curves, ignore_index=True) if all_calibration_curves else pd.DataFrame()
    )

    pathology_summary_path = global_dir / "all_baselines_pathology_calibration_summary.csv"
    baseline_summary_path = global_dir / "all_baselines_baseline_calibration_summary.csv"
    calibration_curve_summary_path = global_dir / "all_baselines_calibration_curve_summary.csv"

    if merge_global:
        global_pathology_summary = _merge_by_baseline(
            pathology_summary_path,
            global_pathology_summary,
            baseline_names=baseline_names,
        )
        global_baseline_summary = _merge_by_baseline(
            baseline_summary_path,
            global_baseline_summary,
            baseline_names=baseline_names,
        )
        global_calibration_curve_summary = _merge_by_baseline(
            calibration_curve_summary_path,
            global_calibration_curve_summary,
            baseline_names=baseline_names,
        )

    global_pathology_summary.to_csv(pathology_summary_path, index=False)
    global_baseline_summary.to_csv(baseline_summary_path, index=False)
    global_calibration_curve_summary.to_csv(calibration_curve_summary_path, index=False)

    _plot_summary_scatter(
        global_baseline_summary,
        output_path=global_dir / "summary_scatter_ece_vs_brier.png",
        metric_x="mean_ece_quantile",
        metric_y="mean_brier_score",
        title="Calibration Summary: Adaptive ECE vs Brier Score",
    )
    _plot_summary_scatter(
        global_baseline_summary,
        output_path=global_dir / "summary_scatter_ece_vs_logloss.png",
        metric_x="mean_ece_quantile",
        metric_y="mean_log_loss",
        title="Calibration Summary: Adaptive ECE vs Log-Loss",
    )

    return {
        "global_dir": str(global_dir),
        "pathology_summary_path": str(pathology_summary_path),
        "baseline_summary_path": str(baseline_summary_path),
        "calibration_curve_summary_path": str(calibration_curve_summary_path),
    }


def main() -> None:
    args = _parse_args()
    result = analyze_baselines(
        baseline_names=list(args.baselines),
        output_root=Path(args.output_root),
        run_filters=set(args.runs) if args.runs else None,
        pathology_filter=set(args.pathologies) if args.pathologies else None,
        n_bins=int(args.n_bins),
        min_bin_count=int(args.min_bin_count),
        merge_global=bool(args.merge_global),
    )
    print(
        json.dumps(
            {
                "global_dir": result["global_dir"],
                "pathology_summary_path": result["pathology_summary_path"],
                "baseline_summary_path": result["baseline_summary_path"],
                "calibration_curve_summary_path": result["calibration_curve_summary_path"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
