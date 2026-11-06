from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "classification_log"
DEFAULT_CLASSIFICATION_DATA_ROOT = PACKAGE_ROOT / "classification_data"
DEFAULT_ISP_ROOT = REPO_ROOT / "isp_dataset"
DOCTOR_PATTERN = re.compile(r"doctor(\d+)_summary\.csv$")
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


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    train_overrides: tuple[str, ...]


EXPERIMENT_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="resnet1d18",
        train_overrides=(
            "model=resnet1d18",
            "ecg.resampled_frequency=500",
            "ecg.normalization=z_norm",
        ),
    ),
    ExperimentSpec(name="ecgfounder_linear_probe", train_overrides=("baseline=ecgfounder_linear_probe",)),
    ExperimentSpec(name="ecgjepa_linear_probe", train_overrides=("baseline=ecgjepa_linear_probe",)),
    ExperimentSpec(name="stmem_linear_probe", train_overrides=("baseline=stmem_linear_probe",)),
    ExperimentSpec(name="merl_linear_probe", train_overrides=("baseline=merl_linear_probe",)),
    ExperimentSpec(name="hubert_ecg_linear_probe", train_overrides=("baseline=hubert_ecg_linear_probe",)),
    ExperimentSpec(name="ecg_cpc_block_s4_1", train_overrides=("baseline=ecg_cpc_block_s4_1",)),
    ExperimentSpec(name="s4_scratch", train_overrides=("baseline=s4_scratch",)),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final PTB-XL -> ISP/PTB-XL classification experiments.")
    parser.add_argument("--device-id", required=True, help="Value passed to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--batch-size", type=int, required=True, help="Training batch size override.")
    parser.add_argument("--num-runs", type=int, required=True, help="Number of repeated runs per baseline.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional custom output root. When omitted, the default final classification directory is used.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional subset of experiment names from EXPERIMENT_SPECS.",
    )
    return parser.parse_args()


def _run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")

def _resolve_output_root(output_root_arg: str | None, experiment_specs: tuple[ExperimentSpec, ...]) -> Path:
    if output_root_arg:
        return Path(output_root_arg)
    return DEFAULT_OUTPUT_ROOT


def _read_summary_values(summary_path: Path) -> pd.Series:
    summary = pd.read_csv(summary_path, index_col=0)
    if "value" not in summary.columns:
        raise RuntimeError(f"Unexpected summary format: {summary_path}")
    return summary["value"]


def _sample_std(values: list[float]) -> float:
    finite_values = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
    if finite_values.size <= 1:
        return 0.0
    return float(finite_values.std(ddof=1))


def _aggregate_doctors_vs_consensus(run_tables: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(run_tables, ignore_index=True)
    rows = []
    for (doctor, metric), group in combined.groupby(["doctor", "metric"], sort=True):
        rows.append(
            {
                "doctor": doctor,
                "metric": metric,
                "n_records": float(group["n_records"].mean()),
                "Doctor": float(group["Doctor"].mean()),
                "Doctor STD": _sample_std(group["Doctor"].tolist()),
                "Matched Consensus": float(group["Matched Consensus"].mean()),
                "Matched Consensus STD": _sample_std(group["Matched Consensus"].tolist()),
                "Doctor - Consensus": float(group["Doctor - Consensus"].mean()),
                "Doctor - Consensus STD": _sample_std(group["Doctor - Consensus"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_panel_weighted(run_tables: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat([table.reset_index() for table in run_tables], ignore_index=True)
    rows = []
    for metric, group in combined.groupby("metric", sort=True):
        rows.append(
            {
                "metric": metric,
                "Consensus": float(group["Consensus"].mean()),
                "Consensus STD": _sample_std(group["Consensus"].tolist()),
                "Doctor Mean": float(group["Doctor Mean"].mean()),
                "Doctor Mean STD": _sample_std(group["Doctor Mean"].tolist()),
                "Highest Expert": float(group["Highest Expert"].mean()),
                "Highest Expert STD": _sample_std(group["Highest Expert"].tolist()),
                "Panel Count": float(group["Panel Count"].mean()),
                "Record Count": float(group["Record Count"].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def _load_doctor_summary(summary_path: Path) -> tuple[int, pd.Series]:
    match = DOCTOR_PATTERN.search(summary_path.name)
    if match is None:
        raise RuntimeError(f"Unexpected doctor summary file name: {summary_path}")
    doctor_index = int(match.group(1))
    return doctor_index, _read_summary_values(summary_path)


def _load_doctor_summaries(metrics_dir: Path) -> dict[int, pd.Series]:
    doctor_candidates = sorted(
        path
        for path in metrics_dir.glob("doctor*_summary.csv")
        if DOCTOR_PATTERN.search(path.name) is not None
    )
    if not doctor_candidates:
        raise RuntimeError(f"No doctor summaries found in {metrics_dir}")

    return dict(_load_doctor_summary(path) for path in doctor_candidates)


def _select_best_doctor_for_metric(
    doctor_summaries: dict[int, pd.Series],
    metric_name: str,
) -> tuple[int | None, float]:
    best_doctor_index: int | None = None
    best_value: float | None = None

    for doctor_index in sorted(doctor_summaries):
        summary = doctor_summaries[doctor_index]
        if metric_name not in summary.index:
            raise RuntimeError(f"Metric '{metric_name}' is missing for doctor{doctor_index}")
        value = float(summary.loc[metric_name])
        if not np.isfinite(value):
            continue
        if best_value is None or value > best_value:
            best_doctor_index = doctor_index
            best_value = value

    if best_doctor_index is None or best_value is None:
        return None, float("nan")
    return best_doctor_index, best_value


def _build_final_summary(run_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for metric_name in SUMMARY_METRICS:
        doctor_values = np.asarray([float(row["doctor_mean"][metric_name]) for row in run_rows], dtype=float)
        consensus_values = np.asarray([float(row["consensus"][metric_name]) for row in run_rows], dtype=float)
        ptbxl_values = np.asarray([float(row["ptbxl_test"][metric_name]) for row in run_rows], dtype=float)
        selected_doctors = []
        highest_expert_values = []
        for row in run_rows:
            doctor_index, value = _select_best_doctor_for_metric(
                row["doctor_summaries"],
                metric_name,
            )
            selected_doctors.append(doctor_index)
            highest_expert_values.append(value)
        highest_expert_values_array = np.asarray(highest_expert_values, dtype=float)
        finite_highest_expert_values = highest_expert_values_array[np.isfinite(highest_expert_values_array)]
        highest_expert_mean = (
            float(finite_highest_expert_values.mean())
            if finite_highest_expert_values.size
            else float("nan")
        )
        highest_expert_std = (
            float(finite_highest_expert_values.std(ddof=1))
            if finite_highest_expert_values.size > 1
            else 0.0
        )
        rows.append(
            {
                "metric": metric_name,
                "Doctor Mean": float(doctor_values.mean()),
                "Doctor Mean STD": float(doctor_values.std(ddof=1)) if doctor_values.size > 1 else 0.0,
                "Consensus": float(consensus_values.mean()),
                "Consensus STD": float(consensus_values.std(ddof=1)) if consensus_values.size > 1 else 0.0,
                "PTB-XL Test": float(ptbxl_values.mean()),
                "PTB-XL Test STD": float(ptbxl_values.std(ddof=1)) if ptbxl_values.size > 1 else 0.0,
                "Highest Expert": highest_expert_mean,
                "Highest Expert STD": highest_expert_std,
                "Expert List": json.dumps(selected_doctors),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def _train_command(
    spec: ExperimentSpec,
    train_dir: Path,
    batch_size: int,
    classification_data_root: Path,
    epochs: int,
    lr: float,
    scheduler_patience: int,
    early_stopping_patience: int,
    training_device: str,
    training_num_workers: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "classification.train",
        *spec.train_overrides,
        "pathology_preset=isp_1",
        f"data.root_dir={classification_data_root}",
        f"training.batch_size={batch_size}",
        f"training.epochs={epochs}",
        f"training.lr={lr}",
        f"training.scheduler_patience={scheduler_patience}",
        f"training.early_stopping_patience={early_stopping_patience}",
        f"training.device={training_device}",
        f"training.num_workers={training_num_workers}",
        f"output.dir={train_dir}",
        f"hydra.run.dir={train_dir}",
    ]


def _isp_predict_command(
    checkpoint_path: Path,
    output_dir: Path,
    isp_root: Path,
    predict_device: str,
    predict_pin_memory: bool,
    predict_num_workers: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "classification.predict",
        f"predict.checkpoint_path={checkpoint_path}",
        f"predict.device={predict_device}",
        f"predict.pin_memory={str(predict_pin_memory)}",
        f"predict.num_workers={predict_num_workers}",
        f"dataset.root_dir={isp_root}",
        f"dataset.signal_root={isp_root / 'data'}",
        f"dataset.map_file={isp_root / 'map_file.csv'}",
        f"output.dir={output_dir}",
        f"hydra.run.dir={output_dir}",
    ]


def _ptbxl_predict_command(
    checkpoint_path: Path,
    output_dir: Path,
    predict_device: str,
    predict_pin_memory: bool,
    predict_num_workers: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "classification.predict_ptbxl_test",
        f"predict.checkpoint_path={checkpoint_path}",
        f"predict.device={predict_device}",
        f"predict.pin_memory={str(predict_pin_memory)}",
        f"predict.num_workers={predict_num_workers}",
        f"output.dir={output_dir}",
        f"hydra.run.dir={output_dir}",
    ]


def run_experiments(
    *,
    device_id: str,
    batch_size: int,
    num_runs: int,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    classification_data_root: Path = DEFAULT_CLASSIFICATION_DATA_ROOT,
    isp_root: Path = DEFAULT_ISP_ROOT,
    epochs: int = 30,
    lr: float = 0.001,
    scheduler_patience: int = 3,
    early_stopping_patience: int = 10,
    training_device: str = "cuda",
    predict_device: str = "cuda",
    predict_pin_memory: bool = True,
    training_num_workers: int = 4,
    predict_num_workers: int = 4,
    experiment_specs: tuple[ExperimentSpec, ...] = EXPERIMENT_SPECS,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    manifests: list[dict[str, object]] = []

    for spec in experiment_specs:
        baseline_root = output_root / spec.name
        baseline_root.mkdir(parents=True, exist_ok=True)
        run_rows: list[dict[str, object]] = []
        doctors_vs_consensus_tables: list[pd.DataFrame] = []
        panel_weighted_tables: list[pd.DataFrame] = []

        for run_index in range(1, num_runs + 1):
            run_name = f"run_{run_index:02d}"
            run_root = baseline_root / run_name
            train_dir = run_root / "train"
            isp_dir = run_root / "isp"
            ptbxl_dir = run_root / "ptbxl_test"
            train_log = run_root / "train.log"
            isp_log = run_root / "isp_predict.log"
            ptbxl_log = run_root / "ptbxl_test_predict.log"
            checkpoint_path = train_dir / "best_model.pt"

            print(f"baseline: {spec.name}, run: {run_index}, phase: train", flush=True)
            _run_command(
                _train_command(
                    spec=spec,
                    train_dir=train_dir,
                    batch_size=batch_size,
                    classification_data_root=classification_data_root,
                    epochs=epochs,
                    lr=lr,
                    scheduler_patience=scheduler_patience,
                    early_stopping_patience=early_stopping_patience,
                    training_device=training_device,
                    training_num_workers=training_num_workers,
                ),
                train_log,
                env,
            )
            print(f"baseline: {spec.name}, run: {run_index}, phase: isp_test", flush=True)
            _run_command(
                _isp_predict_command(
                    checkpoint_path=checkpoint_path,
                    output_dir=isp_dir,
                    isp_root=isp_root,
                    predict_device=predict_device,
                    predict_pin_memory=predict_pin_memory,
                    predict_num_workers=predict_num_workers,
                ),
                isp_log,
                env,
            )
            print(f"baseline: {spec.name}, run: {run_index}, phase: ptbxl_test", flush=True)
            _run_command(
                _ptbxl_predict_command(
                    checkpoint_path=checkpoint_path,
                    output_dir=ptbxl_dir,
                    predict_device=predict_device,
                    predict_pin_memory=predict_pin_memory,
                    predict_num_workers=predict_num_workers,
                ),
                ptbxl_log,
                env,
            )

            doctor_mean = _read_summary_values(isp_dir / "metrics" / "doctors_mean_summary.csv")
            consensus = _read_summary_values(isp_dir / "metrics" / "unified_decision_summary.csv")
            ptbxl_test = _read_summary_values(ptbxl_dir / "metrics" / "test_summary.csv")
            doctor_summaries = _load_doctor_summaries(isp_dir / "metrics")
            doctors_vs_consensus_tables.append(
                pd.read_csv(isp_dir / "metrics" / "doctors_vs_consensus_summary.csv")
            )
            panel_weighted_tables.append(
                pd.read_csv(isp_dir / "metrics" / "panel_weighted_summary.csv", index_col=0)
            )

            run_rows.append(
                {
                    "run_index": run_index,
                    "doctor_mean": doctor_mean,
                    "consensus": consensus,
                    "ptbxl_test": ptbxl_test,
                    "doctor_summaries": doctor_summaries,
                }
            )
            manifests.append(
                {
                    "baseline": spec.name,
                    "run_index": run_index,
                    "checkpoint_path": str(checkpoint_path),
                    "train_output_dir": str(train_dir),
                    "isp_output_dir": str(isp_dir),
                    "ptbxl_test_output_dir": str(ptbxl_dir),
                    "status": "completed",
                }
            )

        final_summary = _build_final_summary(run_rows)
        final_summary.to_csv(baseline_root / "final_summary.csv")
        _aggregate_doctors_vs_consensus(doctors_vs_consensus_tables).to_csv(
            baseline_root / "doctors_vs_consensus_summary.csv",
            index=False,
        )
        _aggregate_panel_weighted(panel_weighted_tables).to_csv(
            baseline_root / "panel_weighted_summary.csv",
        )

    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2))
    return {"output_root": str(output_root), "manifest_path": str(manifest_path)}


def main() -> None:
    args = _parse_args()
    experiment_specs = EXPERIMENT_SPECS
    if args.experiments:
        requested = set(args.experiments)
        available = {spec.name for spec in EXPERIMENT_SPECS}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Unknown experiments requested: {unknown}. Available: {sorted(available)}")
        experiment_specs = tuple(spec for spec in EXPERIMENT_SPECS if spec.name in requested)
    output_root = _resolve_output_root(args.output_root, experiment_specs)
    run_experiments(
        device_id=str(args.device_id),
        batch_size=int(args.batch_size),
        num_runs=int(args.num_runs),
        output_root=output_root,
        experiment_specs=experiment_specs,
    )


if __name__ == "__main__":
    main()
