from __future__ import annotations

import ast
import json
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb


def _parse_scp_codes(value) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return [str(key) for key in parsed]
    raise ValueError(f"Unsupported PTB-XL scp_codes format: {value!r}")


def _default_zip_url(version: str) -> str:
    return f"XXXX{version}/"


def _download_zip(zip_url: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(zip_url, zip_path)


def _zip_candidates(root_dir: Path, raw_dir: Path, configured_zip_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    if configured_zip_path:
        candidates.append(Path(configured_zip_path).expanduser().resolve())
    candidates.extend(
        [
            root_dir / "ptbxl" / "raw.zip",
            raw_dir / "raw.zip",
        ]
    )
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _extract_zip_archive(zip_path: Path, raw_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_dir)


def _resolve_metadata_paths(raw_dir: Path) -> tuple[Path | None, Path | None]:
    metadata_candidates = sorted(raw_dir.rglob("ptbxl_database.csv"))
    scp_candidates = sorted(raw_dir.rglob("scp_statements.csv"))
    metadata_path = metadata_candidates[0] if metadata_candidates else None
    scp_path = scp_candidates[0] if scp_candidates else None
    return metadata_path, scp_path


def _resolve_split(strat_fold: int) -> str:
    if strat_fold <= 8:
        return "train"
    if strat_fold == 9:
        return "val"
    return "test"


def _build_row(metadata_row: pd.Series, relative_npz_path: str, signal: np.ndarray, source_frequency: int) -> dict:
    scp_codes = _parse_scp_codes(metadata_row["scp_codes"])
    record_id = int(metadata_row["ecg_id"]) if "ecg_id" in metadata_row else int(metadata_row.name)
    return {
        "record_id": record_id,
        "patient_id": int(metadata_row["patient_id"]),
        "file_name": relative_npz_path,
        "array_name": "arr_0",
        "sampling_rate": int(source_frequency),
        "signal_length": int(signal.shape[1]),
        "scp_codes": json.dumps(scp_codes),
        "strat_fold": int(metadata_row["strat_fold"]),
        "split": _resolve_split(int(metadata_row["strat_fold"])),
    }


def prepare_ptbxl_dataset(config: dict) -> dict[str, str]:
    root_dir = Path(config["data"]["root_dir"]).resolve()
    raw_dir = root_dir / "ptbxl" / "raw"
    prepared_dir = root_dir / "ptbxl" / "prepared"
    npz_dir = prepared_dir / "npz"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    configured_zip_path = config["prepare"].get("raw_zip_path")
    zip_candidates = _zip_candidates(root_dir, raw_dir, configured_zip_path)
    zip_path = zip_candidates[0]

    if config["prepare"]["download"]:
        zip_url = config["prepare"].get("zip_url") or _default_zip_url(config["prepare"]["ptbxl_version"])
        if not zip_path.exists():
            _download_zip(zip_url=zip_url, zip_path=zip_path)
    else:
        existing_zip = next((candidate for candidate in zip_candidates if candidate.exists()), None)
        if existing_zip is None:
            raise FileNotFoundError(
                "PTB-XL raw.zip was not found. Provide it in the default location "
                "or set prepare.raw_zip_path explicitly."
            )
        zip_path = existing_zip

    metadata_path, scp_path = _resolve_metadata_paths(raw_dir)
    if metadata_path is None or scp_path is None:
        _extract_zip_archive(zip_path=zip_path, raw_dir=raw_dir)
        metadata_path, scp_path = _resolve_metadata_paths(raw_dir)

    if metadata_path is None or scp_path is None:
        raise FileNotFoundError(
            "PTB-XL metadata files are missing after raw.zip extraction."
        )

    metadata = pd.read_csv(metadata_path)
    dataset_root = metadata_path.parent
    source_frequency = int(config["prepare"]["source_frequency"])
    source_column = "filename_hr" if source_frequency == 500 else "filename_lr"
    if source_column not in metadata.columns:
        raise KeyError(f"PTB-XL metadata does not contain '{source_column}'")

    rows: list[dict] = []
    progress_bar = tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Converting PTB-XL to NPZ",
    )
    for _, metadata_row in progress_bar:
        record_rel_path = metadata_row[source_column]
        signal, _ = wfdb.rdsamp(str(dataset_root / record_rel_path))
        signal = signal.T.astype(np.float32)
        record_id = int(metadata_row["ecg_id"])
        npz_path = npz_dir / f"{record_id}.npz"
        np.savez_compressed(npz_path, arr_0=signal)
        rows.append(
            _build_row(
                metadata_row=metadata_row,
                relative_npz_path=str(Path("npz") / npz_path.name),
                signal=signal,
                source_frequency=source_frequency,
            )
        )

    prepared = pd.DataFrame(rows)
    output_paths = {}
    for split in ("train", "val", "test"):
        split_path = prepared_dir / f"{split}_map_ptbxl.csv"
        prepared[prepared["split"] == split].to_csv(split_path, index=False)
        output_paths[split] = str(split_path)
    return output_paths
