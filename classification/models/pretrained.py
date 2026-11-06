from __future__ import annotations

import re
import zipfile
from http.cookiejar import CookieJar
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import HTTPCookieProcessor, build_opener, urlretrieve
import shutil
import torch


ECGFOUNDER_CHECKPOINT_URL = (
    "XXXX"
)
ECGFOUNDER_CHECKPOINT_NAME = "12_lead_ECGFounder.pth"
ECGJEPA_CHECKPOINT_DRIVE_ID = "1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l"
ECGJEPA_CHECKPOINT_NAME = "ecg_jepa.pt"
STMEM_ENCODER_CHECKPOINT_DRIVE_ID = "1E7J-A1HqWa2f08T6Sfk5uWk-_CFJhOYQ"
STMEM_ENCODER_CHECKPOINT_NAME = "st_mem_encoder.pth"
MERL_RESNET18_ENCODER_DRIVE_ID = "1NG__hPjm8_5c9BiUmCzjlC2Og6ArCtww"
MERL_RESNET18_ENCODER_CHECKPOINT_NAME = "merl_resnet18_encoder.pth"
HUBERT_ECG_REPO_ID = "Edoardo-BS/hubert-ecg-base"
HUBERT_ECG_MODEL_NAME = "model.safetensors"
HUBERT_ECG_CONFIG_NAME = "config.json"
ECG_CPC_CHECKPOINT_DRIVE_ID = "1flOuBii-rjXn81j0sGjrhrIm08TsRimq"
ECG_CPC_ARCHIVE_NAME = "ECG-CPC Checkpoint.zip"
ECG_CPC_CONFIG_NAME = "config_last_11597276_ckpt.yaml"
ECG_CPC_CHECKPOINT_NAME = "last_11597276.ckpt"


def get_default_pretrained_root() -> Path:
    """Return the default local cache root for public pretrained checkpoints."""

    return Path.cwd() / "classification_data" / "pretrained_checkpoints"


def resolve_ecgfounder_checkpoint_path(configured_path: str | None = None) -> Path:
    """Resolve the local target path for the ECGFounder checkpoint."""

    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return (get_default_pretrained_root() / "ecgfounder" / ECGFOUNDER_CHECKPOINT_NAME).resolve()


def ensure_ecgfounder_checkpoint(configured_path: str | None = None) -> str:
    """Return a local ECGFounder checkpoint path, downloading it when missing."""

    checkpoint_path = resolve_ecgfounder_checkpoint_path(configured_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        print(f"Downloading ECGFounder checkpoint to: {checkpoint_path}")
        urlretrieve(ECGFOUNDER_CHECKPOINT_URL, checkpoint_path)
        print(f"ECGFounder checkpoint downloaded successfully: {checkpoint_path}")
    else:
        source = "configured path" if configured_path else "default cache"
        print(f"Using existing ECGFounder checkpoint from {source}: {checkpoint_path}")
    return str(checkpoint_path)


def _extract_google_drive_confirm_token(cookie_jar: CookieJar, response, payload: bytes) -> str | None:
    """Extract the Google Drive confirmation token required for large file downloads."""

    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            return cookie.value
    response_url = response.geturl()
    query_token = parse_qs(urlparse(response_url).query).get("confirm")
    if query_token:
        return query_token[0]
    html = payload.decode("utf-8", errors="ignore")
    patterns = [
        r'confirm=([0-9A-Za-z_-]+)',
        r'name="confirm"\s+value="([0-9A-Za-z_-]+)"',
        r'"confirm":"([0-9A-Za-z_-]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    return None


def _extract_google_drive_download_form(payload: bytes) -> tuple[str, dict[str, str]] | None:
    """Extract the Google Drive virus-scan confirmation form action and hidden fields."""

    html = payload.decode("utf-8", errors="ignore")
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
        html,
        flags=re.DOTALL,
    )
    if not form_match:
        return None
    action = unescape(form_match.group(1))
    inputs_html = form_match.group(2)
    hidden_inputs = re.findall(
        r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
        inputs_html,
    )
    if not hidden_inputs:
        return None
    return action, {name: unescape(value) for name, value in hidden_inputs}


def _looks_like_html_file(path: Path) -> bool:
    """Return whether a downloaded file looks like an HTML page instead of a checkpoint."""

    if not path.exists():
        return False
    prefix = path.read_bytes()[:512].lstrip()
    return prefix.startswith(b"<!DOCTYPE html") or prefix.startswith(b"<html")


def _download_with_progress(url: str, destination: Path) -> None:
    """Download a file while printing a simple progress indicator."""

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = int(downloaded * 100 / total_size)
        print(f"Download progress: {percent}% ({downloaded}/{total_size} bytes)")

    urlretrieve(url, destination, reporthook=reporthook)


def _download_google_drive_file(file_id: str, destination: Path) -> None:
    """Download a public Google Drive file to a local destination."""

    cookie_jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))
    base_url = "XXXX"
    initial_url = f"{base_url}?{urlencode({'export': 'download', 'id': file_id})}"
    with opener.open(initial_url) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            with destination.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
            return
        payload = response.read()
    download_form = _extract_google_drive_download_form(payload)
    if download_form is not None:
        action, params = download_form
        with opener.open(f"{action}?{urlencode(params)}") as response, destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
        return
    confirm_token = _extract_google_drive_confirm_token(cookie_jar, response, payload)
    if confirm_token is None:
        raise RuntimeError("Failed to resolve the Google Drive confirmation token.")
    confirmed_url = f"{base_url}?{urlencode({'export': 'download', 'confirm': confirm_token, 'id': file_id})}"
    with opener.open(confirmed_url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def resolve_ecgjepa_checkpoint_path(configured_path: str | None = None) -> Path:
    """Resolve the local target path for the ECG-JEPA checkpoint."""

    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return (Path.cwd() / "classification_data" / ECGJEPA_CHECKPOINT_NAME).resolve()


def ensure_ecgjepa_checkpoint(configured_path: str | None = None) -> str:
    """Return a local ECG-JEPA checkpoint path, downloading it when missing."""

    checkpoint_path = resolve_ecgjepa_checkpoint_path(configured_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    should_download = not checkpoint_path.exists() or _looks_like_html_file(checkpoint_path)
    if should_download:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"Downloading ECG-JEPA checkpoint to: {checkpoint_path}")
        _download_google_drive_file(ECGJEPA_CHECKPOINT_DRIVE_ID, checkpoint_path)
        if _looks_like_html_file(checkpoint_path):
            checkpoint_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded ECG-JEPA checkpoint is not a valid checkpoint file: {checkpoint_path}"
            )
        print(f"ECG-JEPA checkpoint downloaded successfully: {checkpoint_path}")
    else:
        source = "configured path" if configured_path else "default path"
        print(f"Using existing ECG-JEPA checkpoint from {source}: {checkpoint_path}")
    return str(checkpoint_path)


def resolve_stmem_checkpoint_path(configured_path: str | None = None) -> Path:
    """Resolve the local target path for the ST-MEM encoder checkpoint."""

    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return (Path.cwd() / "classification_data" / STMEM_ENCODER_CHECKPOINT_NAME).resolve()


def ensure_stmem_checkpoint(configured_path: str | None = None) -> str:
    """Return a local ST-MEM encoder checkpoint path, downloading it when missing."""

    checkpoint_path = resolve_stmem_checkpoint_path(configured_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    should_download = not checkpoint_path.exists() or _looks_like_html_file(checkpoint_path)
    if should_download:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"Downloading ST-MEM checkpoint to: {checkpoint_path}")
        _download_google_drive_file(STMEM_ENCODER_CHECKPOINT_DRIVE_ID, checkpoint_path)
        if _looks_like_html_file(checkpoint_path):
            checkpoint_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded ST-MEM checkpoint is not a valid checkpoint file: {checkpoint_path}"
            )
        print(f"ST-MEM checkpoint downloaded successfully: {checkpoint_path}")
    else:
        source = "configured path" if configured_path else "default path"
        print(f"Using existing ST-MEM checkpoint from {source}: {checkpoint_path}")
    return str(checkpoint_path)


def resolve_merl_checkpoint_path(configured_path: str | None = None) -> Path:
    """Resolve the local target path for the MERL ResNet18 encoder checkpoint."""

    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return (Path.cwd() / "classification_data" / MERL_RESNET18_ENCODER_CHECKPOINT_NAME).resolve()


def ensure_merl_checkpoint(configured_path: str | None = None) -> str:
    """Return a local MERL ResNet18 encoder checkpoint path, downloading it when missing."""

    checkpoint_path = resolve_merl_checkpoint_path(configured_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    should_download = not checkpoint_path.exists() or _looks_like_html_file(checkpoint_path)
    if should_download:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"Downloading MERL checkpoint to: {checkpoint_path}")
        _download_google_drive_file(MERL_RESNET18_ENCODER_DRIVE_ID, checkpoint_path)
        if _looks_like_html_file(checkpoint_path):
            checkpoint_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded MERL checkpoint is not a valid checkpoint file: {checkpoint_path}"
            )
        print(f"MERL checkpoint downloaded successfully: {checkpoint_path}")
    else:
        source = "configured path" if configured_path else "default path"
        print(f"Using existing MERL checkpoint from {source}: {checkpoint_path}")
    return str(checkpoint_path)


def resolve_hubert_ecg_checkpoint_dir(configured_path: str | None = None) -> Path:
    """Resolve the local directory for the HuBERT-ECG checkpoint files."""

    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        return path if path.is_dir() else path.parent
    return (get_default_pretrained_root() / "hubert_ecg").resolve()


def ensure_hubert_ecg_checkpoint(configured_path: str | None = None) -> str:
    """Return a local HuBERT-ECG checkpoint directory, downloading it when missing."""

    checkpoint_dir = resolve_hubert_ecg_checkpoint_dir(configured_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / HUBERT_ECG_MODEL_NAME
    config_path = checkpoint_dir / HUBERT_ECG_CONFIG_NAME
    should_download = not model_path.exists() or not config_path.exists() or _looks_like_html_file(model_path)
    if should_download:
        if model_path.exists():
            model_path.unlink()
        if config_path.exists():
            config_path.unlink()
        print(f"Downloading HuBERT-ECG checkpoint to: {checkpoint_dir}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            snapshot_download = None
        if snapshot_download is not None:
            snapshot_download(
                repo_id=HUBERT_ECG_REPO_ID,
                local_dir=str(checkpoint_dir),
                local_dir_use_symlinks=False,
                allow_patterns=[HUBERT_ECG_MODEL_NAME, HUBERT_ECG_CONFIG_NAME],
            )
        else:
            base_url = f"XXXX{HUBERT_ECG_REPO_ID}/resolve/main"
            urlretrieve(f"{base_url}/{HUBERT_ECG_MODEL_NAME}", model_path)
            urlretrieve(f"{base_url}/{HUBERT_ECG_CONFIG_NAME}", config_path)
        if _looks_like_html_file(model_path):
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded HuBERT-ECG checkpoint is not a valid checkpoint file: {model_path}"
            )
        print(f"HuBERT-ECG checkpoint downloaded successfully: {checkpoint_dir}")
    else:
        source = "configured path" if configured_path else "default path"
        print(f"Using existing HuBERT-ECG checkpoint from {source}: {checkpoint_dir}")
    return str(model_path if configured_path and Path(configured_path).is_file() else checkpoint_dir)


def resolve_ecg_cpc_checkpoint_path(configured_path: str | None = None) -> Path:
    """Resolve the local target path for the extracted ECG-CPC checkpoint."""

    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return (get_default_pretrained_root() / "ecg_cpc" / ECG_CPC_CHECKPOINT_NAME).resolve()


def _extract_ecg_cpc_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract the ECG-CPC archive and return the normalized `.ckpt` path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        checkpoint_member = next((name for name in members if name.endswith(ECG_CPC_CHECKPOINT_NAME)), None)
        if checkpoint_member is None:
            raise RuntimeError(f"ECG-CPC archive does not contain {ECG_CPC_CHECKPOINT_NAME}.")
        archive.extract(checkpoint_member, path=output_dir)
        config_member = next((name for name in members if name.endswith(ECG_CPC_CONFIG_NAME)), None)
        if config_member is not None:
            archive.extract(config_member, path=output_dir)

        extracted_checkpoint = (output_dir / checkpoint_member).resolve()
        normalized_checkpoint = (output_dir / ECG_CPC_CHECKPOINT_NAME).resolve()
        if extracted_checkpoint != normalized_checkpoint:
            normalized_checkpoint.write_bytes(extracted_checkpoint.read_bytes())
        return normalized_checkpoint


def _normalize_ecg_cpc_artifact(artifact_path: Path, output_dir: Path) -> Path:
    """Normalize a downloaded or manually placed ECG-CPC artifact into a `.ckpt` path."""

    if artifact_path.suffix == ".ckpt":
        return artifact_path
    if zipfile.is_zipfile(artifact_path):
        return _extract_ecg_cpc_archive(artifact_path, output_dir)
    try:
        checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive path for invalid downloads
        raise RuntimeError(f"ECG-CPC artifact is neither a valid zip archive nor a checkpoint: {artifact_path}") from exc
    if "state_dict" not in checkpoint:
        raise RuntimeError(f"ECG-CPC checkpoint does not contain a state_dict entry: {artifact_path}")
    normalized_checkpoint = (output_dir / ECG_CPC_CHECKPOINT_NAME).resolve()
    normalized_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.resolve() != normalized_checkpoint:
        normalized_checkpoint.write_bytes(artifact_path.read_bytes())
    return normalized_checkpoint


def ensure_ecg_cpc_checkpoint(configured_path: str | None = None) -> str:
    """Return a local ECG-CPC checkpoint path, downloading and extracting it when missing."""

    if configured_path:
        resolved_path = Path(configured_path).expanduser().resolve()
        if resolved_path.suffix in {".ckpt", ".zip"}:
            return str(_normalize_ecg_cpc_artifact(resolved_path, resolved_path.parent / "ecg_cpc"))
        raise ValueError("ECG-CPC configured checkpoint path must point to either a .zip archive or a .ckpt file.")

    checkpoint_path = resolve_ecg_cpc_checkpoint_path(None)
    archive_path = checkpoint_path.parent / ECG_CPC_ARCHIVE_NAME
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        print(f"Using existing ECG-CPC checkpoint from default cache: {checkpoint_path}")
        return str(checkpoint_path)
    if archive_path.exists() and archive_path.stat().st_size > 0:
        checkpoint_path = _normalize_ecg_cpc_artifact(archive_path, checkpoint_path.parent)
        print(f"Using existing ECG-CPC checkpoint archive from default cache: {checkpoint_path}")
        return str(checkpoint_path)
    if archive_path.exists():
        archive_path.unlink()
    print(f"Downloading ECG-CPC checkpoint to: {archive_path}")
    _download_google_drive_file(ECG_CPC_CHECKPOINT_DRIVE_ID, archive_path)
    checkpoint_path = _normalize_ecg_cpc_artifact(archive_path, checkpoint_path.parent)
    print(f"ECG-CPC checkpoint downloaded successfully: {checkpoint_path}")
    return str(checkpoint_path)
