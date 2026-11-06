from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification.data.datasets import ECGClassificationDataset, load_ptbxl_dataframe
from classification.data.labels import get_pathology_preset
from classification.data.preprocess import ECGPreprocessor
from classification.models import (
    create_model,
    ensure_ecg_cpc_checkpoint,
    ensure_ecgfounder_checkpoint,
    ensure_ecgjepa_checkpoint,
    ensure_hubert_ecg_checkpoint,
    ensure_merl_checkpoint,
    ensure_stmem_checkpoint,
    freeze_backbone,
    freeze_ecg_cpc_backbone,
    freeze_ecgjepa_backbone,
    freeze_hubert_ecg_backbone,
    freeze_merl_backbone,
    freeze_stmem_backbone,
    load_ecg_cpc_pretrained_state_dict,
    load_ecgjepa_pretrained_state_dict,
    load_hubert_ecg_pretrained_state_dict,
    load_merl_pretrained_state_dict,
    load_pretrained_state_dict,
    load_stmem_pretrained_state_dict,
)
from .checkpoint import save_checkpoint
from .metrics import compute_aggregated_metrics, compute_metrics, select_best_validation_threshold, sigmoid
from .windowing import get_full_signal_length_seconds, run_model_with_optional_sliding_windows, sliding_window_enabled


def _freeze_all_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _explicit_group_name(model_name: str, parameter_name: str) -> str:
    if model_name == "ecg_cpc":
        if parameter_name.startswith("head."):
            return "head"
        match = re.match(r"ts_encoder\.predictor\.predictor\.norms\.(\d+)\.", parameter_name)
        if match:
            return f"predictor_norm{match.group(1)}"
        match = re.match(r"ts_encoder\.predictor\.predictor\.s4_layers\.(\d+)\.", parameter_name)
        if match:
            return f"s4_layer{match.group(1)}"
        match = re.match(r"ts_encoder\.encoder\.layers\.(\d+)\.", parameter_name)
        if match:
            return f"conv{match.group(1)}"
    return parameter_name.split(".", 1)[0]


def _unfreeze_explicit_groups(
    model,
    model_name: str,
    group_names: list[str],
) -> tuple[int, list[str], list[tuple[str, int]]]:
    if not group_names:
        raise ValueError("training.trainable_groups must contain at least one group name.")

    _freeze_all_parameters(model)

    grouped_parameters: dict[str, list[tuple[str, torch.nn.Parameter]]] = {}
    for name, parameter in model.named_parameters():
        group_name = _explicit_group_name(model_name, name)
        grouped_parameters.setdefault(group_name, []).append((name, parameter))

    missing_groups = [group_name for group_name in group_names if group_name not in grouped_parameters]
    if missing_groups:
        available_groups = ", ".join(sorted(grouped_parameters))
        raise ValueError(
            "Unknown training.trainable_groups requested: "
            f"{missing_groups}. Available groups for model={model_name}: {available_groups}"
        )

    selected_names: list[str] = []
    selected_groups: list[tuple[str, int]] = []
    selected_parameter_count = 0
    for group_name in group_names:
        group_count = 0
        for name, parameter in grouped_parameters[group_name]:
            parameter.requires_grad = True
            selected_names.append(name)
            group_count += int(parameter.numel())
        selected_groups.append((group_name, group_count))
        selected_parameter_count += group_count

    return selected_parameter_count, selected_names, selected_groups


def _build_preprocessor(
    cfg: dict,
    observed_length_seconds_override: float | None = None,
    normalization_override: str | None = None,
    crop_mode_override: str | None = None,
    filter_override: dict | None = None,
    normalization_stats: dict | None = None,
) -> ECGPreprocessor:
    """Create the ECG preprocessing pipeline from Hydra config."""

    return ECGPreprocessor(
        leads=list(cfg["ecg"]["leads"]),
        resampled_frequency=int(cfg["ecg"]["resampled_frequency"]),
        observed_length_seconds=float(
            cfg["ecg"]["observed_length_seconds"]
            if observed_length_seconds_override is None
            else observed_length_seconds_override
        ),
        normalization=(
            cfg["ecg"]["normalization"]
            if normalization_override is None
            else normalization_override
        ),
        normalization_params=dict(cfg["ecg"].get("normalization_params", {})),
        crop_mode=cfg["ecg"].get("crop_mode", "first") if crop_mode_override is None else crop_mode_override,
        filter_cfg=dict(cfg["ecg"].get("filter", {})) if filter_override is None else dict(filter_override),
        normalization_stats=normalization_stats,
        lead_swap=cfg["ecg"].get("lead_swap"),
    )


def _build_dataset(
    cfg: dict,
    split: str,
    preset: dict,
    normalization_stats: dict | None = None,
) -> ECGClassificationDataset:
    dataset_cfg = cfg["dataset"]
    frame = load_ptbxl_dataframe(dataset_cfg["split_files"][split])
    observed_length_seconds_override = None
    normalization_override = None
    crop_mode_override = None
    filter_override = None
    if split == "val" and sliding_window_enabled(cfg, stage="validation"):
        observed_length_seconds_override = get_full_signal_length_seconds(cfg)
        normalization_override = "none"
        crop_mode_override = "first"
        filter_override = {"name": "none", "params": {}}
    preprocessor = _build_preprocessor(
        cfg,
        observed_length_seconds_override=observed_length_seconds_override,
        normalization_override=normalization_override,
        crop_mode_override=crop_mode_override,
        filter_override=filter_override,
        normalization_stats=normalization_stats,
    )
    return ECGClassificationDataset(
        dataframe=frame,
        signal_root=dataset_cfg["signal_root"],
        backend=dataset_cfg["backend"],
        pathology_names=preset["pathology_names"],
        merge_map=preset["merge_map"],
        label_column=dataset_cfg["label_column"],
        preprocessor=preprocessor,
    )


def _pos_weight_from_targets(targets: np.ndarray) -> torch.Tensor | None:
    positives = targets.sum(axis=0)
    negatives = targets.shape[0] - positives
    if np.any(positives == 0):
        return None
    return torch.as_tensor(negatives / positives, dtype=torch.float32)


def _compute_normalization_stats(cfg: dict, preset: dict) -> dict | None:
    """Compute train-split normalization statistics for global mean/std normalization."""

    if cfg["ecg"]["normalization"] not in ("global_mean_std", "global_scalar_std"):
        return None

    dataset_cfg = cfg["dataset"]
    frame = load_ptbxl_dataframe(dataset_cfg["split_files"]["train"])
    observed_length_seconds_override = None
    if sliding_window_enabled(cfg, stage="validation") or sliding_window_enabled(cfg, stage="prediction"):
        observed_length_seconds_override = get_full_signal_length_seconds(cfg)
    preprocessor = _build_preprocessor(
        cfg,
        observed_length_seconds_override=observed_length_seconds_override,
        normalization_override="none",
        crop_mode_override="first",
    )
    dataset = ECGClassificationDataset(
        dataframe=frame,
        signal_root=dataset_cfg["signal_root"],
        backend=dataset_cfg["backend"],
        pathology_names=preset["pathology_names"],
        merge_map=preset["merge_map"],
        label_column=dataset_cfg["label_column"],
        preprocessor=preprocessor,
    )

    channel_sum = None
    channel_sum_sq = None
    total_samples = 0
    scalar_sum = 0.0
    scalar_sum_sq = 0.0
    for index in range(len(dataset)):
        signal = dataset[index]["signal"].numpy().astype(np.float64)
        if cfg["ecg"]["normalization"] == "global_mean_std":
            if channel_sum is None:
                channel_sum = np.zeros(signal.shape[0], dtype=np.float64)
                channel_sum_sq = np.zeros(signal.shape[0], dtype=np.float64)
            channel_sum += signal.sum(axis=1)
            channel_sum_sq += np.square(signal).sum(axis=1)
            total_samples += signal.shape[1]
        else:
            scalar_sum += float(signal.sum())
            scalar_sum_sq += float(np.square(signal).sum())
            total_samples += signal.size

    if total_samples == 0:
        raise RuntimeError("Failed to compute ECG normalization statistics from the PTB-XL train split.")

    if cfg["ecg"]["normalization"] == "global_mean_std":
        if channel_sum is None or channel_sum_sq is None:
            raise RuntimeError("Failed to compute ECG normalization statistics from the PTB-XL train split.")
        mean = channel_sum / total_samples
        variance = np.maximum(channel_sum_sq / total_samples - np.square(mean), 1e-12)
        std = np.sqrt(variance)
        return {
            "normalization_stats": {
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            }
        }

    scalar_mean = scalar_sum / total_samples
    scalar_var = max(scalar_sum_sq / total_samples - scalar_mean**2, 1e-12)
    scalar_std = float(np.sqrt(scalar_var))
    return {
        "normalization_stats": {
            "mean": float(scalar_mean),
            "std": float(scalar_std),
        }
    }


def _initialize_model_for_training(cfg: dict, num_classes: int, input_channels: int):
    """Instantiate a model and optionally load pretrained public weights."""

    model_cfg = cfg["model"]
    model = create_model(
        model_name=model_cfg["name"],
        model_params=dict(model_cfg.get("params", {})),
        num_classes=num_classes,
        input_channels=input_channels,
    )
    pretrained_checkpoint_path = model_cfg.get("pretrained_checkpoint_path")
    if model_cfg["name"] == "ecg_cpc":
        pretrained_checkpoint_path = ensure_ecg_cpc_checkpoint(pretrained_checkpoint_path)
    elif model_cfg["name"] == "ecgfounder":
        pretrained_checkpoint_path = ensure_ecgfounder_checkpoint(pretrained_checkpoint_path)
    elif model_cfg["name"] == "ecgjepa":
        pretrained_checkpoint_path = ensure_ecgjepa_checkpoint(pretrained_checkpoint_path)
    elif model_cfg["name"] == "stmem":
        pretrained_checkpoint_path = ensure_stmem_checkpoint(pretrained_checkpoint_path)
    elif model_cfg["name"] == "merl":
        pretrained_checkpoint_path = ensure_merl_checkpoint(pretrained_checkpoint_path)
    elif model_cfg["name"] == "hubert_ecg":
        pretrained_checkpoint_path = ensure_hubert_ecg_checkpoint(pretrained_checkpoint_path)
    elif pretrained_checkpoint_path:
        raise ValueError(
            "model.pretrained_checkpoint_path is currently supported only for model=ecg_cpc, model=ecgfounder, model=ecgjepa, model=stmem, model=merl, or model=hubert_ecg."
        )
    if pretrained_checkpoint_path:
        if model_cfg["name"] == "ecg_cpc":
            missing_keys = load_ecg_cpc_pretrained_state_dict(model, pretrained_checkpoint_path)
            expected_missing_prefixes = ("head.",)
        elif model_cfg["name"] == "ecgfounder":
            missing_keys = load_pretrained_state_dict(model, pretrained_checkpoint_path)
            expected_missing_prefixes = ("dense.",)
        elif model_cfg["name"] == "ecgjepa":
            missing_keys = load_ecgjepa_pretrained_state_dict(model, pretrained_checkpoint_path)
            expected_missing_prefixes = ("classifier.",)
        elif model_cfg["name"] == "merl":
            missing_keys = load_merl_pretrained_state_dict(model, pretrained_checkpoint_path)
            expected_missing_prefixes = ("linear.",)
        elif model_cfg["name"] == "hubert_ecg":
            missing_keys = load_hubert_ecg_pretrained_state_dict(model, pretrained_checkpoint_path)
            expected_missing_prefixes = tuple()
        else:
            missing_keys = load_stmem_pretrained_state_dict(model, pretrained_checkpoint_path)
            expected_missing_prefixes = ("head.",)
        if expected_missing_prefixes:
            unexpected_missing = [
                key for key in missing_keys
                if not key.startswith(expected_missing_prefixes)
            ]
            if unexpected_missing:
                raise RuntimeError(
                    f"Missing keys when loading {model_cfg['name']} checkpoint: "
                    + ", ".join(sorted(unexpected_missing))
                )

    trainable_scope = cfg["training"].get("trainable_scope", "all")
    if trainable_scope == "head_only":
        if model_cfg["name"] == "ecg_cpc":
            freeze_ecg_cpc_backbone(model)
        elif model_cfg["name"] == "ecgfounder":
            freeze_backbone(model)
        elif model_cfg["name"] == "ecgjepa":
            freeze_ecgjepa_backbone(model)
        elif model_cfg["name"] == "stmem":
            freeze_stmem_backbone(model)
        elif model_cfg["name"] == "merl":
            freeze_merl_backbone(model)
        elif model_cfg["name"] == "hubert_ecg":
            freeze_hubert_ecg_backbone(model)
        else:
            raise ValueError(
                "training.trainable_scope=head_only is currently supported only for model=ecg_cpc, model=ecgfounder, model=ecgjepa, model=stmem, model=merl, or model=hubert_ecg."
            )
    elif trainable_scope == "group_list":
        group_names = list(cfg["training"].get("trainable_groups", []))
        selected_parameter_count, selected_names, selected_groups = _unfreeze_explicit_groups(
            model,
            model_name=model_cfg["name"],
            group_names=group_names,
        )
        print(
            "Applied explicit group fine-tuning: "
            f"selected={selected_parameter_count} "
            f"groups={len(selected_groups)} "
            f"tensors={len(selected_names)}"
        )
        print("Trainable explicit groups:")
        for group_name, group_count in selected_groups:
            print(f"  - {group_name}: {group_count}")
    elif trainable_scope != "all":
        raise ValueError(f"Unsupported training.trainable_scope '{trainable_scope}'")
    return model


def _maybe_apply_random_crop(signal_batch: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Apply batch-wise random cropping during training when requested."""

    crop_seconds = cfg["training"].get("random_crop_seconds")
    if crop_seconds is None:
        return signal_batch
    crop_size = int(float(crop_seconds) * int(cfg["ecg"]["resampled_frequency"]))
    if crop_size <= 0:
        raise ValueError("training.random_crop_seconds must resolve to a positive crop length.")
    if signal_batch.shape[-1] <= crop_size:
        return signal_batch[..., :crop_size]
    max_start = signal_batch.shape[-1] - crop_size
    starts = torch.randint(0, max_start + 1, (signal_batch.shape[0],), device=signal_batch.device)
    return torch.stack(
        [signal_batch[index, :, start:start + crop_size] for index, start in enumerate(starts.tolist())],
        dim=0,
    )


def _run_epoch(model, data_loader, criterion, optimizer, device: torch.device, cfg: dict):
    model.train()
    running_loss = 0.0
    progress = tqdm(data_loader, total=len(data_loader), desc="Train", leave=False)
    for batch in progress:
        signal = batch["signal"].to(device)
        signal = _maybe_apply_random_crop(signal, cfg)
        target = batch["target"].to(device)
        optimizer.zero_grad()
        logits = model(signal)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        running_loss += loss_value
        progress.set_postfix(loss=f"{loss_value:.4f}")
    return running_loss / max(len(data_loader), 1)


def _evaluate(model, data_loader, criterion, device: torch.device, cfg: dict, preprocess_metadata: dict | None = None):
    model.eval()
    running_loss = 0.0
    logits_list = []
    targets_list = []
    with torch.no_grad():
        progress = tqdm(data_loader, total=len(data_loader), desc="Valid", leave=False)
        for batch in progress:
            signal = batch["signal"].to(device)
            target = batch["target"].to(device)
            logits, _ = run_model_with_optional_sliding_windows(
                model=model,
                signal_batch=signal,
                cfg=cfg,
                stage="validation",
                preprocess_metadata=preprocess_metadata,
            )
            loss = criterion(logits, target)
            loss_value = float(loss.detach().cpu().item())
            running_loss += loss_value
            logits_list.append(logits.detach().cpu().numpy())
            targets_list.append(target.detach().cpu().numpy())
            progress.set_postfix(loss=f"{loss_value:.4f}")
    logits = np.concatenate(logits_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    return running_loss / max(len(data_loader), 1), targets, logits


def run_training(cfg: dict) -> dict:
    preset = get_pathology_preset(cfg["pathology_preset"]["name"])
    print(f"Pathology preset: {preset['preset_name']}")
    print(f"Number of classes: {len(preset['target_labels'])}")
    print("Classifying pathologies:")
    for pathology in preset["target_labels"]:
        print(f"- {pathology}")

    preprocess_metadata = _compute_normalization_stats(cfg, preset)

    train_dataset = _build_dataset(cfg, "train", preset, normalization_stats=(preprocess_metadata or {}).get("normalization_stats"))
    val_dataset = _build_dataset(cfg, "val", preset, normalization_stats=(preprocess_metadata or {}).get("normalization_stats"))

    batch_size = int(cfg["training"]["batch_size"])
    collate_fn = train_dataset.collate_fn
    num_workers = int(cfg["training"]["num_workers"])
    pin_memory = bool(cfg["training"].get("pin_memory", False))
    persistent_workers = bool(cfg["training"].get("persistent_workers", False)) if num_workers > 0 else False
    prefetch_factor = cfg["training"].get("prefetch_factor", 2)
    if num_workers == 0:
        prefetch_factor = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    device_name = cfg["training"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    model = _initialize_model_for_training(
        cfg=cfg,
        num_classes=len(train_dataset.target_labels),
        input_channels=len(cfg["ecg"]["leads"]),
    ).to(device)

    pos_weight = _pos_weight_from_targets(train_dataset.targets)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(cfg["training"]["lr"]))
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        factor=float(cfg["training"]["scheduler_factor"]),
        patience=int(cfg["training"]["scheduler_patience"]),
    )

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / cfg["output"]["checkpoint_name"]
    training_summary_path = output_dir / "training_summary.json"

    best_val_loss = float("inf")
    best_epoch = -1
    best_payload = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(int(cfg["training"]["epochs"])):
        print(f"\nEpoch {epoch + 1}/{int(cfg['training']['epochs'])}")
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, cfg=cfg)
        val_loss, val_targets, val_logits = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg=cfg,
            preprocess_metadata=preprocess_metadata,
        )
        scheduler.step(val_loss)

        thresholds = select_best_validation_threshold(val_targets, val_logits, dict(cfg["training"]["metrics_threshold"]))
        val_probabilities = sigmoid(val_logits)
        val_predictions = (val_probabilities > thresholds).astype(np.float32)
        metrics, _ = compute_metrics(val_targets, val_predictions, val_probabilities, train_dataset.target_labels)
        aggregated_metrics = compute_aggregated_metrics(
            targets=val_targets,
            predictions=val_predictions,
            probabilities=val_probabilities,
            label_names=train_dataset.target_labels,
        )
        mean_f1 = float(metrics.loc["f1-score"].mean())
        current_lr = float(optimizer.param_groups[0]["lr"])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mean_f1": mean_f1,
                "macro_specificity": float(aggregated_metrics.loc["Specificity", "macro"]),
                "micro_specificity": float(aggregated_metrics.loc["Specificity", "micro"]),
                "macro_sensitivity": float(aggregated_metrics.loc["Sensitivity", "macro"]),
                "micro_sensitivity": float(aggregated_metrics.loc["Sensitivity", "micro"]),
                "macro_gmean": float(aggregated_metrics.loc["G-mean", "macro"]),
                "micro_gmean": float(aggregated_metrics.loc["G-mean", "micro"]),
                "macro_roc_auc": float(aggregated_metrics.loc["ROC-AUC", "macro"]),
                "micro_roc_auc": float(aggregated_metrics.loc["ROC-AUC", "micro"]),
                "macro_f1": float(aggregated_metrics.loc["f1-score", "macro"]),
                "micro_f1": float(aggregated_metrics.loc["f1-score", "micro"]),
            }
        )

        improved = val_loss < best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            model_params = dict(cfg["model"].get("params", {}))
            if cfg["model"]["name"] == "hubert_ecg" and hasattr(model, "hubert"):
                if hasattr(model.hubert, "config") and hasattr(model.hubert.config, "to_dict"):
                    model_params["hubert_config"] = model.hubert.config.to_dict()
            best_payload = {
                "state_dict": model.state_dict(),
                "thresholds": thresholds.tolist(),
                "pathology_names": train_dataset.target_labels,
                "pathology_preset": preset["preset_name"],
                "model_name": cfg["model"]["name"],
                "model_params": model_params,
                "input_channels": len(cfg["ecg"]["leads"]),
                "config": cfg,
                "preprocess": preprocess_metadata or {},
                "best_val_loss": val_loss,
            }
            save_checkpoint(checkpoint_path, best_payload)
            training_summary_path.write_text(
                json.dumps(
                    {
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "checkpoint_path": str(checkpoint_path),
                    },
                    indent=2,
                )
            )
        else:
            epochs_without_improvement += 1

        print(f"train_loss: {train_loss:.6f}")
        print(f"val_loss: {val_loss:.6f}")
        print(
            "val_macro: "
            f"specificity={aggregated_metrics.loc['Specificity', 'macro']:.6f}, "
            f"sensitivity={aggregated_metrics.loc['Sensitivity', 'macro']:.6f}, "
            f"gmean={aggregated_metrics.loc['G-mean', 'macro']:.6f}, "
            f"roc_auc={aggregated_metrics.loc['ROC-AUC', 'macro']:.6f}, "
            f"f1={aggregated_metrics.loc['f1-score', 'macro']:.6f}"
        )
        print(
            "val_micro: "
            f"specificity={aggregated_metrics.loc['Specificity', 'micro']:.6f}, "
            f"sensitivity={aggregated_metrics.loc['Sensitivity', 'micro']:.6f}, "
            f"gmean={aggregated_metrics.loc['G-mean', 'micro']:.6f}, "
            f"roc_auc={aggregated_metrics.loc['ROC-AUC', 'micro']:.6f}, "
            f"f1={aggregated_metrics.loc['f1-score', 'micro']:.6f}"
        )
        print(f"mean_f1: {mean_f1:.6f}")
        print(f"lr: {current_lr:.8f}")
        print(f"best_val_loss: {best_val_loss:.6f}")
        print(f"best_epoch: {best_epoch}")
        print(f"best_checkpoint_path: {checkpoint_path}")
        print(f"improved: {improved}")
        print(f"epochs_without_improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= int(cfg["training"]["early_stopping_patience"]):
            print("Early stopping")
            break

    if best_payload is None:
        raise RuntimeError("Training finished without producing a checkpoint payload.")

    save_checkpoint(checkpoint_path, best_payload)
    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)
    training_summary_path.write_text(
        json.dumps(
            {"best_val_loss": best_val_loss, "best_epoch": best_epoch, "checkpoint_path": str(checkpoint_path)},
            indent=2,
        )
    )
    return {"checkpoint_path": str(checkpoint_path), "best_val_loss": best_val_loss, "best_epoch": best_epoch}
