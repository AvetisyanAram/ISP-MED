from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ActivationFunction(nn.Module):
    """Small activation wrapper used by HuBERT-ECG classifiers."""

    def __init__(self, activation: str):
        super().__init__()
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class HuBERTECGClassifier(nn.Module):
    """HuBERT-ECG backbone with a task-specific classification head."""

    def __init__(
        self,
        hubert_model: nn.Module,
        num_classes: int,
        classifier_hidden_size: int | None = None,
        activation: str = "tanh",
        use_label_embedding: bool = False,
        classifier_dropout_prob: float = 0.1,
        input_channels: int = 12,
    ):
        super().__init__()
        self.hubert = hubert_model
        if hasattr(self.hubert, "config"):
            if hasattr(self.hubert.config, "mask_time_prob"):
                self.hubert.config.mask_time_prob = 0.0
            if hasattr(self.hubert.config, "mask_feature_prob"):
                self.hubert.config.mask_feature_prob = 0.0
        self.input_channels = input_channels
        self.use_label_embedding = use_label_embedding
        self.num_classes = num_classes
        self.classifier_hidden_size = classifier_hidden_size
        self.activation = ActivationFunction(activation)
        self.classifier_dropout = nn.Dropout(classifier_dropout_prob)
        self._build_head()

    def _build_head(self) -> None:
        hidden_size = int(self.hubert.config.hidden_size)
        if self.use_label_embedding:
            self.label_embedding = nn.Embedding(self.num_classes, hidden_size)
            self.classifier = None
            return
        if self.classifier_hidden_size is None:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, self.classifier_hidden_size),
                self.activation,
                nn.Linear(self.classifier_hidden_size, self.num_classes),
            )

    def replace_hubert(self, hubert_model: nn.Module) -> None:
        """Swap the HuBERT backbone and rebuild the head if needed."""

        self.hubert = hubert_model
        self._build_head()

    def _pool_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected HuBERT hidden states with shape (B, T, C), got {hidden_states.shape}")
        return hidden_states.mean(dim=1)

    def _compute_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = self.classifier_dropout(pooled)
        if self.use_label_embedding:
            return torch.cosine_similarity(pooled.unsqueeze(1), self.label_embedding.weight.unsqueeze(0), dim=-1)
        if self.classifier is None:
            raise RuntimeError("Classifier head is not initialized.")
        return self.classifier(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            batch_size, channels, samples = x.shape
            if channels != self.input_channels:
                raise ValueError(f"Expected {self.input_channels} leads, got {channels}")
            x = x.reshape(batch_size, channels * samples)
        if x.ndim != 2:
            raise ValueError(f"Expected flattened input of shape (B, T), got {x.shape}")
        outputs = self.hubert(x, return_dict=True)
        pooled = self._pool_hidden(outputs.last_hidden_state)
        return self._compute_logits(pooled)


def _require_transformers() -> tuple[object, object]:
    try:
        from transformers import AutoConfig, HubertConfig, HubertModel
    except ImportError as exc:  # pragma: no cover - tested via import checks
        raise ImportError(
            "HuBERT-ECG requires the transformers package. Install it to use model=hubert_ecg."
        ) from exc
    return AutoConfig, HubertConfig, HubertModel


def hubert_ecg_classifier(
    num_classes: int,
    input_channels: int = 12,
    **kwargs,
) -> HuBERTECGClassifier:
    """Create a HuBERT-ECG classifier wrapper with a randomly initialized backbone."""

    AutoConfig, HubertConfig, HubertModel = _require_transformers()
    if input_channels != 12:
        raise ValueError("HuBERT-ECG expects 12 input leads.")
    hubert_config = kwargs.get("hubert_config")
    if isinstance(hubert_config, dict):
        config = HubertConfig(**hubert_config)
    else:
        config = AutoConfig.for_model("hubert")
    hubert = HubertModel(config)
    return HuBERTECGClassifier(
        hubert_model=hubert,
        num_classes=num_classes,
        input_channels=input_channels,
        classifier_hidden_size=kwargs.get("classifier_hidden_size"),
        activation=kwargs.get("activation", "tanh"),
        use_label_embedding=kwargs.get("use_label_embedding", False),
        classifier_dropout_prob=kwargs.get("classifier_dropout_prob", 0.1),
    )


def load_pretrained_state_dict(model: HuBERTECGClassifier, checkpoint_path: str) -> list[str]:
    """Load HuBERT-ECG pretrained weights from a local checkpoint directory or file."""

    AutoConfig, _, HubertModel = _require_transformers()
    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_dir():
        hubert = HubertModel.from_pretrained(str(path), local_files_only=True)
        model.replace_hubert(hubert)
        return []
    config_path = path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"HuBERT-ECG config.json not found next to checkpoint: {config_path}")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - tested via import checks
        raise ImportError(
            "HuBERT-ECG requires safetensors to load local checkpoints. Install safetensors."
        ) from exc
    config = AutoConfig.from_pretrained(str(config_path), local_files_only=True)
    hubert = HubertModel(config)
    state_dict = load_file(str(path))
    missing_keys, unexpected_keys = hubert.load_state_dict(state_dict, strict=False)
    model.replace_hubert(hubert)
    return list(missing_keys) + list(unexpected_keys)


def freeze_backbone(model: HuBERTECGClassifier) -> None:
    """Freeze the HuBERT-ECG backbone parameters."""

    model.hubert.requires_grad_(False)
