from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from .vendor_s4 import S4Model


class _LambdaLayer(nn.Module):
    """Apply a lightweight tensor transform inside an nn.Sequential block."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _conv1d(
    in_planes: int,
    out_planes: int,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
    bn: bool = True,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build the benchmark-compatible Conv1d -> Norm -> ReLU block."""

    layers: list[nn.Module] = [
        nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            dilation=dilation,
            bias=not bn,
        )
    ]
    if bn:
        if layer_norm:
            layers.append(_LambdaLayer(lambda x: x.transpose(1, 2)))
            layers.append(nn.LayerNorm(out_planes))
            layers.append(_LambdaLayer(lambda x: x.transpose(1, 2)))
        else:
            layers.append(nn.BatchNorm1d(out_planes))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class _TransposeToSequence(nn.Module):
    """Convert `(batch, channels, samples)` features to `(batch, steps, channels)`."""

    def forward(self, x):
        return x.transpose(1, 2)


class ConvSequenceEncoder(nn.Module):
    """Benchmark-compatible convolutional encoder used by the ECG-CPC backbone."""

    def __init__(
        self,
        input_channels: int,
        features: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        dilations: list[int] | None = None,
        normalization: bool = True,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        dilations = dilations or [1] * len(features)
        if not (len(features) == len(kernel_sizes) == len(strides) == len(dilations)):
            raise ValueError("features, kernel_sizes, strides, and dilations must have the same length.")

        blocks: list[nn.Module] = []
        in_channels = input_channels
        for out_channels, kernel_size, stride, dilation in zip(features, kernel_sizes, strides, dilations):
            blocks.append(
                _conv1d(
                    in_planes=in_channels,
                    out_planes=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bn=normalization,
                    layer_norm=layer_norm,
                )
            )
            in_channels = out_channels
        blocks.append(_TransposeToSequence())
        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


class ECGCPCPredictor(nn.Module):
    """S4 predictor block inside the ECG-CPC backbone."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        state_dim: int = 8,
        model_dim: int = 512,
        layers: int = 4,
        dropout: float = 0.2,
        tie_dropout: bool = False,
        prenorm: bool = False,
        batchnorm: bool = False,
        causal: bool = True,
        backbone: str = "s42",
    ) -> None:
        super().__init__()
        self.predictor = S4Model(
            d_input=input_dim if input_dim != model_dim else None,
            d_output=None,
            d_state=state_dim,
            d_model=model_dim,
            n_layers=layers,
            dropout=dropout,
            tie_dropout=tie_dropout,
            prenorm=prenorm,
            l_max=sequence_length,
            transposed_input=False,
            bidirectional=not causal,
            layer_norm=not batchnorm,
            pooling=False,
            backbone=backbone,
        )

    def forward(self, x):
        return self.predictor(x)


class ECGCPCTimeSeriesEncoder(nn.Module):
    """Minimal benchmark-compatible `ts_encoder` container for ECG-CPC."""

    def __init__(
        self,
        input_channels: int,
        signal_length: int,
        encoder_features: list[int],
        encoder_kernel_sizes: list[int],
        encoder_strides: list[int],
        encoder_dilations: list[int] | None = None,
        predictor_state_dim: int = 8,
        predictor_model_dim: int = 512,
        predictor_layers: int = 4,
        predictor_dropout: float = 0.2,
        predictor_tie_dropout: bool = False,
        predictor_prenorm: bool = False,
        predictor_batchnorm: bool = False,
        predictor_causal: bool = True,
        predictor_backbone: str = "s42",
    ) -> None:
        super().__init__()
        self.encoder = ConvSequenceEncoder(
            input_channels=input_channels,
            features=encoder_features,
            kernel_sizes=encoder_kernel_sizes,
            strides=encoder_strides,
            dilations=encoder_dilations,
            normalization=True,
            layer_norm=False,
        )
        self.predictor = ECGCPCPredictor(
            input_dim=encoder_features[-1],
            sequence_length=signal_length,
            state_dim=predictor_state_dim,
            model_dim=predictor_model_dim,
            layers=predictor_layers,
            dropout=predictor_dropout,
            tie_dropout=predictor_tie_dropout,
            prenorm=predictor_prenorm,
            batchnorm=predictor_batchnorm,
            causal=predictor_causal,
            backbone=predictor_backbone,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.predictor(encoded)


class ECGCPCClassifier(nn.Module):
    """ECG-CPC linear-probe classifier used in the PTB-XL -> ISP pipeline."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 12,
        signal_length: int = 600,
        encoder_features: list[int] | None = None,
        encoder_kernel_sizes: list[int] | None = None,
        encoder_strides: list[int] | None = None,
        encoder_dilations: list[int] | None = None,
        predictor_state_dim: int = 8,
        predictor_model_dim: int = 512,
        predictor_layers: int = 4,
        predictor_dropout: float = 0.2,
        predictor_tie_dropout: bool = False,
        predictor_prenorm: bool = False,
        predictor_batchnorm: bool = False,
        predictor_causal: bool = True,
        predictor_backbone: str = "s42",
    ) -> None:
        super().__init__()
        encoder_features = encoder_features or [512, 512, 512, 512]
        encoder_kernel_sizes = encoder_kernel_sizes or [3, 1, 1, 1]
        encoder_strides = encoder_strides or [2, 1, 1, 1]
        self.ts_encoder = ECGCPCTimeSeriesEncoder(
            input_channels=input_channels,
            signal_length=signal_length,
            encoder_features=encoder_features,
            encoder_kernel_sizes=encoder_kernel_sizes,
            encoder_strides=encoder_strides,
            encoder_dilations=encoder_dilations,
            predictor_state_dim=predictor_state_dim,
            predictor_model_dim=predictor_model_dim,
            predictor_layers=predictor_layers,
            predictor_dropout=predictor_dropout,
            predictor_tie_dropout=predictor_tie_dropout,
            predictor_prenorm=predictor_prenorm,
            predictor_batchnorm=predictor_batchnorm,
            predictor_causal=predictor_causal,
            predictor_backbone=predictor_backbone,
        )
        self.head = nn.Linear(predictor_model_dim, num_classes)

    def forward(self, x):
        """Return multilabel logits for an ECG batch."""

        x = torch.nan_to_num(x)
        sequence_features = self.ts_encoder(x)
        pooled_features = sequence_features.mean(dim=1)
        return torch.nan_to_num(self.head(pooled_features))


def ecg_cpc_classifier(**kwargs) -> ECGCPCClassifier:
    """Factory compatible with the model registry."""

    return ECGCPCClassifier(**kwargs)


def freeze_backbone(model: ECGCPCClassifier) -> None:
    """Freeze the pretrained ECG-CPC backbone and keep only the head trainable."""

    for parameter in model.ts_encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True


class _DynamicStubModule(types.ModuleType):
    """Module stub that materializes placeholder classes on demand during unpickling."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (), {})
        setattr(self, name, value)
        return value


@contextmanager
def _checkpoint_stub_modules():
    """Provide lightweight placeholder modules required to unpickle benchmark checkpoints."""

    module_names = [
        "clinical_ts",
        "clinical_ts.template_modules",
        "clinical_ts.task",
        "clinical_ts.task.ecg",
        "clinical_ts.metric",
        "clinical_ts.metric.base",
        "clinical_ts.loss",
        "clinical_ts.loss.supervised",
        "clinical_ts.loss.selfsupervised",
        "clinical_ts.ts",
        "clinical_ts.ts.encoder",
        "clinical_ts.ts.head",
        "clinical_ts.ts.s4",
        "clinical_ts.config",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    created_modules: dict[str, types.ModuleType] = {}
    try:
        for name in module_names:
            module = _DynamicStubModule(name)
            module.__file__ = "<checkpoint_stub>"
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = module
            created_modules[name] = module

        created_modules["clinical_ts"].template_modules = created_modules["clinical_ts.template_modules"]
        created_modules["clinical_ts"].task = created_modules["clinical_ts.task"]
        created_modules["clinical_ts"].metric = created_modules["clinical_ts.metric"]
        created_modules["clinical_ts"].loss = created_modules["clinical_ts.loss"]
        created_modules["clinical_ts"].ts = created_modules["clinical_ts.ts"]
        created_modules["clinical_ts"].config = created_modules["clinical_ts.config"]
        created_modules["clinical_ts.task"].ecg = created_modules["clinical_ts.task.ecg"]
        created_modules["clinical_ts.metric"].base = created_modules["clinical_ts.metric.base"]
        created_modules["clinical_ts.loss"].supervised = created_modules["clinical_ts.loss.supervised"]
        created_modules["clinical_ts.loss"].selfsupervised = created_modules["clinical_ts.loss.selfsupervised"]
        created_modules["clinical_ts.ts"].encoder = created_modules["clinical_ts.ts.encoder"]
        created_modules["clinical_ts.ts"].head = created_modules["clinical_ts.ts.head"]
        created_modules["clinical_ts.ts"].s4 = created_modules["clinical_ts.ts.s4"]
        yield
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load the ECG-CPC checkpoint state dict while stubbing non-runtime benchmark modules."""

    with _checkpoint_stub_modules():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise RuntimeError("ECG-CPC checkpoint does not contain a state_dict entry.")
    return checkpoint["state_dict"]


def load_pretrained_state_dict(model: ECGCPCClassifier, checkpoint_path: str) -> list[str]:
    """Load benchmark ECG-CPC encoder/predictor weights into the downstream classifier."""

    checkpoint_state_dict = _load_checkpoint_state_dict(checkpoint_path)
    model_state = model.state_dict()
    pretrained_state = {
        key: value.clone()
        for key, value in checkpoint_state_dict.items()
        if key.startswith("ts_encoder.encoder.") or key.startswith("ts_encoder.predictor.")
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    if not pretrained_state:
        raise RuntimeError(f"Unexpected ECG-CPC checkpoint layout: {checkpoint_path}")
    incompatible = model.load_state_dict(pretrained_state, strict=False, assign=True)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys when loading ECG-CPC checkpoint: " + ", ".join(sorted(incompatible.unexpected_keys))
        )
    ignored_suffixes = (".omega", ".z")
    return [key for key in incompatible.missing_keys if not key.endswith(ignored_suffixes)]
