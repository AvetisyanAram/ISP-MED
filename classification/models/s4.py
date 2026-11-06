from __future__ import annotations

import torch.nn as nn

from .vendor_s4 import S4Model


class S4Classifier(nn.Module):
    """Supervised S4 classifier used for train-from-scratch downstream baselines."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 12,
        signal_length: int = 250,
        d_state: int = 8,
        d_model: int = 512,
        n_layers: int = 4,
        dropout: float = 0.2,
        tie_dropout: bool = False,
        prenorm: bool = False,
        layer_norm: bool = True,
        bidirectional: bool = True,
        backbone: str = "s42",
    ) -> None:
        super().__init__()
        self.encoder = S4Model(
            d_input=input_channels,
            d_output=num_classes,
            d_state=d_state,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            tie_dropout=tie_dropout,
            prenorm=prenorm,
            l_max=signal_length,
            transposed_input=True,
            bidirectional=bidirectional,
            layer_norm=layer_norm,
            pooling=True,
            backbone=backbone,
        )

    def forward(self, x):
        """Return multilabel logits for a `(batch, channels, samples)` ECG tensor."""

        return self.encoder(x)


def s4_classifier(**kwargs) -> S4Classifier:
    """Factory compatible with the model registry."""

    return S4Classifier(**kwargs)
