from __future__ import annotations

from copy import deepcopy

from .ecg_cpc import ecg_cpc_classifier
from .ecgjepa import ecgjepa_classifier
from .ecgfounder import ecgfounder_net1d
from .hubert_ecg import hubert_ecg_classifier
from .merl import merl_resnet18
from .resnet1d import resnet1d18
from .s4 import s4_classifier
from .stmem import stmem_classifier


MODEL_REGISTRY = {
    "resnet1d18": resnet1d18,
    "ecg_cpc": ecg_cpc_classifier,
    "ecgjepa": ecgjepa_classifier,
    "ecgfounder": ecgfounder_net1d,
    "hubert_ecg": hubert_ecg_classifier,
    "s4": s4_classifier,
    "stmem": stmem_classifier,
    "merl": merl_resnet18,
}


def create_model(model_name: str, num_classes: int, input_channels: int, model_params: dict | None = None):
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available models: {sorted(MODEL_REGISTRY)}")
    kwargs = deepcopy(model_params or {})
    kwargs["num_classes"] = num_classes
    kwargs["input_channels"] = input_channels
    return MODEL_REGISTRY[model_name](**kwargs)
