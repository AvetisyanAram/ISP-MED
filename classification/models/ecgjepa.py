from __future__ import annotations

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Multi-head self-attention block compatible with public ECG-JEPA checkpoints."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, sequence_length, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, dim)
        return self.proj(output)


class MLP(nn.Module):
    """Transformer feed-forward block compatible with public ECG-JEPA checkpoints."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Transformer block compatible with the `encoder_blocks.blocks.*` checkpoint layout."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EncoderBlocks(nn.Module):
    """Container module that preserves the public `encoder_blocks.blocks.*` state dict names."""

    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ECGJEPAEncoder(nn.Module):
    """Encoder used by the public `sehunfromdaegu/ECG_JEPA` multiblock checkpoint."""

    def __init__(
        self,
        input_channels: int = 8,
        signal_length: int = 2500,
        patch_size: int = 50,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if signal_length % patch_size != 0:
            raise ValueError(f"signal_length={signal_length} must be divisible by patch_size={patch_size}.")
        self.input_channels = input_channels
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_patches_per_lead = signal_length // patch_size
        self.num_tokens = input_channels * self.num_patches_per_lead

        self.W_P = nn.Linear(patch_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(self.num_tokens, dim))
        self.encoder_blocks = EncoderBlocks(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"ECG-JEPA expects {self.input_channels} leads after preprocessing, got {x.shape[1]}."
            )
        if x.shape[2] != self.signal_length:
            raise ValueError(
                f"ECG-JEPA expects {self.signal_length} samples after preprocessing, got {x.shape[2]}."
            )
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.input_channels, self.num_patches_per_lead, self.patch_size)
        x = x.reshape(batch_size, self.num_tokens, self.patch_size)
        x = self.W_P(x)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.encoder_blocks(x)
        return self.norm(x)


class ECGJEPAClassifier(nn.Module):
    """Frozen-encoder classifier wrapper for public ECG-JEPA linear evaluation."""

    def __init__(
        self,
        input_channels: int = 8,
        num_classes: int = 1,
        signal_length: int = 2500,
        patch_size: int = 50,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling
        self.encoder = ECGJEPAEncoder(
            input_channels=input_channels,
            signal_length=signal_length,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )
        self.classifier = nn.Linear(dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return token-level ECG-JEPA encoder features."""

        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        if self.pooling == "mean":
            pooled = features.mean(dim=1)
        elif self.pooling == "max":
            pooled = features.max(dim=1).values
        else:
            raise ValueError(f"Unsupported ECG-JEPA pooling '{self.pooling}'.")
        return self.classifier(pooled)


def ecgjepa_classifier(**kwargs) -> ECGJEPAClassifier:
    """Create an ECG-JEPA classifier compatible with public multiblock checkpoints."""

    return ECGJEPAClassifier(**kwargs)


def load_pretrained_state_dict(model: ECGJEPAClassifier, checkpoint_path: str) -> list[str]:
    """Load a public `sehunfromdaegu/ECG_JEPA` encoder checkpoint and reset the classifier head."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint:
        raise RuntimeError(
            "Unsupported ECG-JEPA checkpoint format. Expected a dictionary with an `encoder` state dict."
        )
    encoder_state_dict = checkpoint["encoder"]
    if not isinstance(encoder_state_dict, dict):
        raise RuntimeError("ECG-JEPA checkpoint field `encoder` must be a state dict.")
    incompatible = model.encoder.load_state_dict(encoder_state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys when loading ECG-JEPA checkpoint: "
            + ", ".join(sorted(incompatible.unexpected_keys))
        )
    unexpected_missing = [key for key in incompatible.missing_keys if not key.startswith("classifier.")]
    if unexpected_missing:
        raise RuntimeError(
            "Missing encoder keys when loading ECG-JEPA checkpoint: "
            + ", ".join(sorted(unexpected_missing))
        )
    return list(incompatible.missing_keys)


def freeze_backbone(model: ECGJEPAClassifier) -> None:
    """Freeze the ECG-JEPA encoder and keep only the linear classifier trainable."""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier.")
