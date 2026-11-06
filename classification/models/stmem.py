from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardFn(nn.Module):
    """Feed-forward submodule with the official ST-MEM `fn.net.*` layout."""

    def __init__(self, input_dim: int, hidden_dim: int, drop_out_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForward(nn.Module):
    """Pre-norm feed-forward block with the official ST-MEM naming layout."""

    def __init__(self, input_dim: int, hidden_dim: int, drop_out_rate: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fn = FeedForwardFn(input_dim=input_dim, hidden_dim=hidden_dim, drop_out_rate=drop_out_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class AttentionFn(nn.Module):
    """Attention submodule with the official ST-MEM `fn.to_qkv` and `fn.to_out` layout."""

    def __init__(
        self,
        input_dim: int,
        heads: int,
        dim_head: int = 64,
        qkv_bias: bool = True,
        attn_drop_out_rate: float = 0.0,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(input_dim, hidden_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_out_rate)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        query, key, value = [
            tensor.reshape(tensor.shape[0], tensor.shape[1], self.heads, -1).transpose(1, 2)
            for tensor in qkv
        ]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        output = attention @ value
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(output)


class Attention(nn.Module):
    """Pre-norm attention block with the official ST-MEM naming layout."""

    def __init__(
        self,
        input_dim: int,
        heads: int,
        dim_head: int = 64,
        qkv_bias: bool = True,
        attn_drop_out_rate: float = 0.0,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fn = AttentionFn(
            input_dim=input_dim,
            heads=heads,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            attn_drop_out_rate=attn_drop_out_rate,
            drop_out_rate=drop_out_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class TransformerBlock(nn.Module):
    """Transformer block used by the official ST-MEM encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        dim_head: int = 64,
        qkv_bias: bool = True,
        drop_out_rate: float = 0.0,
        attn_drop_out_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = Attention(
            input_dim=input_dim,
            heads=heads,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            attn_drop_out_rate=attn_drop_out_rate,
            drop_out_rate=drop_out_rate,
        )
        self.ff = FeedForward(input_dim=input_dim, hidden_dim=hidden_dim, drop_out_rate=drop_out_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class STMEMEncoder(nn.Module):
    """Encoder used by the official ST-MEM downstream transfer recipe."""

    def __init__(
        self,
        input_channels: int = 12,
        seq_len: int = 2250,
        patch_size: int = 75,
        width: int = 768,
        depth: int = 12,
        mlp_dim: int = 3072,
        heads: int = 12,
        dim_head: int = 64,
        qkv_bias: bool = True,
        drop_out_rate: float = 0.0,
        attn_drop_out_rate: float = 0.0,
    ):
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by patch_size={patch_size}.")
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.width = width
        self.depth = depth
        self.num_patches = seq_len // patch_size
        self.to_patch_embedding = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, width),
            nn.LayerNorm(width),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width)) for _ in range(input_channels))
        for index in range(depth):
            self.add_module(
                f"block{index}",
                TransformerBlock(
                    input_dim=width,
                    hidden_dim=mlp_dim,
                    heads=heads,
                    dim_head=dim_head,
                    qkv_bias=qkv_bias,
                    drop_out_rate=drop_out_rate,
                    attn_drop_out_rate=attn_drop_out_rate,
                ),
            )
        self.norm = nn.LayerNorm(width)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        if series.shape[1] != self.input_channels:
            raise ValueError(f"ST-MEM expects {self.input_channels} leads after preprocessing, got {series.shape[1]}.")
        if series.shape[2] != self.seq_len:
            raise ValueError(f"ST-MEM expects {self.seq_len} samples after preprocessing, got {series.shape[2]}.")
        batch_size, num_leads, _ = series.shape
        x = series.reshape(batch_size, num_leads, self.num_patches, self.patch_size)
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding[:, 1:self.num_patches + 1, :].unsqueeze(1)
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(batch_size, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(batch_size, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.lead_embeddings], dim=0)
        lead_embeddings = lead_embeddings.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_patches + 2, -1)
        x = x + lead_embeddings
        x = x.reshape(batch_size, num_leads * (self.num_patches + 2), self.width)
        for index in range(self.depth):
            x = getattr(self, f"block{index}")(x)
        x = x.reshape(batch_size, num_leads, self.num_patches + 2, self.width)
        x = x[:, :, 1:-1, :]
        return self.norm(torch.mean(x, dim=(1, 2)))


class STMEMClassifier(nn.Module):
    """Frozen-encoder classifier wrapper used for ST-MEM linear probing."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 1,
        seq_len: int = 2250,
        patch_size: int = 75,
        width: int = 768,
        depth: int = 12,
        mlp_dim: int = 3072,
        heads: int = 12,
        dim_head: int = 64,
        qkv_bias: bool = True,
        drop_out_rate: float = 0.0,
        attn_drop_out_rate: float = 0.0,
    ):
        super().__init__()
        self.encoder = STMEMEncoder(
            input_channels=input_channels,
            seq_len=seq_len,
            patch_size=patch_size,
            width=width,
            depth=depth,
            mlp_dim=mlp_dim,
            heads=heads,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            drop_out_rate=drop_out_rate,
            attn_drop_out_rate=attn_drop_out_rate,
        )
        self.head = nn.Linear(width, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder-level features for a batch of ECG signals."""

        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def stmem_classifier(**kwargs) -> STMEMClassifier:
    """Create an ST-MEM classifier compatible with the official encoder checkpoint."""

    return STMEMClassifier(**kwargs)


def load_pretrained_state_dict(model: STMEMClassifier, checkpoint_path: str) -> list[str]:
    """Load an official ST-MEM encoder checkpoint and reset the classification head."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unsupported ST-MEM checkpoint format. Expected a state dict under `model` or `state_dict`.")
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("head.")
    }
    incompatible = model.encoder.load_state_dict(filtered_state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys when loading ST-MEM checkpoint: "
            + ", ".join(sorted(incompatible.unexpected_keys))
        )
    unexpected_missing = [key for key in incompatible.missing_keys if not key.startswith("head.")]
    if unexpected_missing:
        raise RuntimeError(
            "Missing encoder keys when loading ST-MEM checkpoint: "
            + ", ".join(sorted(unexpected_missing))
        )
    return list(incompatible.missing_keys)


def freeze_backbone(model: STMEMClassifier) -> None:
    """Freeze the ST-MEM encoder and keep only the classification head trainable."""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("head.")
