from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """A 1D convolution layer with TensorFlow-style same padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        padding = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = padding // 2
        pad_right = padding - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    """A 1D max-pooling layer with TensorFlow-style same padding."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = max(0, self.kernel_size - 1)
        pad_left = padding // 2
        pad_right = padding - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.max_pool(x)


class Swish(nn.Module):
    """The activation used by the public ECGFounder implementation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    """Residual bottleneck block used by Net1D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: int,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: bool,
        is_first_block: bool = False,
        use_bn: bool = True,
        use_do: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.middle_channels = int(out_channels * ratio)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=self.middle_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        )

        self.bn2 = nn.BatchNorm1d(self.middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=self.middle_channels,
            out_channels=self.middle_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=groups,
        )

        self.bn3 = nn.BatchNorm1d(self.middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(
            in_channels=self.middle_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        )

        reduction = 2
        self.se_fc1 = nn.Linear(out_channels, out_channels // reduction)
        self.se_fc2 = nn.Linear(out_channels // reduction, out_channels)
        self.se_activation = Swish()

        if downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x

        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out)

        squeeze = out.mean(-1)
        squeeze = self.se_fc1(squeeze)
        squeeze = self.se_activation(squeeze)
        squeeze = self.se_fc2(squeeze)
        squeeze = torch.sigmoid(squeeze)
        out = torch.einsum("bcl,bc->bcl", out, squeeze)

        if self.downsample:
            identity = self.max_pool(identity)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            pad_left = (self.out_channels - self.in_channels) // 2
            pad_right = self.out_channels - self.in_channels - pad_left
            identity = F.pad(identity, (pad_left, pad_right), "constant", 0)
            identity = identity.transpose(-1, -2)

        return out + identity


class BasicStage(nn.Module):
    """A stage of consecutive Net1D residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: int,
        kernel_size: int,
        stride: int,
        groups: int,
        i_stage: int,
        m_blocks: int,
        use_bn: bool = True,
        use_do: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.i_stage = i_stage
        self.verbose = verbose
        self.block_list = nn.ModuleList()
        current_in_channels = in_channels
        for i_block in range(m_blocks):
            is_first_block = i_stage == 0 and i_block == 0
            downsample = i_block == 0
            block_stride = stride if downsample else 1
            block = BasicBlock(
                in_channels=current_in_channels,
                out_channels=out_channels,
                ratio=ratio,
                kernel_size=kernel_size,
                stride=block_stride,
                groups=groups,
                downsample=downsample,
                is_first_block=is_first_block,
                use_bn=use_bn,
                use_do=use_do,
            )
            self.block_list.append(block)
            current_in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.block_list:
            out = block(out)
        return out


class ECGFounderNet1D(nn.Module):
    """Public Net1D backbone used by ECGFounder."""

    def __init__(
        self,
        input_channels: int = 12,
        base_filters: int = 64,
        ratio: int = 1,
        filter_list: Iterable[int] = (64, 160, 160, 400, 400, 1024, 1024),
        m_blocks_list: Iterable[int] = (2, 2, 2, 3, 3, 4, 4),
        kernel_size: int = 16,
        stride: int = 2,
        groups_width: int = 16,
        num_classes: int = 1,
        use_bn: bool = False,
        use_do: bool = False,
        return_features: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.base_filters = base_filters
        self.ratio = ratio
        self.filter_list = list(filter_list)
        self.m_blocks_list = list(m_blocks_list)
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_width = groups_width
        self.num_classes = num_classes
        self.use_bn = use_bn
        self.use_do = use_do
        self.return_features = return_features

        self.first_conv = MyConv1dPadSame(
            in_channels=input_channels,
            out_channels=base_filters,
            kernel_size=kernel_size,
            stride=2,
        )
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        self.stage_list = nn.ModuleList()
        stage_in_channels = base_filters
        for i_stage, (out_channels, m_blocks) in enumerate(zip(self.filter_list, self.m_blocks_list)):
            stage = BasicStage(
                in_channels=stage_in_channels,
                out_channels=out_channels,
                ratio=ratio,
                kernel_size=kernel_size,
                stride=stride,
                groups=out_channels // groups_width,
                i_stage=i_stage,
                m_blocks=m_blocks,
                use_bn=use_bn,
                use_do=use_do,
            )
            self.stage_list.append(stage)
            stage_in_channels = out_channels

        self.feature_dim = stage_in_channels
        self.dense = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.first_conv(x)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)

        for stage in self.stage_list:
            out = stage(out)

        features = out.mean(-1)
        logits = self.dense(features)
        if self.return_features:
            return logits, features
        return logits


def ecgfounder_net1d(**kwargs) -> ECGFounderNet1D:
    """Create an ECGFounder-compatible Net1D model."""

    return ECGFounderNet1D(**kwargs)


def load_pretrained_state_dict(model: ECGFounderNet1D, checkpoint_path: str) -> list[str]:
    """Load a public ECGFounder checkpoint into the backbone and reset the head."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("dense.")
    }
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            "Unexpected keys when loading ECGFounder checkpoint: "
            + ", ".join(sorted(unexpected))
        )
    return list(incompatible.missing_keys)


def freeze_backbone(model: ECGFounderNet1D) -> None:
    """Freeze all Net1D parameters except the classification head."""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("dense.")
