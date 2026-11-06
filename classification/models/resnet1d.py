from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self, output_size: int = 1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        activation: str | None = None,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "elu":
            layers.append(nn.ELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, bias: bool):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = torch.div(in_dim + self.stride - 1, self.stride, rounding_mode="floor")
        padding = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = torch.div(padding, 2, rounding_mode="floor")
        pad_right = padding - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = torch.div(in_dim + self.stride - 1, self.stride, rounding_mode="floor")
        padding = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = torch.div(padding, 2, rounding_mode="floor")
        pad_right = padding - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.max_pool(x)


class BasicBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        drop_prob: float = 0.0,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.downsample = downsample
        self.conv1 = MyConv1dPadSame(in_channels, out_channels, kernel_size, stride, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=drop_prob)
        self.conv2 = MyConv1dPadSame(out_channels, out_channels, kernel_size, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.do1(self.relu1(self.bn1(self.conv1(x))))
        out = self.do2(self.relu2(self.bn2(self.conv2(out))))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return out + identity


class BottleneckBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        drop_prob: float = 0.0,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.downsample = downsample
        self.conv1 = MyConv1dPadSame(in_channels, out_channels, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=drop_prob)
        self.conv2 = MyConv1dPadSame(out_channels, out_channels, kernel_size, stride, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=drop_prob)
        self.conv3 = MyConv1dPadSame(out_channels, out_channels * 4, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * 4)
        self.relu3 = nn.ReLU()
        self.do3 = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.do1(self.relu1(self.bn1(self.conv1(x))))
        out = self.do2(self.relu2(self.bn2(self.conv2(out))))
        out = self.do3(self.relu3(self.bn3(self.conv3(out))))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return out + identity


class ResNet1d(nn.Module):
    def __init__(
        self,
        block_type: type[nn.Module],
        layers: list[int],
        input_channels: int = 12,
        base_filters: int = 64,
        kernel_size: int | list[int] = 3,
        stride: int = 2,
        num_classes: int = 1,
        dropout_prob: float = 0.0,
        fix_feature_dim: bool = True,
        kernel_size_stem: int | None = None,
        stride_stem: int = 2,
        pooling_stem: bool = True,
        concat_pooling: bool = False,
        hidden_layers_head: list[int] | None = None,
        dropout_prob_head: float = 0.5,
        act_head: str = "relu",
        bn_head: bool = True,
        bn_final_head: bool = False,
    ):
        super().__init__()
        self.base_filters = base_filters
        self.in_channels = input_channels
        if block_type is BasicBlock1d:
            self.block_name = "BasicBlock1d"
            self.block_expansion = 1
        elif block_type is BottleneckBlock1d:
            self.block_name = "BottleneckBlock1d"
            self.block_expansion = 4
        else:
            raise ValueError("Unsupported block type for ResNet1d")

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(layers)
        kernel_size_stem = kernel_size[0] if kernel_size_stem is None else kernel_size_stem

        stem_layers: list[nn.Module] = [
            MyConv1dPadSame(input_channels, self.base_filters, kernel_size_stem, stride_stem, bias=False),
            nn.BatchNorm1d(self.base_filters),
            nn.ReLU(),
        ]
        if pooling_stem:
            stem_layers.append(MyMaxPool1dPadSame(kernel_size=3, stride=2))
        self.stem = nn.Sequential(*stem_layers)

        self.backbone = nn.Sequential()
        for index, num_blocks in enumerate(layers):
            if index == 0:
                self.in_channels = self.base_filters
            else:
                self.in_channels = self.base_filters * self.block_expansion
                self.base_filters = self.base_filters if fix_feature_dim else 2 * self.base_filters
            block = self._create_block(
                block_type=block_type,
                num_blocks=num_blocks,
                stride=stride,
                kernel_size=kernel_size[index],
                dropout_prob=dropout_prob,
            )
            self.backbone.add_module(f"{self.block_name}_block{index}", block)

        self.head = nn.Sequential()
        pooling: nn.Module
        if concat_pooling:
            pooling = AdaptiveConcatPool1d(1)
            input_shape = self.in_channels * 2
        else:
            pooling = nn.AdaptiveMaxPool1d(1)
            input_shape = self.in_channels
        self.head.add_module("pool", pooling)
        self.head.add_module("flatten", nn.Flatten())

        hidden_layers_head = hidden_layers_head or []
        for index, hidden_size in enumerate(hidden_layers_head):
            self.head.add_module(
                f"fc_block_{index}",
                LinearBlock(
                    in_features=input_shape,
                    out_features=hidden_size,
                    dropout=dropout_prob_head,
                    activation=act_head,
                    use_batch_norm=bn_head,
                ),
            )
            input_shape = hidden_size
        self.head.add_module(
            "fc_out",
            LinearBlock(
                in_features=input_shape,
                out_features=num_classes,
                dropout=dropout_prob_head,
                activation=None,
                use_batch_norm=bn_final_head,
            ),
        )

    def _create_block(
        self,
        block_type: type[nn.Module],
        num_blocks: int,
        stride: int,
        kernel_size: int,
        dropout_prob: float,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                MyConv1dPadSame(self.in_channels, self.base_filters * self.block_expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.base_filters * self.block_expansion),
            )
        layers = nn.Sequential()
        layers.add_module(
            f"{self.block_name}_layer0",
            block_type(self.in_channels, self.base_filters, kernel_size, stride, dropout_prob, downsample),
        )
        self.in_channels = self.base_filters * self.block_expansion
        for index in range(1, num_blocks):
            layers.add_module(
                f"{self.block_name}_layer{index}",
                block_type(self.in_channels, self.base_filters, kernel_size=3, stride=1, drop_prob=dropout_prob),
            )
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        return self.head(x)


def resnet1d18(**kwargs) -> ResNet1d:
    return ResNet1d(block_type=BasicBlock1d, layers=[2, 2, 2, 2], **kwargs)


def resnet1d34(**kwargs) -> ResNet1d:
    return ResNet1d(block_type=BasicBlock1d, layers=[3, 4, 6, 3], **kwargs)


def resnet1d50(**kwargs) -> ResNet1d:
    return ResNet1d(block_type=BottleneckBlock1d, layers=[3, 4, 6, 3], **kwargs)


def resnet1d101(**kwargs) -> ResNet1d:
    return ResNet1d(block_type=BottleneckBlock1d, layers=[3, 4, 23, 3], **kwargs)


def resnet1d152(**kwargs) -> ResNet1d:
    return ResNet1d(block_type=BottleneckBlock1d, layers=[3, 8, 36, 3], **kwargs)
