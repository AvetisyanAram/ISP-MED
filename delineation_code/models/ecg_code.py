import torch
import torchlibrosa as tl
from torch import nn

from models.backport import ConvNormActivation


def make_mobilenet_block(in_filters, out_filters, shape, stride):
    return (
        ConvNormActivation(in_filters, in_filters, shape, stride, groups=in_filters),
        ConvNormActivation(in_filters, out_filters, 1),
    )


class ECGCode(nn.Module):
    def __init__(
        self,
        n_classes,
        in_channels=1,
        head_size=256,
        sr=50,
        n_fft=150,
        hop_length=1,
        n_mels=64,
    ):
        super(ECGCode, self).__init__()

        self.signal_to_mel = nn.Sequential(
            tl.Spectrogram(n_fft=n_fft, hop_length=hop_length),
            tl.LogmelFilterBank(sr=sr, n_fft=n_fft, n_mels=n_mels),
        )

        self.mobile_net = nn.Sequential(
            ConvNormActivation(in_channels, 32, 3, 2),
            *make_mobilenet_block(in_filters=32, out_filters=64, shape=3, stride=1),
            *make_mobilenet_block(in_filters=64, out_filters=128, shape=3, stride=2),
            *make_mobilenet_block(in_filters=128, out_filters=256, shape=3, stride=2),
            *make_mobilenet_block(in_filters=256, out_filters=512, shape=3, stride=2),
            *make_mobilenet_block(in_filters=512, out_filters=1024, shape=3, stride=2),
            *make_mobilenet_block(in_filters=1024, out_filters=1024, shape=3, stride=1),
        )

        self.yoho = nn.Sequential(
            *make_mobilenet_block(in_filters=1024, out_filters=512, shape=3, stride=1),
            *make_mobilenet_block(in_filters=512, out_filters=256, shape=3, stride=1),
            *make_mobilenet_block(in_filters=256, out_filters=128, shape=3, stride=1),
        )

        self.head = nn.Sequential(nn.Conv1d(head_size, 3 * n_classes, 1), nn.Sigmoid())

    def forward(self, x):
        x_mels = []
        for i in range(x.shape[1]):
            x_mels.append(self.signal_to_mel(x[:, i]))
        x = torch.cat(x_mels, 1)

        x = self.mobile_net(x)
        x = self.yoho(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x


class ECGCodeLoss(nn.Module):
    def __init__(self, class_coeff=1, reg_coeff=1, pos_weight=1):
        super(ECGCodeLoss, self).__init__()
        self.class_coeff = class_coeff
        self.reg_coeff = reg_coeff
        self.pos_weight = pos_weight

    def count_class_loss(self, inp, target, index, class_pos_weight):
        return torch.where(
            abs(inp[index::9] - target[index::9]) < 0,
            (inp[index::9] - target[index::9]) ** 2 * 0,
            (inp[index::9] - target[index::9]) ** 2 * class_pos_weight,
        )

    def count_reg_loss(self, inp, target, index, reg_pos_weight):
        return torch.where(
            (inp[index::9] - target[index::9]) ** 2
            + (inp[index + 1 :: 9] - target[index + 1 :: 9]) ** 2
            < 0.15,
            (inp[index::9] - target[index::9]) ** 2 * 0,
            (
                (inp[index::9] - target[index::9]) ** 2
                + (inp[index + 1 :: 9] - target[index + 1 :: 9]) ** 2
            )
            * target[index - 1 :: 9]
            * reg_pos_weight,
        )

    def forward(self, inp: torch.Tensor, target: torch.Tensor):
        inp = torch.transpose(inp, 1, 2)
        target = torch.transpose(target, 1, 2)
        inp = inp.reshape(-1)
        target = target.reshape(-1)

        p_class_loss = self.count_class_loss(inp, target, 0, 1)
        q_class_loss = self.count_class_loss(inp, target, 3, 1)
        t_class_loss = self.count_class_loss(inp, target, 6, 1)
        p_reg_loss = self.count_reg_loss(inp, target, 1, 1)
        q_reg_loss = self.count_reg_loss(inp, target, 4, 1)
        t_reg_loss = self.count_reg_loss(inp, target, 7, 1)
        class_loss = torch.cat((p_class_loss, q_class_loss, t_class_loss))
        reg_loss = torch.cat((p_reg_loss, q_reg_loss, t_reg_loss))
        loss = class_loss * self.class_coeff + reg_loss * self.reg_coeff
        loss = loss.mean()
        return loss
