import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetECG(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(UNetECG, self).__init__()

        # Encoder blocks
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder blocks
        self.dec4 = self.conv_block(256 + 256, 256)
        self.dec3 = self.conv_block(128 + 128, 128)
        self.dec2 = self.conv_block(64 + 64, 64)
        self.dec1 = self.conv_block(32 + 32, 32)

        # Deconvolutions (upsampling layers)
        self.up4 = nn.ConvTranspose1d(512, 256, kernel_size=8, stride=2, padding=3)
        self.up3 = nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=3)
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3)
        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3)

        # Final output layer
        self.final_conv = nn.Conv1d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)  # (batch, 32, l)
        enc2 = self.enc2(F.max_pool1d(enc1, 2))  # (batch, 64, l/2)
        enc3 = self.enc3(F.max_pool1d(enc2, 2))  # (batch, 128, l/4)
        enc4 = self.enc4(F.max_pool1d(enc3, 2))  # (batch, 256, l/8)

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool1d(enc4, 2))  # (batch, 512, l/16)

        # Decoder path with concatenations
        dec4 = self.up4(bottleneck)  # (batch, 256, l/8)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))  # (batch, 512, l/8)

        dec3 = self.up3(dec4)  # (batch, 128, l/4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))  # (batch, 256, l/4)

        dec2 = self.up2(dec3)  # (batch, 64, l/2)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))  # (batch, 128, l/2)

        dec1 = self.up1(dec2)  # (batch, 32, l)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))  # (batch, 64, l)

        # Final layer
        out = self.final_conv(dec1)  # (batch, 4, l)

        return out


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inp: torch.Tensor, target: torch.Tensor):
        """
        Args:
            inp: Predicted logits of shape (batch_size, 4, l)
            target: Ground-truth binary tensor of shape (batch_size, 4, l)
        """
        if inp.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: inp.shape {inp.shape} vs target.shape {target.shape}"
            )

        loss = self.loss_fn(inp, target)
        return loss
