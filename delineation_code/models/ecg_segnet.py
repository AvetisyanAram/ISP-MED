import torch
import torch.nn as nn


class SegNetECG(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super(SegNetECG, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            ),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, 2 * hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(2 * hidden_size, 2 * hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
        )
        self.output = torch.nn.Linear(2 * hidden_size, out_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features, _ = self.features(x)
        output = self.classifier(features)
        output = self.output(output)
        output = output.permute(0, 2, 1)

        return output


class SegnetLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(SegnetLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inp: torch.Tensor, target: torch.Tensor):
        """
        Args:
            inp: Predicted logits of shape (batch_size, 4, l)
            target: Ground-truth binary tensor of shape (batch_size, 4, l)
        """
        batch_size, _, seq_len = target.shape
        inp = inp.view(batch_size, 4, seq_len)

        if inp.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: inp.shape {inp.shape} vs target.shape {target.shape}"
            )

        loss = self.loss_fn(inp, target)
        return loss
