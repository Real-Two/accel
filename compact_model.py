"""A small, CPU-oriented segmentation model for the acceleration challenge.

It keeps the input/output contract of the original U-Net (14x128x128 ->
1x128x128) while making the training workload realistic for CPU deployment.
"""

import torch.nn as nn


class CompactSeg(nn.Module):
    def __init__(self, in_channels=14, out_channels=1, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)
