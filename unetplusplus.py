import torch
import torch.nn as nn
from unet_parts import DoubleConv, DownSample, NestedUpSample

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        channels = (64, 128, 256, 512, 1024)
        self.down1 = DownSample(in_channels, channels[0])
        self.down2 = DownSample(channels[0], channels[1])
        self.down3 = DownSample(channels[1], channels[2])
        self.down4 = DownSample(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[4])
        self.nodes = nn.ModuleDict()
        for j in range(1, 5):
            for i in range(0, 5 - j):
                skip_channels = j * channels[i]
                self.nodes[f"{i}_{j}"] = NestedUpSample(
                    channels[i + 1], skip_channels, channels[i]
                    )
        self.out = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        x00, p1 = self.down1(x)
        x10, p2 = self.down2(p1)
        x20, p3 = self.down3(p2)
        x30, p4 = self.down4(p3)
        x40 = self.bottleneck(p4)
        features = {
            (0, 0): x00,
            (1, 0): x10,
            (2, 0): x20,
            (3, 0): x30,
            (4, 0): x40
            }
        for j in range(1, 5):
            for i in range(0, 5 - j):
                same_scale = [features[(i, k)] for k in range(j)]
                features[(i, j)] = self.nodes[f"{i}_{j}"](
                    features[(i + 1, j - 1)], *same_scale
                    )
        return self.out(features[(0, 4)])