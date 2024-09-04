from typing import List
from math import floor

import torch as th
import torch.nn as nn


class CausalConv1dLayer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) - (stride - 1)
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            stride=stride,
        )

    def forward(self, x):
        x = self.conv(x)
        trailing_pad = floor(self.padding / self.stride)
        if trailing_pad != 0: x = x[..., :-trailing_pad]
        return x
    

class CausalConv1d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_sizes: List[int],
            strides: List[int],
    ):
        super().__init__()
        self.convs = nn.Sequential()
        current_in_channels = in_channels
        for k, s in zip(kernel_sizes, strides):
            conv = CausalConv1dLayer(current_in_channels, out_channels, k, s)
            self.convs.append(conv)
            current_in_channels = out_channels

    def forward(self, x):
        return self.convs(x)
