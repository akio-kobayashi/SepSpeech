import pytorch_lightning as pt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import math
from einops import rearrange

class QntSpeakerNetwork(nn.Module):
    def __init__(self, width:int, in_channels:int, out_channels:int, kernel_size:int, stride:int, num_speakers:int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(width, in_channels, out_channels, kernel_size, stride),
            LayerNorm2d(out_channels, width),
        )
        self.linear=nn.Sequential(
            nn.ReLU(),
            nn.Linear(width, num_speakers),
        )

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        y = self.conv(x) # (B C T H)
        y = torch.mean(torch.mean(y, dim=-2), dim=-2)  # (B, H)
        z = self.linear(y)        # (B, S)
        return y, z

class QntSpeakerAdaptationLayer(nn.Module):
    def __init__(self, adpt_type, widths=None) -> None:
        super().__init__()
        self.linear=None
        if adpt_type == 'residual':
            assert widths is not None
            self.linear = nn.Linear(widths[0]+widths[1], widths[0])

    def forward(self, x:torch.Tensor, s:torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            # (B, C, T) x (B, C)
            y = x * s.unsqueeze(-1)
            return y
        else:
            B, C, T, H = x.shape
            s = s.unsqueeze(-1).repeat((1, 1, T, 1))
            y = self.linear(torch.cat((x, s), dim=-2)) # B C T H_1+H_2 -> B C T H_1
            return x + y

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, channels, width):
        super.__init__([channels,width])

    def forward(self, x):
        _, _, C, W = x.shape
        x = rearrange(x, 'b c t h -> b t c h')
        super().forward(x)
        x = rearrange(x, 'b t c h -> b c t h')
        return x
    
class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.layer = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        def forward(self, x):
            return self.layer(x)

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        # out = int((in + 2*padding - (kernel_size - 1) -1)/stride + 1)
        #     = int(in + 2*padding - kernel_size - 1 -1 + 1)
        #     = int(in + 2*padding - kernel_size - 1)
        #     = in + 2*padding - kernel_size - 1
        assert kernel_size % 2 == 1
        self.layer = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride=(stride, 1),
                               padding=(0, kernel_size//2)
                               groups=in_channels)
        
        def forward(self, x):
            return self.layer(x)

class DepthwiseConvTranspose2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        # out = (in-1)*stride -2*padding + (kernel-1) + output_padding
        # output_padding = 1 + 2*padding - kernel + 1
        #                = 2*padding +2 -kernel
        output_padding = 2 * (kernel_size//2) + 2 - kernel_size
        self.layer = nn.ConvTranspose2d(in_channels,
                                        in_channels,
                                        kernel_size,
                                        stride=(stride, 1),
                                        padding=(0, kernel_size//2),
                                        output_padding=output_padding,
                                        groups=in_channels)
        
        def forward(self, x):
            return self.layer(x)
        
class Conv2d(nn.Module):
    def __init__(self, width, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.block=nn.Sequential(
            LayerNorm2d(in_channels, width),
            PointwiseConv2d(in_channels, out_channels),
            nn.ReLU(),
            LayerNorm2d(out_channels, width),
            DepthwiseConv2d(out_channels, kernel_size, stride=stride),
            nn.ReLU(),
            LayerNorm2d(out_channels, width//2),
            PointwiseConv2d(out_channels, out_channels)
        )
    def foward(self, x):
        return self.block(x)

class ConvTranspose2d(nn.Module):
    def __init__(self, width, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.block=nn.Sequential(
            LayerNorm2d(in_channels, width),
            PointwiseConv2d(in_channels, out_channels),
            nn.ReLU(),
            LayerNorm2d(out_channels, width),
            DepthwiseConvTranspose2d(out_channels, kernel_size, stride=stride),
            nn.ReLU(),
            LayerNorm2d(out_channels, width*2),
            PointwiseConv2d(out_channels, out_channels)
        )
    def foward(self, x):
        return self.block(x)
    
class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x):
        return rearrange(x, self.pattern)
