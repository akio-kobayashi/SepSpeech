import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from typing import Tuple

class LayerScale(nn.Module):
    def __init__(self, dim:int, scale:float) -> None:
        super().__init__()
        self.chwise_scaler = nn.Parameter(torch.ones(dim) * scale)
        #torch.register_parameter('chwise_scaler', chwise_scaler)

    def forward(self, x:Tensor) -> Tensor:
        # x: b c t f
        x = rearrange(x, 'b c t f-> b t f c')
        x = torch.mul(x, self.chwise_scaler)
        x = rearrange(x, 'b t f c -> b c t f')
        return x

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, bias=True)->None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=bias)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.depthwise(x)
    
class PointwiseConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, bias=False)->None:
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x:Tensor) -> Tensor:
        return self.pointwise(x)

class ConvNeXTBlock(nn.Module):
    def __init__(self, in_channels:int, kernel_size:int, padding=0, scale=0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseConv(in_channels, kernel_size, padding),
            LayerNorm(in_channels),
            PointwiseConv(in_channels, in_channels*4),
            nn.GELU(),
            PointwiseConv(in_channels*4, in_channels),
            LayerScale(in_channels, scale)
        )
    def forward(self, x:Tensor) -> Tensor:
        # (b 1 t f)
        return self.block(x)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim) 

    def forward(self, x):
        x = rearrange(x, 'b c t f -> b t f c')
        x = self.layer_norm(x)
        x = rearrange(x, 'b t f c -> b c t f')
        return x

class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pointwise1 = PointwiseConv(in_channels, out_channels)
        self.kernel1 = DepthwiseConv(in_chanels=in_channels,
                                     kernel_size=3,
                                     padding=3//2
                                     )
        self.kernel2 = DepthwiseConv(in_chanels=in_channels,
                                     kernel_size=3,
                                     padding=3//2
                                     )
        self.kernel3 = DepthwiseConv(in_chanels=in_channels,
                                     kernel_size=3,
                                     padding=3//2
                                     )
        self.pointwise2 = PointwiseConv(in_channels, out_channels)
    
    def forward(self, x):
        assert x.shape[-1]%4 == 0
        n_dim = x.shape[-1]//4
        y = self.pointwise1(x)
        z = torch.zero_like(y)
        z[:, :, :, :n_dim] = y[:, :, :, :n_dim]
        z[:, :, :, n_dim:2*n_dim] = self.kernel1(y[:, :, :, n_dim:2*n_dim])
        z[:, :, :, 2*n_dim:3*n_dim] = self.kernel2(z[:, :, :, n_dim:2*n_dim] + y[:, :, :, n_dim:2*n_dim])
        z[:, :, :, 3*n_dim:] = self.kernel3(z[:, :, :, 2*n_dim:3*n_dim] + y[:, :, :, 3*n_dim:])
        z = self.pointwise2(z)
        return z

def RepeatConvNeXTBlock(cntf_channels, kernel_size, padding=0, repeat=3):
    repeats = []
    for n in range(repeat):
        repeats.append(ConvNeXTBlock(cntf_channels, kernel_size, padding))
    return nn.Sequential(*repeats)

class CNTF(nn.Module):
    def __init__(self, dim=80, depth=2, cntf_channels=104, output_dim=1024, kernel_size=3, repeat=3) -> None:
        super().__init__()
        self.kernel_size=kernel_size
        self.repeat=repeat
        self.cntf = nn.Sequential(
            nn.Conv2d(1, cntf_channels, kernel_size, stride=2, padding=kernel_size//2),
            LayerNorm(cntf_channels),
            RepeatConvNeXTBlock(cntf_channels, kernel_size, padding=kernel_size//2, repeat=repeat),
            LayerNorm(cntf_channels),
            nn.Conv2d(cntf_channels, 2*cntf_channels, kernel_size, stride=2, padding=kernel_size//2),
            RepeatConvNeXTBlock(2*cntf_channels, kernel_size, padding=kernel_size//2, repeat=repeat),
            LayerNorm(2*cntf_channels),
            #nn.Conv2d(2*cntf_channels, 3*cntf_channels, kernel_size=(3,1), stride=(2,1), padding=(kernel_size//2, 0)),
            #RepeatConvNeXTBlock(3*cntf_channels, kernel_size, padding=kernel_size//2, repeat=repeat),
            #LayerNorm(3*cntf_channels),
        )
        #self.linear = nn.Linear(3*cntf_channels*(dim//4), output_dim)
        self.linear = nn.Linear(2*cntf_channels*(dim//4), output_dim)

    def forward(self, x:Tensor) -> Tensor:
        # x (b t f -> b 1 t f)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cntf(x)
        # x (b c t f) -> (b t (c f))
        x = rearrange(x, 'b c t f -> b t (c f)')
        return self.linear(x)
    
    def _valid_lengths(self, input_lengths, kernel_size=3, stride=1, padding=0, dilation=1.)->list:
        leng=[]
        for l in input_lengths:
            l = int(np.floor((l + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1))
            leng.append(l)
        return leng

    def valid_lengths(self, input_lengths:list) -> list:
        leng = self._valid_lengths(input_lengths, self.kernel_size, stride=2, padding=self.kernel_size//2)   # Conv2d T=T//2
        for n in range(self.repeat):
            leng = self._valid_lengths(leng, self.kernel_size, stride=1, padding=self.kernel_size//2)        # CNTF
        leng = self._valid_lengths(leng, self.kernel_size, stride=2, padding=self.kernel_size//2)            # Conv2d T=T//2
        for n in range(self.repeat):
            leng = self._valid_lengths(leng, self.kernel_size, stride=1, padding=self.kernel_size//2)        # CNTF
        #leng = self._valid_lengths(leng, self.kernel_size, stride=2, padding=self.kernel_size//2)            # Conv2d T=T//2
        #for n in range(self.repeat):
        #    leng = self._valid_lengths(leng, self.kernel_size, stride=1, padding=self.kernel_size//2)        # CNTF
        
        return leng
