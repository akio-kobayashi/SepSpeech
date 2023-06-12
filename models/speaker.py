import math
import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import torchaudio
from typing import Tuple

'''
    エンロール話者
'''
class SpeakerNetwork(nn.Module):
    def __init__(self, encoder, in_channels:int, out_channels:int, kernel_size:int, num_speakers:int) -> None:
        super(SpeakerNetwork, self).__init__()
        from models.sepformer import Encoder
        self.encoder = encoder
        self.conv = nn.Conv1d(
            in_channels,      # 256
            out_channels,
            kernel_size,      # 16
            1,
            kernel_size//2, # padding
            1
        )
        self.act = nn.ReLU()
        self.linear=nn.Linear(out_channels, num_speakers)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        y = self.encoder(x)
        y = self.conv(y)
        y = torch.mean(y, dim=-1) # (B, C)
        z = self.linear(self.act(y)) # (B, S)
        return y, z

class SpeakerAdaptationLayer(nn.Module):
    def __init__(self, adpt_type='factor', channels=None) -> None:
        super(SpeakerAdaptationLayer, self).__init__()
        self.linear=None
        self.adpt_type=adpt_type
        if adpt_type == 'residual' or adpt_type == 'concat':
            assert channels is not None
            self.linear = nn.Linear(channels*2, channels)

    def forward(self, x:torch.Tensor, s:torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            # (B, C, T) x (B, C)
            y = x * s.unsqueeze(-1)
            return y
        elif self.adpt_type == 'residual':
            s = s.unsqueeze(-1).repeat((1, 1, x.shape[-1]))
            y = self.linear(torch.cat((x, s), dim=-1))
            return x + y
        elif self.adpt_type == 'concat':
            s = s.unsqueeze(-1).repeat((1, 1, x.shape[-1]))
            y = self.linear(torch.cat((x, s), dim=-1))
            return y


'''        
class SpeakerAdaptationLayer(nn.Module):
    def __init__(self) -> None:
        super(SpeakerAdaptationLayer, self).__init__()

    def forward(self, x:torch.Tensor, s:torch.Tensor) -> torch.Tensor:
        # (B, C, T) x (B, C)
        y = x * s.unsqueeze(-1)
        return y

class ResSpeakerAdaptationLayer(nn.Module):
    def __init__(self, channels) -> None:
        super(ResSpeakerAdaptationLayer, self).__init__()
        self.linear = nn.Linear(channels*2, channels)
        
    def forward(self, x:torch.Tensor, s:torch.Tensor) -> torch.Tensor:
        s = s.unsqueeze(-1).repeat((1, 1, x.shape[-1]))
        y = self.linear(torch.cat((x, s), dim=-1))
        return x + y
'''
