
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.speaker import SpeakerNetwork
from typing import Tuple
import argparse
import yaml
import math
from einops import rearrange

'''
    Encoder 
'''
class LearnableEncoder(nn.Module):
    def __init__(self, chin=1, chout=2048, kernel_size=400, stride=160) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(chin, chout, kernel_size, stride)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.encoder(x)
    
class LearnableDecoder(nn.Module):
    def __init__(self, chin=2048, chout=1, kernel_size=400, stride=160) -> None:
        super().__init__()
        self.decoder = nn.ConvTranspose1d(chin, chout, kernel_size, stride)

    def forward(self, x:Tensor) -> Tensor:
        return self.decoder(x)

class LSTMBlock(nn.Module):
    def __init__(self, dim=1024, hidden_dim=256, eps=1.e-8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.PReLU(),
            nn.Linear(dim, dim),
            nn.PReLU(),
            nn.LayerNorm(dim, eps=eps),
        )
        self.lstm = nn.LSTM(bidirectional=False,
                            num_layers=1,
                            hidden_size=hidden_dim,
                            input_size=dim,
                            proj_size=dim)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)

    def forward(self, x:Tensor) -> Tensor:
        # input tensor shape: (B, C, T)
        x = self.block(x)
        x = rearrange('b t c -> t b c')
        x = self.lstm(x)
        x = rearrange('t b c -> b t c')
        x = x + self.norm1(x)

        return self.norm2(x)

class SpeakerBlock(nn.Module):
    def __init__(self, chin=2048, chout=256, kernel_size=3, num_speakers=1000):
        super().__init__()
        self.conv = nn.Conv1d(
            chin,      # 2048
            chout,     # 256
            kernel_size,      # 16
            1,
            kernel_size//2, # padding
            1
        )
        self.act = nn.ReLU()
        self.linear=nn.Linear(chin, num_speakers)
        
    def forward(self, s:Tensor):
        s = self.conv(s)
        s = self.linear(s)
        return s

class FeatureConcatBlock(nn.Module):
    def __init__(self, x_dim=2048, s_dim=256, output_dim=1024, eps=1.e-8):
        super().__init__()
        
        self.pre = nn.Sequential(nn.PReLU(),
                                 nn.LayerNorm(x_dim, eps=eps)
                                )
        self.post = nn.Sequential(nn.Linear(x_dim+s_dim, output_dim),
                                  nn.PReLU(),
                                  nn.LayerNorm(output_dim, eps=eps)
                                  )
        
    def forward(self, x:Tensor, s:Tensor):
        ''' x (B, T, C_1), s (B, C_2)'''
        x = self.pre(x)
        s = s.unsqueeze(1).repeat((1, 1, x.shape[1]))
        y = torch.cat((x, s), dim=-1) 
        y = self.post(y)

        return y

class MaskingBlock(nn.Module):
    def __init__(self, x_dim=1024, output_dim=2048):
        self.pre = nn.Sequential(nn.Linear(x_dim, output_dim),
                                 nn.Sigmoid()
        )
    def forward(self, src, mask):
        mask = self.pre(mask)
        return src*mask
    
class E3Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.depth = 4

        self.encoder = LearnableEncoder()
        self.speaker_block = SpeakerBlock()
        self.concate_block = FeatureConcatBlock()

        self.lstm_block=nn.ModuleList()
        for _ in self.depth:
         self.lstm_block.append(LSTMBlock())

        self.masking_block = MaskingBlock()
        self.decoder = LearnableDecoder()

    def valid_length(self, length):
        #length = math.ceil(length * self.resample)
        # learnable encoder 
        # 2n or 2n+1 -> n+1
        length = int ( (length + 2 * self.encoder_padding - self.encoder_kernel - 1 - 1)/self.encoder_strie + 1 )
        # learnable decoder
        # n+1 -> 2n 
        length = int ( (length - 1) * self.encoder_stride - 2*self.decoder_padding + self.decoder_kernel - 1 + self.decoder_output_padding + 1)
        
        return length

    def forward(self, x:Tensor, s:Tensor):
        _, orig_length = x.shape
        x = x.unsqueeze(1) # B, T -> B, 1, T
        # encoder
        x = self.encoder(x) # B, C, T
        # speaker block
        s = self.speaker_block(s)

        y = rearrange(x, 'b c t -> b t c')
        y = self.concate_block(y, s)

        # lstm block
        y = self.lstm_block(y)

        # masking
        y = self.masking_block(x, y)

        # decoder block 
        y = rearrange(y, 'b t c -> b c t')
        y = self.decoder(y)
        y = rearrange(y, 'b c t -> b (c t)')

        return y[:, :orig_length]

if __name__ == '__main__':
    with open("../config.yaml", 'r') as yf:
        config = yaml.safe_load(yf)

    model = E3Net(config)
    x = torch.rand(4, 65536)
    s = torch.rand(4, 65536)
    y = model(x, s)
    print(x.shape)
    print(y.shape)
    