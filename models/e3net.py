
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
    def __init__(self, chin=1, chout=2048, kernel_size=400, stride=160, padding=200) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(chin, chout, kernel_size, stride, padding)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.encoder(x)
    
class LearnableDecoder(nn.Module):
    def __init__(self, chin=2048, chout=1, kernel_size=400, stride=160, padding=0, output_padding=0) -> None:
        super().__init__()
        self.decoder = nn.ConvTranspose1d(chin, chout, kernel_size, stride, padding, output_padding)

    def forward(self, x:Tensor) -> Tensor:
        return self.decoder(x)

class LSTMBlock(nn.Module):
    def __init__(self, dim=1024, hidden_dim=256, eps=1.e-8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.PReLU(),
            nn.Linear(dim, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim, eps=eps),
        )
        self.lstm = nn.LSTM(bidirectional=False,
                            num_layers=1,
                            hidden_size=hidden_dim,
                            input_size=hidden_dim
                            )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x:Tensor) -> Tensor:
        # input tensor shape: (B, C, T)
        x = self.block(x)
        x = rearrange(x, 'b t c -> t b c')
        x, _ = self.lstm(x)
        x = rearrange(x, 't b c -> b t c')
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
        self.linear=nn.Linear(chout, num_speakers)
        
    def forward(self, s:Tensor):
        s = self.conv(s) # B, C, T
        s = torch.mean(s, dim=-1)
        z = self.linear(s)
        return s, z

class FeatureConcatBlock(nn.Module):
    def __init__(self, x_dim=2048, s_dim=256, output_dim=256, eps=1.e-8):
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
        s = s.unsqueeze(1).repeat((1, x.shape[1], 1))
        y = torch.cat((x, s), dim=-1) 
        y = self.post(y)

        return y

class MaskingBlock(nn.Module):
    def __init__(self, x_dim=256, output_dim=2048):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(x_dim, output_dim),
                                 nn.Sigmoid()
        )
    def forward(self, src, mask):
        src = rearrange(src, 'b c t -> b t c')
        mask = self.pre(mask)
        return src*mask
    
class E3Net(nn.Module):
    def __init__(self, config, spk_net):
        super().__init__()

        self.encoder = LearnableEncoder(**config['encoder'])
        self.speaker_block = spk_net
        self.concate_block = FeatureConcatBlock(**config['concat'])

        block=nn.ModuleList()
        for _ in range(config['depth']):
            block.append(LSTMBlock())
        self.lstm_block = nn.Sequential(*block)
        
        self.masking_block = MaskingBlock(**config['masking'])
        self.decoder = LearnableDecoder(**config['decoder'])

    '''
    def valid_length(self, length):
        #length = math.ceil(length * self.resample)
        # learnable encoder 
        # 2n or 2n+1 -> n+1
        padding=200
        kernel_size=400
        stride=160
        output_padding=0
        length = int ( (length + 2 * padding - kernel_size - 1 - 1)/stride + 1 )
        # learnable decoder
        # n+1 -> 2n 
        length = int ( (length - 1) * stride - 2*0 + kernel_size - 1 + output_padding + 1)
        
        return length
    '''
    
    def forward(self, x:Tensor, s:Tensor):
        _, input_length = x.shape
        x = x.unsqueeze(1) # B, T -> B, 1, T
        # encoder
        x = self.encoder(x) # B, C, T
        # speaker block
        s = s.unsqueeze(1) # B, T -> B, 1, T
        s = self.encoder(s)
        embed = self.speaker_block(s)

        y = rearrange(x, 'b c t -> b t c')
        y = self.concate_block(y, embed)

        # lstm block
        y = self.lstm_block(y)

        # masking
        y = self.masking_block(x, y)

        # decoder block 
        y = rearrange(y, 'b t c -> b c t')
        y = self.decoder(y)
        y = rearrange(y, 'b c t -> b (c t)')

        _, output_length = y.shape
        start = (output_length - input_length)//2
        return y[:, start:start+input_length], embed

if __name__ == '__main__':
    with open("config.yaml", 'r') as yf:
        config = yaml.safe_load(yf)

    model = E3Net(config['e3net'])
    #length = 65536
    length = 161231
    #length = 1600
    x = torch.rand(4, length)
    s = torch.rand(4, length)
    y, _, _ = model(x, s)
    print(x.shape)
    print(y.shape)
    
