import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from einops import rearrange
import argparse
import math
from models.e3net import LearnableEncoder
from inspect import isfunction
from models.conformer import ConformerBlock

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(x):
    return x is not None

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm(x)
        x = rearrange(x, 'b t c -> b c t')
        return x

class DownSample(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        #self.conv = nn.Conv1d(dim*2, default(dim_out, dim), 1)
        self.conv = nn.Conv1d(dim, default(dim_out, dim), 3, 2, padding=3//2)

    def forward(self, x):
        #x = rearrange(x, 'b c (t p) -> b (c p) t', p=2)
        return self.conv(x)

class CTCBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = LearnableEncoder(chout=config['dim']) # (B C T) C=2048 channel
        self.downsampler = DownSample(config['dim'], None)
        self.post_downsampler = DownSample(config['dim'], None) if config['downsample'] == 4 else None
        ''' 
        #self.conformer = torchaudio.models.Conformer(
        #    input_dim=config['dim'],
        #    num_heads=config['num_heads'],
        #    ffn_dim=config['ffn_dim'],
        #    num_layers=config['num_layers'],
        #    depthwise_conv_kernel_size=config['kernel_size'],
        #    dropout=config['dropout'],
        #)
        '''
        layers=[]
        for n in range(config['num_layers']):
            layers.append(
                ConformerBlock(
                    dim = config['dim'],
                    dim_head = config['dim']//config['num_heads'],
                    heads = config['num_heads'],
                    ff_mult = config['ffn_dim']//config['dim'],
                    conv_expansion_factor = 2,
                    conv_kernel_size = 31,
                    attn_dropout = 0.,
                    ff_dropout = 0.,
                    conv_dropout = 0.
                )
            )
        self.conformer = nn.Sequential(*layers) # (B, T, F)
        
        self.fc = nn.Linear(config['dim'], config['outdim'])
        
    def forward(self, x):
        # x: (B, T, C)
        if x.dim() == 2:
            x = rearrange(x, 'b (c t) -> b t c', c=1)
        x = rearrange(x, 'b t c -> b c t')
        y = self.encoder(x)
        y = self.downsampler(y)
        if self.post_downsampler is not None:
            y = self.post_downsampler(y)
        y = rearrange(y, 'b c t -> b t c')
        y = self.conformer(y)
        y = self.fc(y)
        return y
    
    def valid_length(self, length):
        length = math.floor( (length + 2* 200 - (400-1)) /160 + 1)
        length = math.floor( (length + 2*1 - (3-1)/ 2) + 1)
        if self.post_downsampler is not None:
            length = math.floor( (length + 2*1 - (3-1)/2) + 1)
        return length

    def pad(self, x):
        '''
        B, T, C = x.shape
        p = 4-T%4 if self.post_downsampler is not None else 2-T%2
        x = rearrange(F.pad(rearrange(x, 'b t c -> b c t'), (0, p)), 'b c t -> b t c')
        '''
        return x
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

