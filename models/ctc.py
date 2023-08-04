import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import argparse
import math

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm(x)
        x = rearrange(x, 'b t c -> b c t')
        return x
    
class CTCBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.kernel_size = config['kernel_size']  # kernel_size = 8
        self.stride = config['stride']            # stride = 4
        self.padding = config['padding']          # padding = kernel_size//2 = 4
        self.chout = config['chout']              # chout = 256
        self.outdim = config['outdim']
        
        self.pre = nn.Sequential(
            LayerNorm(dim=1),
            nn.Conv1d(in_channels=1,
                      out_channels=self.chout,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding),
            nn.PReLU(),
            LayerNorm(dim=self.chout),
            nn.Conv1d(in_channels=self.chout,
                      out_channels=self.chout,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding),
            nn.PReLU(),
            LayerNorm(dim=self.chout),
            nn.Conv1d(in_channels=self.chout,
                      out_channels=self.chout,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding),
            nn.PReLU(),
            LayerNorm(dim=self.chout),
        )
        self.fc = nn.Linear(self.chout, self.outdim)
        
    def forward(self, x):
        # x: (B, C, T)
        y = self.pre(x)
        y = rearrange(y, 'b c t -> b t c')
        y = self.fc(y)
        return y
    
    def valid_length(self, length):
        length = math.ceil((length + 2* self.padding - self.kernel_size)/self.stride) + 1
        length = math.ceil((length + 2* self.padding - self.kernel_size)/self.stride) + 1
        length = math.ceil((length + 2* self.padding - self.kernel_size)/self.stride) + 1
        return length
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    
