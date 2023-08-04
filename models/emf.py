import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.models.emformer import Emformer
import numpy as np
from einops import rearrange
import argparse, yaml
import math

class EmformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_dim = config['input_dim']
        self.num_heads = config['num_heads']
        self.ffn_dim = config['ffn_dim']
        self.num_layers = config['num_layers']
        self.segment_length = config['segment_length']
        self.dropout = config['dropout']                           # 0.0
        self.activation = config['activation']                     # 'relu'
        self.left_context_length = config['left_context_length']   # 0
        self.right_context_length = config['right_context_length'] # 0
        self.max_memory_size = config['max_memory_size']           # 0
        self.tan_on_mem = config['tanh_on_mem']                    # False

        self.emformer = Emformer(**config)
        
    def forward(self, x, lengths):
        # x: (B, C, T) -> (B, T, C)
        x = F.pad(x, (0, self.right_context_length), value=0.)
        x = rearrange(x, 'b c t -> b t c')
        y, _ = self.emformer(x, lengths)
        y = rearrange(y, 'b t c -> b c t')
        
        return y

    def infer(self, x, lengths, states=None):
        # x (B, C, T+right_context) -> (B, T+right_context, C)
        x = rearrange(x, 'b c t -> b t c')
        y, _, states = self.emformer(x, lengths, states)
        y = rearrange(y, 'b t c -> b c t')
        
        return y, states
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    model=EmformerBlock(config['emformer'])
    x = torch.rand(4, 256, 512)
    length=torch.tensor([512])
    y = model(x, length)
    print(y.shape)
