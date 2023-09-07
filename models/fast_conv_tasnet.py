'''
 Facebook (fair) Denoiser から拝借。
'''
import pytorch_lightning as pt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import math
from einops import rearrange
#from models.ctc import CTCBlock
#from models.unet import UNet
from models.conv_tasnet_re import ConvTasNet
from models.e3net import LearnableEncoder, LearnableDecoder

class FastConvTasNet(ConvTasNet):
    def __init__(self, config:dict) -> None:
        super().__init__(config)
        assert config['tasnet']['resample'] == 1
        self.input_encoder = LearnableEncoder(chout=config['tasnet']['feat_channels'])
        self.output_decoder = LearnableDecoder(chin=config['tasnet']['feat_channels'])
        
    def valid_length(self, length):
        padding=200
        kernel_size=400
        stride=160
        output_padding=0
        length = int ( (length + 2 * padding - kernel_size - 1 - 1)/stride + 1 )
        # learnable decoder
        # n+1 -> 2n 
        length = int ( (length - 1) * stride - 2*0 + kernel_size - 1 + output_padding + 1)
        return length
    
    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        if mix.dim() == 2:
            mix = rearrange(mix, 'b (c t) -> b c t', c=1)
        if enr.dim() == 2:
            enr = rearrange(enr, 'b (c t) -> b c t', c=1)

        x = mix
        intput_length = mix.shape[-1]
        _valid_length = self.valid_length(input_length)
        if _valid_length < input_length:
            _length = input_length
            while(1):
                _valid_length = self.valid_length(_length)
                if _valid_length >= _length:
                    break
                _length += 1
                
        x = F.pad(x, (0, _length - input_length))
        
        x = self.input_encoder(x)
        y = self.input_encoder(enr)
        
        x, y, _ = super().forward(x, y)

        x = self.output_decoder(x)

        x = rearrange(x, 'b c t -> b (c t)')
        # len(x) > length
        return x[...,:input_length], y, None
    
