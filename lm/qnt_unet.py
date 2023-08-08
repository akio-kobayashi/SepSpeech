import pytorch_lightning as pt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import math
from einops import rearrange
from lm.qnt_modules  import QntSpeakerNetwork, QntSpeakerAdaptationLayer, PointwiseConv2d, DepthwiseConv2d, DepthwiseConvTranspose2d, Conv2d, ConvTranspose2d
class Rearrange(nn.Module):

def get_padded_value(x, valid_length):
    assert x.dim() == 3
    B, C, T = x.shape
    assert T > valid_length
    pad = torch.zeros((1, C, 1))
    pad[:, :, 0] = mixture[:, :, -1]
    pad = pad.repeat(1, 1, T-valid_length)
    
    return torch.concat ([x, pad], dim=-1)
    
class QntEncoder(nn.Module):
    def __init__(self, sym_dim, emb_dim, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        assert kernel_size % 2 == 0
        self.embed = nn.Embedding(sym_dim, emb_dim)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding=kernel_size//2
                    )
        
    def forward(self, x):
        x = self.embed(x)
        x = self.conv(x)  # (B C T) -> (B C' T H)
        return x

class QntDecoder(nn.Module):
    def __init__(self, sym_dim, emb_dim, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        assert kernel_size % 2 == 0
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              paddig=kernel_size//2
                    )
        self.linear = nn.Linear(emb_dim, sym_dim)
        
    def forward(self, x):
        x = self.conv(x)   # (B C T H) -> 
        x = self.linear(x) # (B C T O)
        return x
    
class TFEncoder(nn.Module):
    def __init__(self, dim, n_layers=5, n_heads=8):
        super(TFEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=n_heads,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=n_layers)

    def forward(self, x, mask=None, padding_mask=None):
        return self.encoder(x, mask, padding_mask)

class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi,
                          num_layers=layers,
                          hidden_size=dim,
                          input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden

class UNet(nn.Module):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.normalize = config['normalize']
        self.floor = config['floor']
        self.depth = config['depth']

        encodec_dim=config['encodec_dim']       # 8
        cb_size=config['encodec_codebook_size'] # 1024
        emb_size=config['embedding size']
        encdec_kernel_size=config['encdec_kernel_size']
        in_channels=config['in_channels']
        mid_channels=config['mid_channels']
        out_channels=config['out_channels']
        max_channels=config['max_channels']
        self.kernel_size=config['kernel_size']
        growth=config['growth']
        self.stride=config['stride']

        self.qnt_encoder = QntEncoder(cb_size, emb_size, encodec_dim, in_channels, encdec_kernel_size)
        self.qnt_decoder = QntDecoder(cb_size, emb_size, in_channels, encodec_dim, encdec_kernel_size)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.transform = nn.ModuleList()

        for index in range(self.depth):
            encode = []
            encode += [
                Conv2d(emb_size, in_channels, mid_channels, self.kernel_size, self.stride)
            ]
            self.encoder.append(nn.Sequential(*encode))

            transf = []
            transf += [
                QntSpeakerAdaptationLayer(config['adpt_type'], [emb_size, emb_size])
            ]
            self.transform.append(nn.Sequential(*transf))

            decode = []
            decode += [
                ConvTranspose2d(emb_size, mid_channels, out_channels, self.kernel_size, self.stride)
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            out_channels = mid_channels
            in_channels = mid_channels
            mid_channels = min(int(growth * mid_channels), max_channels)
            
        self.lstm=None
        self.attention=None
        if config['unet']['attention'] is False:
            self.lstm = BLSTM(emb_size, bi=not config['unet']['causal'])
        else:
            self.attention = TFEncoder(emb_size)

        self.speaker = QntSpeakerNetwork(emb_size,
                                         mid_channels,
                                         mid_channels,
                                         kernel_size,
                                         num_speakers
                                         )
            
    def valid_length(self, length):
        #length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size)/self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        #length = int(math.ceil(length/self.resample))
        return int(length)
    
    @property
    def total_stride(self):
        #return self.stride ** self.depth // self.resample
        return self.stride ** self.depth

    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        if mix.dim() == 2:
            mix = mix.unsqueeze(1) 
        
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=1, keepdim=True)
            mix = mix/(self.floor + std)
        else:
            std = 1

        B, C, T, H = mix.shape
        mix = rearrange('b c t h -> b c h t')
        mix = get_padded_value(mix, self.valid_length(T))
        mix = rearrange('b c h t -> b c t h')
        
        skips = []
        enc_s, y = None,None
        if enr is not None:
            enc_s, y = self.speaker(enr)
        for n, [encode, transform] in enumerate(zip(self.encoder, self.transform)):
            x = encode(x)
            if enc_s is not None:
                x = transform(x, enc_s)
            skips.append(x)
            
        if self.lstm is not None:
            # B C T H -> T (B C) H
            x = rearrange(x, 'b c t h -> t (b c) h')
            x, _ = self.lstm(x)
            x = rearrange(x, 't (b c) h -> b t c h', b=B)
        else:
            # B C T H -> (B C) T H
            x = rearrange(x, 'b c t h -> (b c) t h')
            x = self.attention(x)
            x = rearrange(x, '(b c) t h -> b c t h', b=B)
            
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[...,:x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)
        x = rearrange(x, 'b c t -> b (c t)')
        x = x[..., :length]
        return std * x, y
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    model = UNet(config['unet'])
    
