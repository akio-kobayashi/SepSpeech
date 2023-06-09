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
    
def sinc(t):
    """sinc.
    :param t: the input tensor
    """
    return torch.where(t == 0, torch.tensor(1., device=t.device, dtype=t.dtype), torch.sin(t) / t)

def kernel_upsample2(zeros=56):
    """kernel_upsample2.
    """
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel

def kernel_upsample2(zeros=56):
    """kernel_upsample2.
    """
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel

def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape
    kernel = kernel_upsample2(zeros).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = torch.stack([x, out], dim=-1)
    return y.view(*other, -1)

def kernel_downsample2(zeros=56):
    """kernel_downsample2.
    """
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel

def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)

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

class EncoderBlock(nn.Module):
    def __init__(self, chin:int, chout:int, kernel_size:int, stride:int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(chin, chout, kernel_size, stride),
            nn.ReLU(),
            nn.Conv1d(chout, chout, kernel_size, 1),
            nn.ReLU()
        )
    
    def forward(self, x:Tensor) -> Tensor:
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, chin:int, chout:int, kernel_size:int, stride:int, append=False) -> None:
        super().__init__()
        block = [
            nn.Conv1d(chin, chin, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(chin, chout, kernel_size, stride),
        ]
        if append:
            block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x:Tensor) -> Tensor:
        return self.block(x)

class UNetRadio(nn.Module):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.normalize = config['unet']['normalize']
        self.floor = config['unet']['floor']
        self.resample = config['unet']['resample']
        self.depth = config['unet']['depth']

        in_channels=config['unet']['in_channels']
        mid_channels=config['unet']['mid_channels']
        out_channels=config['unet']['out_channels']
        max_channels=config['unet']['max_channels']
        self.kernel_size=config['unet']['kernel_size']
        growth=config['unet']['growth']
        self.rescale=config['unet']['rescale']
        self.stride=config['unet']['stride']
        reference=config['unet']['reference']

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(self.depth):
            encode = nn.ModuleList()
            encode.append(nn.Conv1d(in_channels, mid_channels, self.kernel_size, self.stride))
            encode.append(nn.ReLU())
            encode.append(nn.Conv1d(mid_channels, mid_channels, 1))
            encode.append(nn.ReLU())
            self.encoder.append(nn.Sequential(*encode))
            decode = nn.ModuleList()
            decode.append(nn.Conv1d(mid_channels, mid_channels, 1))
            decode.append(nn.ReLU())
            decode.append(nn.ConvTranspose1d(mid_channels, out_channels, self.kernel_size, self.stride))
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            out_channels = mid_channels
            in_channels = mid_channels
            mid_channels = min(int(growth * mid_channels), max_channels)

        self.lstm=None
        self.attention=None
        if config['unet']['attention'] is False:
            self.lstm = BLSTM(in_channels, bi=not config['unet']['causal'])
        else:
            self.attention = TFEncoder(in_channels)
            
        if self.rescale:
            rescale_module(self, reference=reference)
            
    def valid_length(self, length):
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size)/self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length/self.resample))
        return int(length)
    
    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix:Tensor) -> Tensor:
        if mix.dim() == 2:
            mix = mix.unsqueeze(1) 
        
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=1, keepdim=True)
            mix = mix/(self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for n, encode in enumerate(self.encoder):
            x = encode(x)
            skips.append(x)
        if self.lstm is not None:
            x = x.permute(2, 0, 1)
            x, _ = self.lstm(x)
            x = x.permute(1, 2, 0)
        else:
            x = x.permute(0, 2, 1)
            x = self.attention(x)
            x = x.permute(0, 2, 1)
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
        return std * x
        
