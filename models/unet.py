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
from models.ctc import CTCBlock

class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = rearrange(pe, '(b t) c -> b c t', b=1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B, C, T = x.shape
        x = x + self.pe[:, :, :T]
        return self.dropout(x)

class ConcatBlock(nn.Module):
    def __init__(self, input_dim, aux_dim, output_dim, eps=1.e-8):
        super().__init__()
        
        self.pre = nn.Sequential(nn.PReLU(),
                                 nn.LayerNorm(input_dim, eps=eps)
                                )
        self.post = nn.Sequential(nn.Linear(input_dim+aux_dim, output_dim),
                                  nn.PReLU(),
                                  nn.LayerNorm(output_dim, eps=eps)
                                )
        
    def forward(self, x:Tensor, s:Tensor):
        ''' x (b c1 t), s (b c2 t)'''
        B, C, T = x.shape
        y = rearrange(x, 'b c t -> b t c')
        y = self.pre(y)
        s = rearrange(s, 'b (c t) -> b t c', t=1).repeat((1, T, 1))
        y = torch.cat((y, s), dim=-1)
        y = self.post(y)
        y = rearrange(y, 'b t c -> b c t')
        return x + y
    
class MergeBlock(nn.Module):
    def __init__(self, in_channels:int, aux_channels:int, out_channels:int) -> None:
        super().__init__()
        #self.pos_enc = PositionEncoding(aux_channels, max_len=640000)
        #self.linear = nn.Linear(in_channels+aux_channels, out_channels, bias=False)
        
    def forward(self, x:Tensor,  s:Tensor):
        # factoring
        return x * rearrange(s, 'b (c t) -> b c t', t=1)
        #B, C, T = x.shape
        #s = self.pos_enc(rearrange(s, 'b (c t) -> b c t', t=1).repeat((1, 1, T)))
        #y = torch.concat([x, s], dim=-1)
        #z = self.linear(rearrange(y, 'b c t -> b t c'))
        #z = rearrnage(z, 'b t c -> b c t')
        #return x+z
    
class SpeakerNetwork(nn.Module):
    def __init__(self, encoder, in_channels:int, out_channels:int, kernel_size:int, num_speakers:int) -> None:
        super().__init__()
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
        # tensor: (b c t)
        y = self.encoder(x)
        y = self.conv(y)
        y = torch.mean(y, dim=-1) # (B, C)
        z = self.linear(self.act(y)) # (B, S)
        return y, z

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

class UNet(nn.Module):
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
        self.transform = nn.ModuleList()
        #self.transform_d = nn.ModuleList()
        for index in range(self.depth):
            encode = []
            encode += [
                nn.Conv1d(in_channels, mid_channels, self.kernel_size, self.stride),
                nn.ReLU(),
                nn.Conv1d(mid_channels, mid_channels, 1), nn.ReLU(),
            ]
            self.encoder.append(nn.Sequential(*encode))

            #transf = []
            #transf += [
            #    nn.ReLU(),
            #    nn.Linear(mid_channels, mid_channels)
            #]
            #self.transform.append(nn.Sequential(*transf))
            self.transform.append(ConcatBlock(mid_channels,
                                              mid_channels,
                                              mid_channels)
                                  )
            #transf_d = []
            #transf_d += [
                #nn.ReLU(),
                #nn.Linear(mid_channels, mid_channels)
                #]
            #self.transform_d.append(nn.Sequential(*transf_d))

            decode = []
            decode += [
                nn.Conv1d(mid_channels, mid_channels, 1), nn.ReLU(),
                nn.ConvTranspose1d(mid_channels,
                                   out_channels,
                                   self.kernel_size,
                                   self.stride),
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
            self.lstm = BLSTM(in_channels, bi=not config['unet']['causal'])
        else:
            self.attention = TFEncoder(in_channels)
            
        if self.rescale:
            rescale_module(self, reference=reference)
        from models.sepformer import Encoder

        spk_encoder = nn.Sequential(
            nn.Conv1d(config['unet']['in_channels'],
                      config['unet']['mid_channels'],
                      self.kernel_size,
                      self.stride),
            nn.ReLU(),
            nn.Conv1d(config['unet']['mid_channels'],
                      config['unet']['mid_channels'],
                      1),
            nn.ReLU()
        )
        
        self.speaker = SpeakerNetwork(spk_encoder,
                                      config['unet']['mid_channels'],
                                      config['unet']['mid_channels'],
                                      config['unet']['kernel_size'],
                                      config['unet']['num_speakers'])
  
        self.ctc_block=None        
        if config['ctc']['use']:
            self.ctc_block = CTCBlock(config['ctc']['parameters'])

    def valid_length(self, length):
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size)/self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length/self.resample))
        return int(length)
    
    def valid_length_ctc(self, length):
        length = self.valid_length(length)
        length = self.ctc_block.valid_length(length)
        return int(length)
    
    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        if mix.dim() == 2:
            mix = mix.unsqueeze(1) 
        if enr.dim() == 2:
            enr = enr.unsqueeze(1)
            
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
        enc_s, y = None,None
        if enr is not None:
            enc_s, y = self.speaker(enr)
        for n, [encode, transform] in enumerate(zip(self.encoder, self.transform)):
            x = encode(x)
            if enc_s is not None:
                x = transform(x, enc_s)
            skips.append(x)
        if self.lstm is not None:
            x = x.permute(2, 0, 1) # (b c t) -> (t b c)
            x, _ = self.lstm(x)
            x = x.permute(1, 2, 0) # (t b c) -> (b c t)
        else:
            x = x.permute(0, 2, 1) # (b c t) -> (b t c)
            x = self.attention(x)
            x = x.permute(0, 2, 1) # (b t c) -> (b c t)

        #for decode, transform in zip(self.decoder, self.transform_d):
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[...,:x.shape[-1]]
            #if enc_s is not None:
            #    x = rearrange(x, 'b c t -> b t c')
            #    x = transform(x)
            #    x = rearrange(x, 'b t c -> b c t')
            #    x = self.adpt(x, enc_s)
            x = decode(x)

        z = None
        if self.ctc_block is not None:
            z = self.ctc_block(x)
            
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)
        if x.shape[1] == 1:
            x = rearrange(x, 'b c t -> b (c t)')
            x = x[..., :length]
        else:
            x = x[..., :length]
        return std * x, y, z
        
