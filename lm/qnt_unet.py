import pytorch_lightning as pt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import math
from einops import rearrange
#from models.qnt_encode import EncodecQuantizer
#from models.ctc import CTCBlock

class QuantizeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=3//2
            ),
        )
    def forward(self, x):
        # (b 8 t) -> (b out_channels t emedding_dim)
        return self.block(x)
    
class QuantizeDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=8, embedding_dim=256, num_embeddings=1024):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=3//2
            ),
            nn.Linear(embedding_dim, num_embeddings)
        )
    def forward(self, x):
        # (b in_channels, t, embedding_dim)  -> (b 8 t 1024)
        return self.block(x)

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
        ''' x (b c1 t h), s (b c2 t h)'''
        B, C, T, H = x.shape
        x = rearrange(x, 'b c t h -> b t h c')
        y = self.pre(x)
        s = rearrange(s, 'b (t h c) -> b t h c', t=1, c=1).repeat((1, T, H, 1))
        y = torch.cat((x, s), dim=-1) 
        y = x + self.post(y)
        y = rearrange(y, 'b t h c -> b c t h')
        return y
    
class SpeakerNetwork(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, num_speakers:int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size//2, # padding
        )
        self.act = nn.ReLU()
        self.linear=nn.Linear(out_channels, num_speakers)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        # tensor: (b c t h)
        x = self.conv(x)
        y = torch.mean(torch.mean(x, dim=-2), dim=-1) # (b c)
        z = self.linear(self.act(y)) # (b s)
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
        B, C, T, H = x.shape
        x = rearrange(x, 'b c t h -> b t (c h)')
        x = self.encoder(x, mask, padding_mask)
        x = rearrange(x, 'b t (c h) -> b c t h', c=C, h=H)
        return x
    
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
        B, C, T, H = x.shape
        x = rearrange(x, 'b c t h -> t b (c h)')
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return rearrange(x, 't b (c h) -> b c t h', c=C), hidden

class EncoderBlock(nn.Module):
    def __init__(self, chin:int, chout:int, kernel_size:int=3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(chin*4, chout, kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(chout, chout, kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU()
        )
    
    def forward(self, x:Tensor) -> Tensor:
        x = rearrnage(x, 'b c (t p1) (h p2) -> b (c*p1*p2) t h', p1=2, p2=2)
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, chin:int, chout:int, kernel_size:int=3, append=False) -> None:
        super().__init__()
        block = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(chin, chout, kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(chin, chout, kernel_size, stride=1, padding=kerlen_size//2),
        ]
        if append:
            block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x:Tensor) -> Tensor:
        return self.block(x)
    
class QntUNet(nn.Module):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.normalize = config['qnt_unet']['normalize']
        self.floor = config['qnt_unet']['floor']
        #self.resample = config['qnt_unet']['resample']
        self.depth = config['qnt_unet']['depth']

        in_channels=config['qnt_unet']['in_channels']
        mid_channels=config['qnt_unet']['mid_channels']
        out_channels=config['qnt_unet']['out_channels']
        max_channels=config['qnt_unet']['max_channels']
        self.kernel_size=config['qnt_unet']['kernel_size']
        growth=config['qnt_unet']['growth']
        #self.rescale=config['qnt_unet']['rescale']
        self.stride=config['qnt_unet']['stride'] # stride shoud be 2
        #reference=config['qnt_unet']['reference']
        embedding_channels=config['qnt_unet']['embedding_channels']
        num_embeddings=config['qnt_unet']['num_embeddings']
        
        #self.quantizer = EncodecQuantizer() # inference mode
        self.qnt_encoder = QuantizeEncoder(embedding_channels, in_channels)
        self.qnt_decoder = QuantizeDecoder(out_channels, embedding_channels, embedding_dim, num_embeddings)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.transform = nn.ModuleList()
        for index in range(self.depth):
            self.encoder.append(EncoderBlock(
                in_channels,
                mid_channels,
                self.kernel_size
            ))

            self.transform.append(ConcatBlock(mid_channels,
                                              mid_channels,
                                              mid_channels)
                                  )
            self.decoder.insert(0,
                                DecoderBlock(
                                    mid_channels,
                                    out_channels,
                                    self.kernel_size,
                                    append = False if index == 0 else True
                                ))
            out_channels = mid_channels
            in_channels = mid_channels
            mid_channels = min(int(growth * mid_channels), max_channels)

        self.lstm=None
        self.attention=None
        channels = in_channels//self.total_stride() * embedding_dim//self.total_stride()
        if config['qnt_unet']['attention'] is False:
            self.lstm = BLSTM(channels, bi=not config['qnt_unet']['causal'])
        else:
            self.attention = TFEncoder(channels)
            
        #if self.rescale:
        #    rescale_module(self, reference=reference)
        self.speaker = SpeakerNetwork(spk_encoder,
                                      config['qnt_unet']['mid_channels'],
                                      config['qnt_unet']['mid_channels'],
                                      config['qnt_unet']['kernel_size'],
                                      config['qnt_unet']['num_speakers'])
        '''
        self.ctc_block=None        
        if config['ctc']['use']:
            self.ctc_block = CTCBlock(config['ctc']['parameters'])
        '''

    def valid_length(self, length):
        for idx in range(self.depth):
            length = length // 2 # downsample
            length = int (length + 2 * self.kernel_size//2 - self.kernel_size + 1)
        for idx in range(self.depth):
            length = length * 2 # updample
            length = int (length + 2 * self.kernel_size//2 - self.kernel_size + 1)
        return int(length)

    '''
    def valid_length_ctc(self, length):
        length = self.valid_length(length)
        length = self.ctc_block.valid_length(length)
        return int(length)
    '''

    @property
    def total_stride(self):
        #return self.stride ** self.depth // self.resample
        return self.stride ** self.depth
    
    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        # (b emb_dim t emb_size) -> (b chin t emb_size)
        mix = self.qnt_encoder(mix)
        enr = self.qnt_encoder(enr)

        length = mix.shape[-1]
        mix = reshape(F.pad(reshape(mix, 'b c t h -> b c h t'), (0, self.valid_length(length) - length)),'b c h t -> b c t h')
        
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=1, keepdim=True)
            mix = mix/(self.floor + std)
        else:
            std = 1
        
        x = mix
        '''
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        '''
        skips = []
        enc_s, y = self.speaker(enr)        
        for n, [encode, transform] in enumerate(zip(self.encoder, self.transform)):
            x = encode(x)
            if enc_s is not None:
                x = transform(x, enc_s)
            skips.append(x)
        if self.lstm is not None:
            x, _ = self.lstm(x)
        else:
            x = self.attention(x)

        #for decode, transform in zip(self.decoder, self.transform_d):
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[...,:x.shape[-1]]
            x = decode(x)

        '''
        z = None
        if self.ctc_block is not None:
            z = self.ctc_block(x)
        '''
        '''
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)
        '''
        
        x = rearrange(x, 'b c t -> b (c t)')
        x = x[..., :length]
        return std * x, y, z
        
