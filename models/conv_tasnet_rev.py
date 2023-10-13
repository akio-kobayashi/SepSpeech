# https://github.com/funcwj/conv-tasnet/blob/master/nnet/conv_tas_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.speaker import SpeakerNetwork, SpeakerAdaptationLayer
from typing import Tuple
import argparse
import yaml
import math
import models.unet as unet
from einops import rearrange
'''
    ConvTasNet
    SpeakerBeam等で使われる一般的な音源分離アプローチの実装
'''

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles


class Encoder(nn.Module):
    def __init__(self, in_channels:int, kernel_size:int) -> None:
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(
            1,
            in_channels,    # 256
            kernel_size,    # 20
            kernel_size//2, # 10
            kernel_size//2, # padding
            1
        )
        self.act = nn.ReLU()

    def forward(self, x:Tensor) -> Tensor:
        assert x.dim() == 3
        #assert x.dim() == 2 # (B, T)
        #x = x.unsqueeze(1) # add channel
        return self.act(self.conv(x))

    def valid_length(self, length:int) -> int:
        return self.conv.valid_length(length)
    
class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x:Tensor) -> Tensor:
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x

class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x:Tensor) -> Tensor:
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self) -> str:
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)

def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)

class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)
        #self.in_channels, self.out_channels, self.kernel_size, self.stride, _ = args

    def forward(self, x:Tensor, squeeze=False) -> Tensor:
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        #x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        x = super().forward(x)
        if squeeze:
            x = torch.squeeze(x)
        return x

    def valid_length(self, length:int) -> int:
        length = int ( (length + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)/self.stride[0] + 1 )
        return length
    
class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)
        
    def forward(self, x:Tensor, squeeze=False) -> Tensor:
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        #x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        x = super().forward(x)
        if squeeze:
            x = torch.squeeze(x)
        return x
    
    def valid_length(self, length:int)->int:
        length = (length - 1) * self.stride[0] -2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        return length
    
class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x:Tensor) -> Tensor:
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x

    def valid_length(self, length:int) -> int:
        length = self.conv1x1.valid_length(length)
        length = int ( (length + 2 * self.dconv.padding[0] - self.dconv.dilation[0] * (self.dconv.kernel_size[0] - 1) - 1)/self.dconv.stride[0] + 1 )
        length = int ( (length + 2 * self.sconv.padding[0] - self.sconv.dilation[0] * (self.sconv.kernel_size[0] - 1) - 1)/self.sconv.stride[0] + 1 )
        return length

class ConvTasNet(nn.Module):
    '''
        L: config['tasnet']['kernel_size'] = 20
        B: config['tasnet']['in_channels'] = 256
        N: config['tasnet']['enc_channels']= 256
        H: config['tasnet']['conv_channels'] = 512
        X: config['tasnet']['num_blocks'] = 8
        P: config['tasnet']['block_kernel_size'] = 3
        R: config['tasnet']['num_repeats'] =4
    '''
    def __init__(self,config,
                 norm="cLN",
                 non_linear="relu",
                 causal=False):
        super(ConvTasNet, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder_1d = Conv1D(config['tasnet']['feat_channels'],
                                 config['tasnet']['enc_channels'],
                                 config['tasnet']['kernel_size'],
                                 config['tasnet']['kernel_size']// 2,
                                 0)
        # keep T not change
        # T = int((xlen - L) / (L // 2)) + 1
        # before repeat blocks, always cLN
        self.ln = ChannelWiseLayerNorm(config['tasnet']['enc_channels'])
        # n x N x T => n x B x T
        self.proj = Conv1D(config['tasnet']['enc_channels'],
                           config['tasnet']['in_channels'],
                           1)
        # repeat blocks
        # n x B x T => n x B x T
        self.first_block = self._build_repeats(
            1,
            config['tasnet']['num_blocks'],
            in_channels=config['tasnet']['in_channels'],
            conv_channels=config['tasnet']['conv_channels'],
            kernel_size=config['tasnet']['block_kernel_size'],
            norm=norm,
            causal=causal)
        spk_encoder = Encoder(config['tasnet']['in_channels'],
                              config['tasnet']['kernel_size'])
        self.speaker = SpeakerNetwork(spk_encoder,
                                      config['tasnet']['in_channels'],
                                      config['tasnet']['in_channels'],
                                      config['tasnet']['block_kernel_size'],
                                      config['tasnet']['num_speakers'])
        self.adpt = SpeakerAdaptationLayer(config['speaker']['adpt_type'])
        self.repeats = self._build_repeats(
            config['tasnet']['num_repeats']-1,
            config['tasnet']['num_blocks'],
            in_channels=config['tasnet']['in_channels'],
            conv_channels=config['tasnet']['conv_channels'],
            kernel_size=config['tasnet']['block_kernel_size'],
            norm=norm,
            causal=causal)
        # output 1x1 conv
        # n x B x T => n x N x T
        # NOTE: using ModuleList not python list
        # self.conv1x1_2 = torch.nn.ModuleList(
        #     [Conv1D(B, N, 1) for _ in range(num_spks)])
        # n x B x T => n x 2N x T
        self.mask = Conv1D(config['tasnet']['in_channels'],
                           config['tasnet']['enc_channels'], 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d = ConvTrans1D(
            config['tasnet']['enc_channels'],
            config['tasnet']['feat_channels'],
            kernel_size=config['tasnet']['kernel_size'],
            stride=config['tasnet']['kernel_size'] // 2, bias=True)
        
        self.resample = config['tasnet']['resample']
        assert self.resample == 1
        #self.padding_value = 0
        #self.get_padding_value()

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    '''
    def valid_length(self, length):
        length = int(math.ceil(length*self.resample))
        length = int(math.ceil(length/self.resample))
        return int(length)
    '''
    
    def upsample(self, x):
        if self.resample == 2:
            x = unet.upsample2(x)
        elif self.resample ==4:
            x = unet.upsample2(x)
            x = unet.upsample2(x)
        return x
    
    def downsample(self, x):
        if self.resample == 2:
            x = unet.downsample2(x)
        elif self.resample == 4:
            x = unet.downsample2(x)
            x = unet.downsample2(x)
        return x
               
    def forward(self, x:Tensor, s:Tensor) -> Tuple[Tensor, Tensor]:
        # accept 3d tensor
        '''
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        '''
        # in case of only one utterance
        #if x.dim() == 2:
        #x = torch.unsqueeze(x, 0)
        #x = rearrange(x, '(b c) t -> b c t', c=1)
            
        # padding
        _, input_length = x.shape
        #if self.padding_value>0:
        #    x = self.get_padded_value(x)
        _valid_length = self.valid_length(input_length)
        if _valid_length < input_length:
            _length = input_length
            while(1):
                _valid_length = self.valid_length(_length)
                if _valid_length >= _length:
                    break
                _length += 1
        x = F.pad(x, (0, _length - input_length))
        
        # upsample
        #x = self.upsample(x)

        # n x 1 x S => n x N x T
        print(x.shape)
        print(self.encoder_1d(x).shape)
        w = F.relu(self.encoder_1d(x))
        # n x B x T
        y = self.proj(self.ln(w))
        # n x B x T
        y = self.first_block(y)
        spk, z = self.speaker(s)
        y = self.adpt(y, spk)
        y = self.repeats(y)
        # n x 2N x T
        e = torch.chunk(self.mask(y), 1, 1)
        # n x N x T
        if self.non_linear_type == "softmax":
            m = self.non_linear(torch.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(torch.stack(e, dim=0))
        # spks x [n x N x T]
        s = w * m
        out = self.decoder_1d(s, squeeze=True)
        #out = self.decoder_1d(y, squeeze=True)
        #if out.dim() == 1: # in case of batch size = 1
        #    out = torch.unsqueeze(out, 0)
        # downsample
        #out = self.downsample(out)

        # chopping
        _, _, output_length = out.shape
        #assert output_length >= input_length
        #start = (output_length - input_length)//2
        #out = out[:, start:start+input_length]
        out = out[:, :, :input_length]
        # spks x n x S
        return out, z, None # dummy

    def valid_length(self, length:int) -> int:
        length = self.encoder_1d.valid_length(length)
        length = self.proj.valid_length(length)
        for rpt in self.first_block:
            for blk in rpt:
                if type(blk) == Conv1DBlock:
                    length = blk.valid_length(length)
        for rpt in self.repeats:
            for blk in rpt:
                if type(blk) == Conv1DBlock:
                    length = blk.valid_length(length)
        length = self.decoder_1d.valid_length(length)
        return length
    
    '''
    def get_padding_value(self):
        start = -1
        end = -1
        s = torch.rand(4, 1000)
        with torch.no_grad():
            for n in range(1000, 1100):
                x = torch.rand(4, n)
                o, _, _ = self.forward(x, s)
                if x.shape[-1] == o.shape[-1]:
                    if start < 0:
                        start = n
                    else:
                        end = n
                        break
        self.padding_value = end - start

    def get_padded_value(self, x):
        v = self.padding_value - x.shape[-1] % self.padding_value
        x = torch.nn.functional.pad(x, pad=(1, v), value=0.)
        return x
    '''
    
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    model = ConvTasNet(config)

    lengths = [1600, 65536, 161231, 323453]
    for length in lengths:
        x = torch.rand(4, length)
        s = torch.rand(4, length)
        y,_ = model(x, s)
        print(x.shape)
        print(y.shape)
