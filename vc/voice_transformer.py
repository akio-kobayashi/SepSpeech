import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from typing import Optional, Any, Union, Callable
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import torch.nn.modules.transformer as tfm

def compute_diagonal_masks(T, S, nu=0.3):
    with torch.no_grad():
        weights=torch.zeros(1, T, S)
        tgt = torch.arange(T).reshape(-1, 1).repeat(1, S).to(float)
        tgt = tgt/T
        src = torch.arange(S).reshape(1, -1).repeat(T, 1).to(float)
        src = S
        weight = torch.exp(-1 * torch.square(src - tgt)/(2.*nu*nu))
        weights[0, :T, :S] = weight
    weights = torch.where(weights==0.0, float('-inf'), torch.log(x))
    return weights

def compute_diagonal_weights(B, T, S, src_lengths, tgt_lengths, nu=0.3, batch_first=True):
    with torch.no_grad():
        weights=torch.ones(B, T, S)
        for b in range(B):
            tgt = torch.arange(tgt_lengths[b]).reshape(-1, 1).repeat(1, src_lengths[b]).to(float)
            tgt = tgt/tgt_lengths[b]
            src = torch.arange(src_lengths[b]).reshape(1, -1).repeat(tgt_lengths[b], 1).to(float)
            src = src/src_lengths[b]
            weight = torch.exp(-1 * torch.square(src - tgt)/(2.*nu*nu))
            weights[b, :tgt_lengths[b], :src_lengths[b]] -= weight
    if batch_first is False:
        weights = weights.transpose(1, 0, 2)

    return weights

'''
class VoiceTransformerEncoderAny(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layers, num_layers, norm=None):
        super(VoiceTransformerEncoderAny, self).__init__()
        self.layers=ModuleList()
        for mod in encoder_layers:
            self.layers.append(mod)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class VoiceTransformerEncoderLayerAny(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                layer_norm_eps=1e-5, batch_first=False, norm_first=False):
        super(VoiceTransformerEncoderLayerAny, self).__init__()
        # _sa_block
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # _ff_block
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x
'''

class VoiceTransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layers, num_layers, norm=None):
        super(VoiceTransformerEncoder, self).__init__()
        self.layers=ModuleList()
        for mod in encoder_layers:
            self.layers.append(mod)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, spk, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, spk, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class VoiceTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_embed, nhead, dim_feedforward=1024, dropout=0.1,
                layer_norm_eps=1e-5, batch_first=False, norm_first=False):
        super(VoiceTransformerEncoderLayer, self).__init__()
        # _sa_block
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # _ff_block
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.projection1 = Linear(d_model+d_embed, d_model)
        self.projection2 = Linear(d_model+d_embed, d_model)
        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, spk, src_mask=None, src_key_padding_mask=None):
        x = self.projection1(torch.cat((src, spk), dim=-1))
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = self.projection2(torch.cat((x, spk), dim=-1))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.projection2(torch.cat((x, spk), dim=-1))
            x = self.norm2(x + self._ff_block(x))
        return x

class VoiceTransformerDecoder(nn.Module):
    __constants__=['norm']

    def __init__(self, decoder_layers, num_layers, norm=None):
        super(VoiceTransformerDecoder, self).__init__()
        self.layers = ModuleList()
        for mod in decoder_layers:
            self.layers.append(mod)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, spk, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, spk, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def _alignment_loss(self, dtwpath, src_len, tgt_len):
        loss=0.
        for mod in self.layers:
            loss+=mod._alignment_loss(dtwpath, src_len, tgt_len)
        return loss

    def _diagonal_attention_loss(self, weights, src_len, tgt_len):
        loss=0.
        for mod in self.layers:
            loss+=mod._diagonal_attention_loss(weights, src_len, tgt_len)
        return loss

    def _get_attention_weight(self):
        weights=[]
        for mod in self.layers:
            weights.append(mod._get_attention_weight())

        return weights

class VoiceTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_embed, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, device=None, dtype=None, nu=0.3):
        super(VoiceTransformerDecoderLayer, self).__init__()
        self.attention_weight=0.
        self.nu=nu
        self.norm_first=norm_first
        self.batch_first=batch_first

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = F.relu
        self.projection1 = nn.Linear(d_model+d_embed, d_model)
        self.projection2 = nn.Linear(d_model+d_embed, d_model)
        self.projection3 = nn.Linear(d_model+d_embed, d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask=None, key_padding_mask=None):
        x, self.attention_weight = self.multihead_attn(x, mem, mem,
                                                       attn_mask=attn_mask,
                                                       key_padding_mask=key_padding_mask,
                                                       need_weights=True)
        return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(self, tgt, memory, spk, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        x = self.projection1(torch.cat((tgt, spk), dim=-1))
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = self.projection2(torch.cat((x, spk), dim=-1))
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = self.projection3(torch.cat((x, spk), dim=-1))
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.projection2(torch.cat((x, spk), dim=-1))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.projection3(torch.cat((x, spk), dim=-1))
            x = self.norm3(x + self._ff_block(x))

        return x

    def _alignment_loss(self, dtwpath, src_len, tgt_len):
        B=self.attention_weight.shape[0]

        loss = 0.
        for b in range(B):
            l1 = F.l1_loss(self.attention_weight[b, :tgt_len[b], :src_len[b]].cuda(),
                             dtwpath[b, :tgt_len[b], :src_len[b]].cuda())
            loss += l1
        return torch.sum(l1)

    def _diagonal_attention_loss(self, weights, src_len, tgt_len):
        #loss = 1/(weights.numel()) * torch.sum(torch.norm(weights.cuda() * self.attention_weight, dim=(1,2), p=1))
        #loss = torch.mean(torch.norm(weights.cuda() * self.attention_weight, dim=(1,2), p=1))
        #loss = 1/(weights.numel()) * torch.sum(torch.norm(weights.cuda() * self.attention_weight, dim=(1,2), p=2))
        loss=0.
        for b in range(self.attention_weight.shape[0]):
            loss += torch.norm(weights.cuda()[b, :tgt_len[b], :src_len[b]] * self.attention_weight[b, :tgt_len[b], :src_len[b]], p=1)
        loss = torch.mean(loss)
        return loss

    def _get_attention_weight(self):
        return self.attention_weight

class VoiceTransformer(nn.Module):
    #factory_kwargs = {'device': device, 'dtype': dtype}

    def __init__(self, d_model=512, d_embed=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None):
        super(VoiceTransformer, self).__init__()

        encoder_layers = [ VoiceTransformerEncoderLayer(d_model, d_embed, nhead, dim_feedforward, dropout,
                                                        layer_norm_eps, batch_first, norm_first) for i in range(num_encoder_layers) ]
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = VoiceTransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)

        decoder_layers = [ VoiceTransformerDecoderLayer(d_model, d_embed, nhead, dim_feedforward, dropout,
                                                        layer_norm_eps, batch_first, norm_first) for i in range(num_decoder_layers) ]
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = VoiceTransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.d_embed = d_embed
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src, tgt, src_spk, tgt_spk, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, src_spk, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_spk, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def _diagonal_attention_loss(self, weights, src_len, tgt_len):
        loss = self.decoder._diagonal_attention_loss(weights, src_len, tgt_len)
        return loss

    def _alignment_loss(self, dtwpath, src_len, tgt_len):
        loss = self.decoder._alignment_loss(dtwpath, src_len, tgt_len)
        return loss

    def _get_attention_weight(self):
        return self.decoder._get_attention_weight()

'''    
class VoiceTransformerAny(nn.Module):
    def __init__(self, d_model=512, d_embed=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=True,
                 device=None, dtype=None):
        super(VoiceTransformerAny, self).__init__()

        encoder_layers = [ VoiceTransformerEncoderLayerAny(d_model, nhead, dim_feedforward, dropout,
                                                           layer_norm_eps, batch_first, norm_first) for i in range(num_encoder_layers) ]
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = VoiceTransformerEncoderAny(encoder_layers, num_encoder_layers, encoder_norm)

        decoder_layers = [ VoiceTransformerDecoderLayer(d_model, d_embed, nhead, dim_feedforward, dropout,
                                                        layer_norm_eps, batch_first, norm_first) for i in range(num_decoder_layers) ]
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = VoiceTransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.d_embed = d_embed
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src, tgt, tgt_spk, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_spk, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def _diagonal_attention_loss(self, weights, src_len, tgt_len):
        loss = self.decoder._diagonal_attention_loss(weights, src_len, tgt_len)
        return loss

    def _alignment_loss(self, dtwpath, src_len, tgt_len):
        loss = self.decoder._alignment_loss(dtwpath, src_len, tgt_len)
        return loss

    def _get_attention_weight(self):
        return self.decoder._get_attention_weight()
'''
