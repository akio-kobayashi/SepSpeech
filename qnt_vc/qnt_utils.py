import torch
import torch.nn as nn
from einops import rearrange

bos_token_id = 1024
eos_token_id = 1025

def make_batch(src, tgt, src_id, tgt_id, ar=True, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _src, _tgt = [], []
    _src_lengths, _tgt_lengths = [], []
    _src_id, _tgt_id = [], []

    print(type(tgt))
    for _s, _t in zip(src, tgt):
        _, T, _ = _s.shape
        _src_lengths.append(T)
        _src.append(rearrange(_s, 'c t f -> t c f'))    
        _src_id.append(src_id)

        print(type(_t))
        if ar is True:
            _t =  append_special_tokens(_t, bos=True)
        _, T, _ = _t.shape
        _tgt_lengths.append(T)
        _tgt.append(rearrange(_t, 'c t f -> t c f'))
        _tgt_id.append(tgt_id)

    _src_id = torch.tensor(_src_id, device=device)
    _tgt_id = torch.tensor(_tgt_id, device=device)
    _src = nn.utils.rnn.pad_sequence(_src, batch_first=True, padding_value=-1).to(device)
    _src = _src.rearrange(_src, 'b t c f -> b c t f')
    _tgt = nn.utils.rnn.pad_sequence(_tgt, batch_first=True, padding_value=-1).to(device)
    _tgt = _tgt.rearrange(_tgt, 'b t c f -> b c t f')

    return _src, _tgt, _src_id, _tgt_id, _src_lengths, _tgt_lengths 

def append_special_tokens(x, bos=True, device=None):
    global bos_token_id
    global eos_token_id

    B, T, F = x.shape # (8, T, N)
    assert B==1

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bos is True:
        bos = rearrange(torch.tensor([bos_token_id], device=device, dtyep=x.dtype), '(b c f) -> b c f', b=B,c=1,f=1)
        expanded = torch.cat([bos, x], dim=-2) # (8, T+1, N)
    else:
        eos = rearrange(torch.tensor([eos_token_id], device=device, dtyep=x.dtype), '(b c f) -> b c f', b=B,c=1,f=1)
        expanded = torch.cat([x, eos], dim=-1)

    return expanded