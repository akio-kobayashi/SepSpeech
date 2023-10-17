import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from loss.stft_loss import stft
from einops import rearrange
from loss.plcpa import complex_stft
import numpy as np

def unwrap(phi, dim=-1):
    assert dim is -1
    dphi = diff(phi, same_size=False)
    dphi_m = ((dphi+np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m==-np.pi)&(dphi>0)] = np.pi
    phi_adj = dphi_m-dphi
    phi_adj[dphi.abs()<np.pi] = 0
    return phi + phi_adj.cumsum(dim)

def diff(x, dim=-1, same_size=False):
    assert dim is -1
    if same_size:
        return F.pad(x[...,1:]-x[...,:-1], (1,0))
    else:
        return x[...,1:]-x[...,:-1]
                
def complex_cepstrum(x, n=None):
    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = torch.round(unwrapped[..., center] / np.pi)
        unwrapped -= np.pi * ndelay[..., None] * torch.arange(samples) / center
        return unwrapped, ndelay

    spectrum = torch.fft.fft(x, n=n)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
    ceps = torch.fft.ifft(log_spectrum).real

    return ceps, ndelay

class CepstrumLoss(nn.Module):
    def __init__(self):
        super().__init__()

    