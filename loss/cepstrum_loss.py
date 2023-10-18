import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
#from loss.stft_loss import stft
from einops import rearrange
#from loss.plcpa import complex_stft
import numpy as np

def unwrap(phi, dim=-1):
    #assert dim is -1
    dphi = diff(phi, same_size=True)
    dphi_m = ((dphi+np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m==-np.pi)&(dphi>0)] = np.pi
    phi_adj = dphi_m-dphi
    phi_adj[torch.abs(dphi)<np.pi] = 0
    return phi + torch.cumsum(phi_adj, dim)

def diff(x, dim=-1, same_size=False):
    #assert dim is -1
    if same_size:
        return F.pad(x[...,1:]-x[...,:-1], (1,0))
    else:
        return x[...,1:]-x[...,:-1]
                
def complex_cepstrum(x,
                     n_fft,
                     hop_length,
                     win_length,
                     window):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = torch.round(unwrapped[..., center] / np.pi)
        unwrapped -= np.pi * ndelay[..., None] * torch.arange(samples).to(device) / center
        return unwrapped, ndelay

    #spectrum = torch.fft.fft(x, n=n)
    spectrum = torch.stft(x, n_fft, hop_length, win_length, window=window, return_complex=True)

    unwrapped_phase, ndelay = _unwrap(torch.angle(spectrum))
    log_spectrum = torch.log(torch.abs(spectrum)) + 1j * unwrapped_phase

    cepstrum = torch.stft(x, n_fft, hop_length, win_length, window=window, return_complex=True)
    return cepstrum.real, ndelay


class CepstrumLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, win_length=512):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.L1Loss(reduction='sum')
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length       
        self.window = torch.hann_window(win_length, device=device)
        
    def forward(self, preds, targets, lengths):
        # (b, t) -> (b f t)
        preds = complex_cepstrum(preds, self.n_fft, self.hop_length, self.win_length, self.window)[0]
        targets = complex_cepstrum(targets, self.n_fft, self.hop_length, self.win_length, self.window)[0]
        
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :, :lengths[b]] = 1.
        return self.loss(preds * mask, targets * mask) / torch.sum(mask)

class MultiResolutionCepstrumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        fft_sizes=[1024, 2048, 512]
        hop_sizes=[120, 240, 50]
        win_lengths=[600, 1200, 240]
        
        self.cep_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.cep_losses += [CepstrumLoss(fs, ss, wl)]
                 
    def forward(self, preds, targets, lengths):
        _loss = 0.
        for f in self.cep_losses:
            _loss += f(preds, targets, lengths)
        return _loss
        
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser=ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args=parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss=MultiResolutionCepstrumLoss()

    source,_=torchaudio.load(args.source)
    std, mean = torch.std_mean(source)
    source = (source-mean)/std
    target,_=torchaudio.load(args.target)
    std, mean = torch.std_mean(target)
    target = (target-mean)/std
    lengths = [source.shape[-1]]
    _loss = loss(source.to(device), target.to(device), lengths)
    print(_loss)
    
