import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from einops import rearrange
import numpy as np
import loss.complex_loss as C

class BiSpectrumLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, win_length=512, compute_log=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.L1Loss(reduction='sum')
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length       
        self.window = torch.hann_window(win_length, device=device)

        n_feat=self.n_fft//2 + 1
        self.f2 = np.arange(self.n_feat//2).tolist() + np.arange(self.n_feat//2, 0, -1).tolist()
        self.f1 = np.arange(self.n_feat)
        self.f3 = (np.array(self.f1) + np.array(self.f2)).tolist()

        self.compute_log = compute_log
        
        self.loss = C.PolarLoss(compute_log=self.compute_log)
        
    def complex_spectrum(self, x):
        return  torch.stft(x, self.n_fft,
                           self.hop_length,
                           self.win_length,
                           self.window, 
                           return_complex=True)

    def bi_spectrum(self, x):
        # (b t f)
        spec = self.complex_spectrum(x)
        outs = torch.stack([ spec[self.f1[..., n]] * spec[self.f2[...,n]] * torch.conj_physical(spec[self.f3[...,n]])
                             for n in range(len(self.f1)) ])
        assert outs.dim() == 3
        return rearrange(outs, 'f b t -> b t f')
    
    def forward(self, preds, targets, lengths):
        # (b, t) -> (b f t)
        preds = bi_spectrum(preds)
        targets = bi_spectrum(targets)
        
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            length = 1 + lengths[b]//self.hop_length # center=True
            mask[b, :, :length] = 1.
            
        return self.loss(preds, targets, mask)

class MultiResolutionBiSpectrumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        fft_sizes=[512, 256, 128]
        hop_sizes=[50, 30, 10]
        win_lengths=[240, 120, 60]
        
        self._losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self._losses += [BiSpectrumLoss(fs, ss, wl)]
                 
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

    loss=MultiResolutionBiSpectrumLoss()

    source,_=torchaudio.load(args.source)
    std, mean = torch.std_mean(source)
    source = (source-mean)/std
    target,_=torchaudio.load(args.target)
    std, mean = torch.std_mean(target)
    target = (target-mean)/std
    lengths = [source.shape[-1]]
    _loss = loss(source.to(device), target.to(device), lengths)
    print(_loss)
    
