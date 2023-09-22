import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from loss.stft_loss import stft
from einops import rearrange

def complex_stft(x, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length, device=x.device)
    x_stft = torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
    return x_stft

class PLCPA_ASYM(nn.Module):
    def __init__(self, config):
        super().__init__()

        '''
        self.wav2compspec = torchaudio.transforms.Spectrogram(
            n_fft=config['n_fft'],
            win_length=config['win_length'],
            hop_length=config['hop_length'],
            window_fn=torch.hamming_window,
            power=None # complex
        )
        '''
        
        self.n_fft=config['n_fft']
        self.win_length=config['win_length']
        self.hop_length=config['hop_length']
        #self.window_fn=torch.hann_window
        
        self.p = config['p']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        
    def forward(self, preds, targets, lengths):
        total_frames = 0
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :lengths[b]] = 1.
            total_frames += int (lengths[b]/self.hop_length) + 1
        preds *= mask
        targets *= mask
        
        #preds = self.wav2compspec(preds) # (C, F, T)
        preds = complex_stft(preds, self.n_fft, self.hop_length, self.win_length)
        preds_angle = torch.angle(preds)
        preds_abs = torch.pow(torch.abs(preds), self.p)
        preds = torch.polar(preds_angle, preds_abs)

        #targets = self.wav2compspec(targets)
        preds = complex_stft(targets, self.n_fft, self.hop_length, self.win_length)
        targets_angle = torch.angle(preds)
        targets_abs = torch.pow(torch.abs(preds), self.p)
        targets = torch.polar(targets_angle, targets_abs)

        '''
        _, N, _ = preds.shape

        L_a = torch.sum(torch.square(preds_abs - targets_abs)) / (total_frames * N)
        L_p = torch.sum(torch.square(torch.abs(preds - targets))) / (total_frames * N)
        L_os = torch.sum(torch.square(F.relu( preds_abs - targets_abs))) / (total_frames * N)
        '''
        L_a = torch.mean(torch.square(preds_abs - targets_abs)) 
        L_p = torch.mean(torch.square(torch.abs(preds - targets))) 
        L_os = torch.mean(torch.square(F.relu( preds_abs - targets_abs)))
        
        return self.alpha * L_a + (1. - self.alpha) * L_p  + self.beta * L_os
        
class MultiResPLCPA_ASYM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        fft_sizes = [1024, 2048, 512]
        hop_sizes = [120, 240, 50]
        win_lengths = [600, 1200, 240]

        self.losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            config['n_fft'] = fs
            config['win_length'] = wl
            config['hop_length'] = ss
            
            self.losses += [PLCPA_ASYM(config)]

    def forward(self, preds, targets, lengths):
        _loss = 0.
        for func in self.losses:
            _loss += func(preds.clone(), targets, lengths)
        return _loss
    
