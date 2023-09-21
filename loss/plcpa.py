import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from loss.stft_loss import stft
from loss.tsos_loss import TSOSLoss
from einops import rearrange

class PLCPA(nn.Module):
    def __init__(self, n_fft=512, win_length=400, hop_length=160, p=0.3, alpha=0.5):
        super().__init__()

        self.wav2compspec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hamming_window,
            power=None # complex
        )

        self.hop_length=hop_length
        self.p = p
        self.alpha = alpha
    
    def forward(self, preds, targets, lengths):
        total_frames = 0
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :, :lengths[b]] = 1.
            total_frames += int (lengths[b]/self.hop_length) + 1
        preds *= mask
        targets *= mask
        
        #spec_length=[ int(lengths[b]/self.hop_length) for b in range(len(preds)) ]
 
        preds = self.wav2compspec(preds) # (C, F, T)
        preds_angle = torch.angle(preds)
        preds_abs = torch.power(torch.abs(preds), self.p)
        preds = torch.polar(preds_angle, preds_abs)

        targets = self.wav2compspec(targets)
        targets_angle = torch.angle(preds)
        targets_abs = torch.power(torch.abs(preds), self.p)
        targets = torch.polar(targets_angle, targets_abs)

        L_a = torch.sum(torch.squre(preds_abs - targets_abs)) / total_frames
        L_p = torch.sum(torch.square(preds - targets)) / total_frames

        return self.alpha * L_a + (1. - self.alpha) * L_p 
    
class PLCPA_ASYM(nn.Module):
    def __init__(self, n_fft=512, win_length=400, hop_length=160, p=0.3, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.plcpa_loss = PLCPA(n_fft, win_length, hop_length, p, alpha)
        self.tsos_loss = TSOSLoss(n_fft, win_length, hop_length, p, gamma)

        self.beta = beta

    def forward(self, preds, targets, lengths):
        return self.plcpa_loss(preds, targets, lengths) + self.beta * self.tsos_loss(preds, targets, lengths)

