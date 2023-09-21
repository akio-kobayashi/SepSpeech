import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class TSOSLoss(nn.Module):
    def __init__(self, n_fft=512, win_length=400, hop_length=160, p=0.3, gamma=0.1):
        super().__init__()
        
        self.wav2spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hamming_window
        )
        
        self.hop_length = hop_length
        self.p = p
        self.gamma = gamma

    def forward(self, preds, targets, lengths):
        
        total_frames = 0
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :, :lengths[b]] = 1.
            total_frames += int (lengths[b]/self.hop_length) + 1
        preds *= mask
        targets *= mask
        
        preds = self.wav2spec(preds) # (C, F, T)
        targets = self.wav2spec(targets)

        p = 0.3
        gamma = 0.1
        L_os = torch.sum(torch.square(F.relu( torch.pow (preds, p) - torch.pow(targets, p) )), dim=1) # (C, T)
        L = gamma * torch.sum(torch.pow(preds, p), dim=1)
        tsos_loss = torch.sum(F.relu(L_os - L)) / total_frames

        return tsos_loss    
