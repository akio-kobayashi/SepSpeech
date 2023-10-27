import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from einops import rearrange
import numpy as np
from loss.ssim_loss SSIMLoss

class SplitL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.L1Loss(reduction='sum')
        
    def forward(self, preds, targets, mask):
        # (b t f)
        assert preds.dim() == 3 and targets.dim() == 3
        
        _loss = self.loss(preds.real * mask, targets.real * mask) / torch.sum(mask) + self.loss(preds.imag * mask , targets.image * mask )/torch.sum(mask)
        
        return _loss

class SplitMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss(reduction='sum')
        
    def forward(self, preds, targets, mask):
        # (b t f)
        assert preds.dim() == 3 and targets.dim() == 3
        
        _loss = self.loss(preds.real * mask, targets.real * mask) / torch.sum(mask) + self.loss(preds.imag * mask , targets.image * mask )/torch.sum(mask)
        
        return _loss

class SplitSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = SSIMLoss(reduction='sum')
        
    def forward(self, preds, targets, mask):
        # (b t f)
        assert preds.dim() == 3 and targets.dim() == 3
        
        _loss = self.loss(preds.real * mask, targets.real * mask) / torch.sum(mask) + self.loss(preds.imag * mask , targets.image * mask )/torch.sum(mask)
        
        return _loss

class PolarLoss(nn.Module):
    def __init__(self, loss_func, weight_mag=1.0, weight_phase=1.0, compute_log=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = loss_func
        self.weight_mag = weight_mag
        self.weight_phase = weight_phase
        self.eps = 1.e-8
        
    def forward(self, preds, targets, mask):
        # (b f t)
        assert preds.dim() == 3 and targets.dim() == 3

        if self.compute_log:
            mag_loss = self.loss(torch.log(torch.abs(preds) + self.eps),  torch.log(torch.abs(targets) + self.eps))
        else:
            mag_loss = self.loss(preds, targets)
        mag_loss = torch_sum(mag_loss*mask)/torch.sum(mask)

        phase_loss = self.loss(torch.angle(preds), torch.angle(targets))
        phase_loss = torch.sum(phase_loss)/torch.sum(mask)

        _loss = self.weight_mag * mag_loss + self.weight_phase * phase_loss
        
        return _loss

class ComplexQuadLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, preds, targets, mask):
        # (b f t)
        assert preds.dim() == 3 and targets.dim() == 3
        
        _loss = 0.5 * torch.sum(torch.square(torch.abs(preds * mask - targets * mask)))/torch.sum(mask) # float
        
        return _loss

class ComplexFourthPowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, preds, targets, mask):
        # (b f t)
        assert preds.dim() == 3 and targets.dim() == 3
        
        sq = torch.square(torch.abs(preds*mask - targets*mask))
        _loss = 0.5 * torch.sum(torch.square(sq))/torch.sum(mask) # float
        
        return _loss

class ComplexCauchyLoss(nn.Module):
    def __init__(self, c=1.0):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c = c
        
    def forward(self, preds, targets, mask):
        # (b f t)
        assert preds.dim() == 3 and targets.dim() == 3
        
        sq = torch.square(torch.abs(preds*mask - targets*mask))
        val = self.c / (2.*torch.log(1. + sq/(self.c*self.c)))
        _loss = 0.5 * torch.sum(val)/torch.sum(mask)
        
        return _loss

class ComplexLogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, preds, targets, mask):
        # (b f t)
        assert preds.dim() == 3 and targets.dim() == 3
        
        sq = torch.square(torch.abs(preds*mask - targets*mask))
        _loss = torch.sum(torch.log(torch.cosh(sq)))/torch.sum(mask)
        
        return _loss

class ComplexLogLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = 1.e-8
        
    def forward(self, preds, targets, mask):
        # (b f t)
        assert preds.dim() == 3 and targets.dim() == 3
        
        val = torch.square(torch.abs(torch.log(preds * mask + self.eps) - torch.log(targets * mask + self.eps)))
        _loss = torch.sum(val)/torch.sum(mask)
        
        return _loss
    
