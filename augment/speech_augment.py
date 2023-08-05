import io
import os
import math
import random
import torch
import torch.nn as nn
from torch import Tensor
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import pandas
import numpy as np

def compute_rms(source:Tensor):
    return torch.sqrt(torch.mean(torch.square(source)))

def compute_adjusted_rms(rms:Tensor, snr:float):
    return rms/(10 ** snr / 20)

def white_noise(length:int):
    return torch.normal(mean=0., std=1., size=(1, length))
    
def narrow_band_noise(length:int, sample_rate:int=16000, central_freq:int=4000, Q=100.0):
    return F.bandpass_biquad(white_noise(length), sample_rate, central_freq, Q, const_skirt_gain=False)

def get_scale(source:Tensor, noise:Tensor, snr_db=20):
    snr = math.exp(snr_db/10)
    scale = snr * (noise.norm(p=2)/source.norm(p=2))
    return (scale * source + noise) / 2


class AugmentBase(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        
        self.sample_rate = config['dataset']['segment']['sample_rate']
        self.min_snr = 0
        self.max_snr = 0
        self.fix_snr = True

    def get_snr(self):
        if self.fix_snr:
            return self.max_snr
        else:
            return np.random.rand() * (self.max_snr - self.min_snr) + self.min_snr
        
    def forward(self, x:Tensor):
        pass

class WhiteNoiseAugment(AugmentBase):
    def __init__(self, config:dict):
        super().__init__(config)
        
        self.min_snr = config['augment']['white_noise']['min_snr']
        self.max_snr = config['augment']['white_noise']['max_snr']
        self.fix_snr = config['augment']['white_noise']['fix_snr']

    def forward(self, x:Tensor):
        with torch.no_grad():
            snr = self.get_snr()
            noise = white_noise(x.shape[-1])
            factor = torch.div(compute_adjusted_rms(compute_rms(torch.abs(x)), snr), compute_rms(noise))

        return factor * noise

class NarrowBandNoiseAugment(AugmentBase):
    def __init__(self, config:dict):
        super().__init__(config)

        self.min_snr = config['augment']['narrow_band']['min_snr']
        self.max_snr = config['augment']['narrow_band']['max_snr']
        self.fix_snr = config['augment']['narrow_band']['fix_snr']
        
        self.min_central_freq = config['augment']['narrow_band']['min_central_freq']
        self.max_central_freq = config['augment']['narrow_band']['max_central_freq']
        self.min_Q = config['augment']['narrow_band']['min_Q']
        self.max_Q = config['augment']['narrow_band']['max_Q']
        
        self.fix_params = config['augment']['narrow_band']['fix_params']
        
    def forward(self, x:Tensor):
        snr = self.get_snr()

        with torch.no_grad():
            noise = white_noise(x.shape[-1])

            if self.fix_params:
                central_freq = self.max_central_freq
                Q = self.max_Q
            else:
                central_freq = np.random.rand() * (self.max_central_freq - self.min_central_freq) + self.min_central_freq
                Q = np.random.rand() * (self.max_Q - self.min_Q) + self.min_Q
            
                noise = F.bandpass_biquad(noise, self.sample_rate, central_freq, Q, const_skirt_gain=False)
                factor = torch.div(compute_adjusted_rms(compute_rms(torch.abs(x)), snr), compute_rms(noise))

        return factor * noise

class RandomAmplitudeModulationAugment(AugmentBase):
    def __init__(self, config:dict):
        super().__init__(config)
        
        self.min_offset = config['augment']['random_modulation']['min_offset']
        self.max_offset = config['augment']['random_modulation']['max_offset']
        
        self.min_factor = config['augment']['random_modulation']['min_factor']
        self.max_factor = config['augment']['random_modulation']['max_factor']
        
        self.min_speed = config['augment']['random_modulation']['min_speed']
        self.max_speed = config['augment']['random_modulation']['max_speed']

        self.fix_params = config['augment']['random_modulation']['fix_params']
        
    def forward(self, x:Tensor):
        with torch.no_grad():
            noise = white_noise(x.shape[-1])

            if self.fix_params:
                offset = self.max_offset
                factor = self.max_factor
                speed = self.max_speed
            else:
                offset = np.random.rand()*(self.max_offset - self.min_offset) + self.min_offset
                factor = np.random.rand()*(self.max_factor - self.min_factor) + self.min_factor
                speed = np.random.rand()*(self.max_speed - self.min_speed) + self.min_speed

            mod = F.lowpass_biquad(factor*noise, self.sample_rate, cutoff_freq=0.5*speed)
            mod += offset

        return x * mod

class LowPassRadio(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sample_rate = config['dataset']['segment']['sample_rate']
        self.cutoff = 7500
        
    def forward(self, x:Tensor):
        with torch.no_grad():
            return F.lowpass_biquad(x, self.sample_rate, cutoff_freq=self.cutoff)
