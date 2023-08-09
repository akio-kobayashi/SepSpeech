import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from augment.speech_augment import AugmentBase

class FadingAugment(AugmentBase):
    def __init__(self, config:dict):
        super().__init__(config)

        self.light_speed=3.e8
        self.scaler = config['augment']['fading']['scaler']
        self.min_velocity = float(config['augment']['fading']['min_velocity'])
        self.max_velocity = float(config['augment']['fading']['max_velocity'])
        self.center_freq = float(config['augment']['fading']['center_freq'])
        self.N = config['augment']['fading']['num_sinusoids']
        self.Fs = self.sample_rate * self.scaler
        
    #def rand_pi(self):
    #    return 2*np.pi*(torch.rand([1,1])-0.5)

    def channel_gain(self, fd, T):
        time = torch.arange(start=0, end=T)
        #.repeat(self.N, 1)
        gain_r = torch.zeros(1, T)
        #gain_i = torch.zeros(1, T)

        alpha = 2 * np.pi * torch.rand(self.N, 1)
        phi   = 2 * np.pi * torch.rand(self.N, 1)
        factor = torch.randn(self.N, 1)
        for n in range(self.N):
            gain_r += factor[n, :] * torch.cos(2*np.pi*fd*time/self.Fs * torch.cos(alpha[n,:])+phi[n,:])
        #gain_i = gain_i + torch.randn(1, 1) * torch.sin(2*np.pi*fd*time/Fs * torch.cos(alpha)+phi)
        return 1./np.sqrt(self.N) * gain_r

    def forward(self, signal):
        velocity = self.min_velocity + (self.max_velocity - self.min_velocity) * np.random.rand()
        fd = velocity * self.center_freq
        fd /= self.light_speed
        signal = torch.nn.functional.interpolate(signal.unsqueeze(0), scale_factor=self.scaler).squeeze().unsqueeze(0)
        T = len(signal.t())

        carrier = torch.arange(start=0, end=T).cuda()
        carrier = torch.sign(2*np.pi*self.center_freq *carrier/self.Fs)
        mod_sig = signal * carrier
        gain = self.channel_gain(fd, T).cuda()
        gain = gain/torch.max(torch.abs(gain))

        demod_sig = (mod_sig * gain) * carrier
        demod_sig = torchaudio.functional.lowpass_biquad(demod_sig, self.Fs, cutoff_freq=self.sample_rate/2)
        demod_sig = torch.nn.functional.interpolate(demod_sig.unsqueeze(0),scale_factor=1./self.scaler).squeeze().unsqueeze(0)

        return demod_sig

