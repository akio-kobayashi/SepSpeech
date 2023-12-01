import argparse
import json
import logging
import sys, os
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
import clarity
from clarity.evaluator.haspi import haspi_v2
from clarity.evaluator.hasqi import hasqi_v2
from clarity.utils.audiogram import Audiogram, Listener
from clarity.enhancer.nalr import NALR
from clarity.enhancer.compressor import Compressor
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np
import scipy.signal as signal
from einops import rearrange

def get_hearing_loss(loss_type):
    # center freqs:     250, 500, 1000, 2000, 4000, 6000
    if loss_type == 'none':
        return np.array([0, 0, 0, 0, 0, 0])
    elif loss_type == 'mild':
        return np.array([10, 15, 19, 25, 31, 35])
    elif loss_type == 'moderate':
        return np.array([20, 20, 25, 35, 45, 50])
    elif loss_type == 'severe':
        return np.array([19, 28, 40, 52, 58, 58])
    else:
        raise ValueError
    
class HASPI:
    def __init__(self, loss_type='none', sample_rate=16000):
        self.sample_rate = sample_rate
        self.level1 = 65 # level in dB SPL corresponding to RMS=1
        self.hearing_loss = get_hearing_loss(loss_type)
        self.freqs = np.array([250, 500, 1000, 2000, 4000, 6000])
        self.audiogram = Audiogram(levels=self.hearing_loss, frequencies=self.freqs)

    def eval(self, x, y):
        score, _ = haspi_v2(x, self.sample_rate,
                            y, self.sample_rate,
                            self.audiogram, self.level1)
        return score
    
class HASQI:
    def __init__(self, loss_type, equalization_mode=1, sample_rate=16000):
        self.sample_rate = sample_rate
        self.level1 = 65 # level in dB SPL corresponding to RMS=1
        self.equalization_mode = 1
        self.hearing_loss = get_hearing_loss(loss_type)
        self.freqs = np.array([250, 500, 1000, 2000, 4000, 6000])
        self.audiogram = Audiogram(levels=self.hearing_loss, frequencies=self.freqs)

    def eval(self, x, y):
        score, _, _, _ = hasqi_v2(x, self.sample_rate,
                                  y, self.sample_rate,
                                  self.audiogram,
                                  self.equalization_mode,
                                  self.level1)
        return score

def read_audio(path):
    wave, sr = torchaudio.load(path)
    return rearrange(wave, 'c t -> (c t)').to('cpu').numpy()

class HearingAid:
    def __init__(self, loss_type='none'):
        self.enhancer = NALR(nfir=220, sample_rate=16000)
        self.compressor = Compressor(threshold=0.35,
                                     attenuation=0.1,
                                     attack=50,
                                     release=1000,
                                     rms_buffer_size=0.064)
        hearing_loss = get_hearing_loss(loss_type)
        freqs = np.array([250, 500, 1000, 2000, 4000, 6000])
        audiogram = Audiogram(levels=hearing_loss, frequencies=freqs)
        self.nalr_fir, _ = self.enhancer.build(audiogram)
        
    def process(self, signal):
        out = self.enhancer.apply(self.nalr_fir, signal)
        out, _, _ = self.compressor.process(out)
        
        return out
    
def main(args):

    _haspi = HASPI(args.hearing_loss)
    eq_mode = 2 if args.hearing_loss == 'none' else 1
    _hasqi = HASQI(args.hearing_loss, equalization_mode=eq_mode)
    _hearing_aid = HearingAid(args.hearing_loss)
    
    reference = read_audio(args.reference)
    target = read_audio(args.target)
        
    if args.hearing_loss!='none':
        target = _hearing_aid.process(target)
        
    haspi = _haspi.eval(reference, target)
    hasqi = _hasqi.eval(reference, target)
        
    print("HASPI = %.4f , HASQI = %.4f" % (haspi, hasqi))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--reference', type=str, help="reference wave file")
    parser.add_argument('--target', type=str, help="target/processed wave file")
    parser.add_argument('--hearing_loss', type=str, default='none', help="[none|mild|moderate|severe]")
    parser.add_argument('--sample_rate', default=16000, type=int)
    args = parser.parse_args()

    main(args)

