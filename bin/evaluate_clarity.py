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
    def __init__(self, loss_type='none'):
        self.sample_rate = 16000
        self.level1 = 65 # level in dB SPL corresponding to RMS=1
        self.hearing_loss = get_hearing_loss(loss_type)
        self.freqs = np.array([250, 500, 1000, 2000, 4000, 6000])
        self.audiogram = Audiogram(levels=self.hearing_loss, frequencies=self.freqs)

    def eval(self, x, y):
        score, _ = haspi_v2(x, self.sample_rate,
                            y, self.sample_rate,
                            self.audiogram, self.level1)
        #print(score)
        return score
    
class HASQI:
    def __init__(self, loss_type):
        self.sample_rate = 16000
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
        #print(score)
        return score

max_lag = 250

def align(source:Tensor, target:Tensor):
    correlation = signal.correlate(rearrange(target, 'c t -> (c t)').detach().numpy(),
    rearrange(source,'c t -> (c t)').detach().numpy(), mode='full')
    lags = signal.correlation_lags(target.shape[-1], source.shape[-1], mode='full')
    lag = lags[np.argmax(correlation)]

    if lag <= max_lag :
        if lag > 0:
            target = target[:, lag:]
            source = source[:, :target.shape[-1]]
        else:
            lag = -lag
            source = source[:, lag:]
            target = target[:, :source.shape[-1]]

    return source, target

def read_audio(path, normalize=True):
    wave, sr = torchaudio.load(path)
    if normalize:
        rms = torch.sqrt(torch.sum(torch.square(wave))/wave.shape[-1])
        wave /= rms
    #std, mean = torch.std_mean(wave)
    #wave = wave - mean
    #print(torch.max(wave))
    return wave

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

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _haspi = HASPI(args.hearing_loss)
    _hasqi = HASQI(args.hearing_loss)
    _hearing_aid = HearingAid(args.hearing_loss)
    
    df_out = pd.DataFrame(index=None, 
                          columns=['key', 'mixture', 'source', 'estimate', 'mix_haspi', 'mix_hasqi', 'est_haspi', 'est_hasqi'])
    keys = []
    est_haspi, est_hasqi = [], []
    mix_haspi, mix_hasqi = [], []
    mixtures, sources, estimates = [], [], []
    df = pd.read_csv(args.input_csv)

    with torch.no_grad():
        for index, row in df.iterrows():

            key = os.path.splitext(os.path.basename(row['source']))[0]
            keys.append(key)
            mixtures.append(row['mixture'])
            sources.append(row['source'])
            estimates.append(row['estimate'])

            mixture = read_audio(row['mixture'])
            source = read_audio(row['source'])
            estimate = read_audio(row['estimate'])

            #mixture, source = align(mixture, source)
            mixture = rearrange(mixture, 'c t -> (c t)').cpu().detach().numpy()
            if not args.no_enhance:
                mixture = _hearing_aid.process(mixture)
            source = rearrange(source, 'c t -> (c t)').cpu().detach().numpy()

            if args.mix :
                mix_haspi.append(_haspi.eval(source, mixture))
                mix_hasqi.append(_hasqi.eval(source, mixture))
            else:
                mix_haspi.append(0.)
                mix_hasqi.append(0.)
                
            #source = read_audio(row['source'])
            #estimate, soruce = align(estimate, source)
            #source = rearrange(source, 'c t -> (c t)')
            estimate = rearrange(estimate, 'c t -> (c t)').cpu().detach().numpy()
            if not args.no_enhance:
                estimate = _hearing_aid.process(estimate)
            
            est_haspi.append(_haspi.eval(source, estimate))
            est_hasqi.append(_hasqi.eval(source, estimate))
            
    df_out['key'], df_out['mixture'], df_out['source'], df_out['estimate'] = keys, mixtures, sources, estimates
    df_out['mix_haspi'], df_out['mix_hasqi'] = mix_haspi, mix_hasqi
    df_out['est_haspi'], df_out['est_hasqi'] = est_haspi, est_hasqi

    haspi, hasqi = np.mean(mix_haspi), np.mean(mix_hasqi)
    print("mixture:  HASPI = %.4f , HASQI = %.4f" % (haspi, hasqi))
    
    haspi, hasqi = np.mean(est_haspi), np.mean(est_hasqi)
    print("estimate: HASPI = %.4f , HASQI = %.4f" % (haspi, hasqi))

    df_out.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--hearing_loss', type=str, default='none')
    parser.add_argument('--sample_rate', default=16000, type=int)
    parser.add_argument('--mix', action='store_true')
    parser.add_argument('--no_enhance', action='store_true')
    args = parser.parse_args()

    main(args)

