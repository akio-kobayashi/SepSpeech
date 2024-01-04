import argparse
import json
import logging
import sys, os
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
import clarity
#from clarity.evaluator.haspi import haspi_v2
#from clarity.evaluator.hasqi import hasqi_v2
from metrics.hasqi_ex import hasqi_v2 
from clarity.utils.audiogram import Audiogram, Listener
from clarity.enhancer.nalr import NALR
from clarity.enhancer.compressor import Compressor
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np
import scipy.signal as signal
from einops import rearrange

class HearingLossRand():
    def __init__(self, gain=3):
        self.gain = gain

    def get_hearing_loss(self):
        v = np.random.randint(0, 4)
        # center freqs:     250, 500, 1000, 2000, 4000, 6000
        if v == 0:
            _loss_type = 'none'
            _loss = np.array([0, 0, 0, 0, 0, 0])
        elif v == 1:
            _loss_type = 'mild'
            _loss = np.array([10, 15, 19, 25, 31, 35])
        elif v == 2:
            _loss_type = 'moderate'
            _loss = np.array([20, 20, 25, 35, 45, 50])
        elif v == 3:
            _loss_type = 'severe'
            _loss = np.array([19, 28, 40, 52, 58, 58])
        else:
            raise ValueError
        r = self.gain * (np.random.rand(6) - 0.5)
        _loss = _loss + r
        _loss = np.where(_loss >= 0., _loss, 0.)
        return _loss, _loss_type
    
'''    
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
'''

class HASQI:
    def __init__(self):
        self.sample_rate = 16000
        self.level1 = 65 # level in dB SPL corresponding to RMS=1
        self.equalization_mode = 1
        #self.hearing_loss = HearingLossRand()
        #self.freqs = np.array([250, 500, 1000, 2000, 4000, 6000])

    def eval(self, x, y, audiogram):
        #_loss, _loss_type = self.hearing_loss.get_hearing_loss()
        #audiogram = Audiogram(levels=_loss, frequencies=self.freqs)
        score, non_linear, linear, _, features = hasqi_v2(x, self.sample_rate,
                                                          y, self.sample_rate,
                                                          audiogram,
                                                          self.equalization_mode,
                                                          self.level1)
        #print(score)
        return score, non_linear, linear, features

'''
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
'''

def read_audio(path, normalize=True):
    wave, sr = torchaudio.load(path)
    if normalize:
        rms = torch.sqrt(torch.sum(torch.square(wave))/wave.shape[-1])
        wave /= rms
    return wave

class HearingAid:
    def __init__(self):
        self.enhancer = NALR(nfir=220, sample_rate=16000)
        self.compressor = Compressor(threshold=0.35,
                                     attenuation=0.1,
                                     attack=50,
                                     release=1000,
                                     rms_buffer_size=0.064)
        #hearing_loss = get_hearing_loss(loss_type)
        #freqs = np.array([250, 500, 1000, 2000, 4000, 6000])
        #audiogram = Audiogram(levels=hearing_loss, frequencies=freqs)
        #self.nalr_fir, _ = self.enhancer.build(audiogram)
        
    def process(self, signal, audiogram):
        nalr_fir, _ = self.enhancer.build(audiogram)
        out = self.enhancer.apply(nalr_fir, signal)
        out, _, _ = self.compressor.process(out)
        
        return out
    
def main(args):

    hearing_loss = HearingLossRand()
    freqs = np.array([250, 500, 1000, 2000, 4000, 6000])

    #_haspi = HASPI(args.hearing_loss)
    _hasqi = HASQI()
    _hearing_aid = HearingAid()
    
    df_out = pd.DataFrame(index=None, 
                          columns=['key', 'target', 'source', 'hasqi', 'linear', 'non_linear', 'source_feats', 'target_feats', 'loss', 'loss_type'])
    keys = []
    hasqi, non_linear, linear = [], [], []
    targets, sources = [], []
    tgt_feats, src_feats = [],[]    
    losses, loss_types = [], []

    df = pd.read_csv(args.input_csv)

    target_key = args.target_key
    source_key = args.source_key
    with torch.no_grad():
        for index, row in df.iterrows():

            loss, loss_type = hearing_loss.get_hearing_loss()
            audiogram = Audiogram(levels=loss, frequencies=freqs)

            if 'key' in row.keys():
                key = row['key']
            else:
                key = os.path.splitext(os.path.basename(row[source_key]))[0]
            
            keys.append(key)
            targets.append(row[target_key])
            sources.append(row[source_key])

            target = read_audio(row[target_key])
            source = read_audio(row[source_key])

            target = rearrange(target, 'c t -> (c t)').cpu().detach().numpy()
            target = _hearing_aid.process(target, audiogram)
            source = rearrange(source, 'c t -> (c t)').cpu().detach().numpy()
            # source signal is processed and compressed in hasqi_v2 function
           
            score, nl, lin, features = _hasqi.eval(source, target, audiogram)

            hasqi.append(score)
            non_linear.append(nl)
            linear.append(lin)
            src_emb, src_bm, src_sl, tgt_emb, tgt_bm, tgt_sl = features
            #print(target.shape)
            #print(tgt_emb.shape)
            #print(tgt_bm.shape)
            #print(tgt_sl.shape)

            path = os.path.join(os.path.dirname(row[args.target_key]), key) + '_source.npz'
            #np.savez(path, src_emb, src_bm, src_sl)
            src_feats.append(path)

            path = os.path.join(os.path.dirname(row[args.target_key]), key) + '_target.npz'
            #np.savez(path, tgt_emb, tgt_bm, tgt_sl)
            tgt_feats.append(path)
            
            path = os.path.join(os.path.dirname(row[args.target_key]), key) + '_loss.npz'
            #np.savez(path, loss)
            losses.append(path)
            loss_types.append(loss_type)

    df_out['key'], df_out['target'], df_out['source'] = keys, targets, sources
    df_out['hasqi'] = hasqi
    df_out['non_linear'] = non_linear
    df_out['linear'] = linear
    df_out['source_feats'], df_out['target_feats'] = src_feats, tgt_feats
    df_out['loss'], df_out['loss_type'] = losses, loss_types
    df_out.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--sample_rate', default=16000, type=int)
    parser.add_argument('--source_key', type=str)
    parser.add_argument('--target_key', type=str)

    args = parser.parse_args()

    main(args)

