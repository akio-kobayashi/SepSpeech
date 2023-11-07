import argparse
import json
import logging
import sys, os
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
#from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from lite.solver import LitSepSpeaker
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np
import scipy.signal as signal
from einops import rearrange

def align(source:Tensor, target:Tensor):
    correlation = signal.correlate(rearrange(target, 'c t -> (c t)').detach().numpy(),
    rearrange(source,'c t -> (c t)').detach().numpy(), mode='full')
    lags = signal.correlation_lags(target.shape[-1], source.shape[-1], mode='full')
    lag = lags[np.argmax(correlation)]

    if lag > 0:
        target = target[:, lag:]
        source = source[:, :target.shape[-1]]
    else:
        lag = -lag
        source = source[:, lag:]
        target = target[:, :source.shape[-1]]

    return source, target

def read_audio(path):
    wave, sr = torchaudio.load(path)
    return wave

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    _pesq = PerceptualEvaluationSpeechQuality(sample_rate, 'wb').to(device)
    _stoi = ShortTimeObjectiveIntelligibility(sample_rate, extended=True).to(device)
    
    df_out = pd.DataFrame(index=None, 
                          columns=['key', 'mixture', 'source', 'enroll', 'estimate', 'mix_result', 'est_result', 'mix_pesq', 'mix_stoi', 'mix_sdr', 'est_pesq', 'est_stoi', 'est_sdr'])
    keys = []
    est_pesq, est_stoi = [], []
    mix_pesq, mix_stoi = [], []
    mixtures, sources, estimates = [], [], []
    df = pd.read_csv(args.input_csv)

    with torch.no_grad():
        for index, row in df.iterrows():

            key = os.path.splitext(os.path.basename(row['source']))[0]
            keys.append(key)
            mixtures.append(row['mixture'])
            sources.append(row['source'])

            mixture = read_audio(row['mixture'])
            source = read_audio(row['source'])
            estimate = read_audio(row['estimate'])

            mixture, source = align(mixture, source)

            mix_pesq.append(_pesq(mixture.cuda(), source.cuda()).cpu().detach().numpy())
            mix_stoi.append(_stoi(mixture.cuda(), source.cuda()).cpu().detach().numpy())

            est_pesq.append(_pesq(estimate.cuda(), source.cuda()).cpu().detach().numpy())
            est_stoi.append(_stoi(estimate.cuda(), source.cuda()).cpu().detach().numpy())
            
    df_out['key'], df_out['mixture'], df_out['source'], df_out['estimate'] = keys, mixtures, sources, estimates
    df_out['mix_pesq'], df_out['mix_stoi'] = mix_pesq, mix_stoi
    df_out['est_pesq'], df_out['est_stoi'] = est_pesq, est_stoi

    pesq, stoi, sdr = np.mean(mix_pesq), np.mean(mix_stoi)
    print("mixture:  PESQ = %.4f , STOI = %.4f" % (pesq, stoi))
    
    pesq, stoi = np.mean(est_pesq), np.mean(est_stoi)
    print("estimate: PESQ = %.4f , STOI = %.4f" % (pesq, stoi))

    df_out.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    args = parser.parse_args()

    main(args)

