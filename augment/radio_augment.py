import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from typing import Tuple
from torch import Tensor
import torchaudio
from augment.speech_augment import LowPassRadio, WhiteNoiseAugment, NarrowBandNoiseAugment, RandomAmplitudeModulationAugment
from augment.fading import FadingAugment
import argparse
import yaml

def padding(x, divisor):
    pad_value = divisor - x.shape[-1] % divisor -1
    return F.pad (x, pad=(1, pad_value ), value=0.)

def radio_augment(config, args, device='cuda'):

    df = pd.read_csv(args.input_csv)

    # augmentation
    if config['augment']['white_noise']['use']:
        white_noise = WhiteNoiseAugment(config).to(device)
    else:
        white_noise = None
    if config['augment']['narrow_band']['use']:
        narrow_band = NarrowBandNoiseAugment(config).to(device)
    else:
        narrow_band = None
            
    if config['augment']['fading']['use']:
        fading_augment = FadingAugment(config).to(device)
    else:
        fading_augment = None

    if config['augment']['lowpass']['use']:
            lowpass = LowPassRadio(config).to(device)
    else:
        lowpass = None
            
    output_root = args.output_root

    for idx, row in df.iterrows():
        with torch.no_grad():
            source, sr = torchaudio.load(row['clean'])

            source = source.to(device)
            '''
            if white_noise is not None:
                noise = white_noise(source)
            else:
                noise = torch.zeros_like(source)
                    
            if narrow_band is not None:
                noise += narrow_band(source)
                
            if fading_augment is not None:
                target = fading_augment(source) + noise
            else:
                target = source + noise
            '''
            if fading_augment is not None:
                target = fading_augment(source)
            else:
                target = source
                
            if white_noise is not None:
                target = white_noise(target)
                    
            if narrow_band is not None:
                target = narrow_band(target)
            
            if lowpass is  not None:
                target = lowpass(target)

            basename = os.path.splitext(os.path.basename(row['clean']))[0].replace('_clean', '_augment')
        
            path = os.path.join(output_root, basename+'.wav')
            torchaudio.save(filepath=path, src=target.cpu(), sample_rate=sr, bits_per_sample=16, encoding='PCM_S')
            df.loc[row['key'], 'noisy'] = path

    df.to_csv(args.output_csv)
        
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--input_csv', type=str, required=True)
parser.add_argument('--output_root', type=str, required=True)
parser.add_argument('--output_csv', type=str, required=True)
parser.add_argument('--use_lowpass', action='store_true')
parser.add_argument('--use_fading', action='store_true')
parser.add_argument('--use_noise', action='store_true')
parser.add_argument('--use_band_noise', action='store_true')

args=parser.parse_args()

with open(args.config, 'r') as yf:
    config = yaml.safe_load(yf)

if args.use_lowpass:
    config['augment']['lowpass']['use']=True
else:
    config['augment']['lowpass']['use']=False
    
if args.use_fading:
    config['augment']['fading']['use']=True
else:
    config['augment']['fading']['use']=False

if args.use_noise:
    config['augment']['white_noise']['use']=True
else:
    config['augment']['white_noise']['use']=False
    
if args.use_band_noise:
    config['augment']['narrow_band']['use']=True
else:
    config['augment']['narrow_band']['use']=False
    
radio_augment(config, args)
            
