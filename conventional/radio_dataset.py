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
import conventional.speech_dataset as sd
from augment.speech_augment import LowPassRadio, WhiteNoiseAugment, NarrowBandNoiseAugment, RandomAmplitudeModulationAugment
from augment.fading import FadingAugment

def get_divisor(model):

    start = -1
    end = -1
    with torch.no_grad():
        for n in range(1000, 1100):
            x = torch.rand(4, n)
            o = model(x)
            if x.shape[-1] == o.shape[-1] :
                if start < 0:
                    start = n
                else:
                    end = n
                    break
    return end - start

def padding(x, divisor):
    pad_value = divisor - x.shape[-1] % divisor -1
    return F.pad (x, pad=(1, pad_value ), value=0.)

'''
    音声強調用データの抽出
    入力: 音声CSV
    出力: 混合音声，ソース音声
        音声データはtorch.Tensor
'''
class RadioDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str,
                 config:dict,
                 sample_rate=16000,
                 segment=0,
                 divisor=0) -> None:
        super(RadioDataset, self).__init__()

        self.df = pd.read_csv(csv_path)
        self.segment = segment if segment > 0 else None
        self.sample_rate = sample_rate
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            self.df = self.df[self.df['length'] <= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None

        # augmentation
        self.lowpass = LowPassRadio(config).cuda()
        if config['augment']['white_noise']['use']:
            self.white_noise = WhiteNoiseAugment(config).cuda()
        else:
            self.white_noise = None
        if config['augment']['narrow_band']['use']:
            self.narrow_band = NarrowBandNoiseAugment(config).cuda()
        else:
            self.narrow_band = None
        #self.random_amp = RandomAmplitudeModulationAugment(config)
        if config['augment']['fading']['use']:
            self.fading_augment = FadingAugment(config).cuda()
        else:
            self.fading_augment = None
            
        self.divisor = divisor
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor]:
        row = self.df.iloc[idx]

        source_path = row['source']
        with torch.no_grad():
            source, sr = torchaudio.load(source_path)
            std, mean = torch.std_mean(source, dim=-1)
            source = (source - mean)/std

            if self.divisor > 0 and source.shape[-1] % self.divisor > 0:
                source = sd.padding(source, self.divisor)

            if self.white_noise is not None:
                noise = self.white_noise(source)
            else:
                noise = torch.zeros_like(source)
            if self.narrow_band is not None:
                noise += self.narrow_band(source)

            if self.fading_augment is not None:
                mixture = self.fading_augment(source) + noise
            else:
                mixture = source + noise

            mixture = self.lowpass(mixture)
            std, mean = torch.std_mean(mixture, dim=-1)
            mixture = (mixture - mean)/std
            #mixture = self.random_amp(self.lowpass(source)) + noise
        
        return torch.t(mixture), torch.t(source)

def data_processing(data:Tuple[Tensor,Tensor]) -> Tuple[Tensor, Tensor, list]:
    mixtures = []
    sources = []
    lengths = []

    for mixture, source in data:
        # w/o channel
        mixtures.append(mixture)
        sources.append(source)
        lengths.append(len(mixture))

    mixtures = nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
    sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)

    mixtures = mixtures.squeeze()
    sources = sources.squeeze()

    if mixtures.dim() == 1:
        mixtures = mixtures.unsqueeze(0)
        sources = sources.unsqueeze(0)
        
    return mixtures, sources, lengths
