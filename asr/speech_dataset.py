import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from typing import Tuple
from torch import Tensor
import torchaudio
from typing import Tuple

'''
    音声認識用データの抽出
    入力: 音声CSV
    出力: ソース音声，ラベル，キー
        音声データはtorch.Tensor
'''
class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, config:dict, segment=10) -> None:
        super(SpeechDataset, self).__init__()

        self.df = pd.read_csv(csv_path)
        self.segment = segment if segment > 0 else None
        self.sample_rate = sample_rate
        if self.segment is not None:
            max_len = len(self.df)
            self.max_segment_length = int(self.segment * self.sample_rate)
            self.df = self.df[self.df['length'] <= self.max_segment_length]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.max_segment_length = None

        self.wav2spec = torchaudio.transforms.Spectrogram(
            sample_rate=config['sample_rate'],
            nfft=config['nfft'],
            win_length=config['win_length'],
            hop_length=config['hop_length'],
            window_fn=torch.hamming_window
        )
        self.spec2mel = torchaudio.transforms.MelScale(
            n_mels=config['n_mels'],
            sample_rate=config['sample_rate'],
            n_stft=config['nfft']//2+1
        )

        npz=np.load(config['global_mean_std'])
        mean, std = npz['mean'], npz['std']
        
        self.eps = 1.e-8
        self.specaug = True if config['specaug'] else False
        if self.specaug:
            self.time_stretch = torchaudio.transforms.TimeStretch(n_freq=config['nfft']+1)
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=config['freq_mask'])
            self.time_masking = torchaudio.transforms.TimeMasking(freq_mask_param=config['time_mask'])
 

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.df.iloc[idx]
        
        source_path = row['source']
        source, _ = torchaudio.load(source_path, normalize=True)
        spec = self.wav2spec(source)
        if self.specaug:
            rnd = np.random.rand() * 2/10.0 + 0.9 # 0.9 -- 1.1
            spec = self.time_stretch(spec, rnd)
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        melspec = torch.log(self.spec2mel(spec)+self.eps) # (1, n_mels, time)
        melspec = torch.t(melspec.squeeze()) # (time, n_mels)
        melspac = (melspec - self.mean)/self.std

        label_path = row['label']
        with open(label_path, 'r') as  f:
            line = f.readline()
            label = [ int(x) for x in line.strip().split() ]
            label = torch.tensor(label, dtype=int)
        return source, label, row['key']

def data_processing(data:Tuple[Tensor, list, str]) -> Tuple[Tensor, Tensor, list, list]:
    inputs = []
    labels = []
    input_lengths = []
    label_lengths = []
    keys=[]

    for source, label, key in data:
        # w/o channel
        inputs.append(source)
        labels.append(label)
        input_lengths.append(source.shape[0]) # (time, n_mels)
        label_lengths.append(len(label))
        keys.append(key)

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)
        
    return inputs, labels, input_lengths, label_lengths, keys

def compute_global_mean_std(csv_path:str, config:dict) -> Tuple[Tensor, Tensor]:
    wav2spec = torchaudio.transforms.Spectrogram(
        sample_rate=config['sample_rate'],
        nfft=config['nfft'],
        win_length=config['win_length'],
        hop_length=config['hop_length'],
        window_fn=torch.hamming_window
    )
    spec2mel = torchaudio.transforms.MelScale(
        n_mels=config['n_mels'],
        sample_rate=config['sample_rate'],
        n_stft=config['nfft']//2+1
    )
    eps = 1.e-8

    sum, sq_sum = 0., 0.
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        source_path, label_path, key = row['source'], row['label'], row['key']
        source, _ = torchaudio.load(source_path, normalize=True)
        spec = wav2spec(source)
        melspec = torch.log(spec2mel(spec)+eps) # (1, n_mels, time)
        melspec = torch.t(melspec.squeeze()) # (time, n_mels)

        sum += melspec
        sq_sum += melspec*melspec
    mean = sum/len(df)
    sq_mean = sq_sum/len(df)
    std = sq_mean - mean*mean

    assert std > 0
    std = torch.sqrt(std)

    mean = mean.to('cpu').detach().numpy().copy()
    std = std.to('cpu').detach().numpy().copy()

    path = config['global_mean_std']
    np.savez(path, mean=mean, std=std)

    return mean, std
