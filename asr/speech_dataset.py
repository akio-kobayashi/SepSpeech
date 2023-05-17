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

'''
    音声認識用データの抽出
    入力: 音声CSV，エンロールCSV
    出力: 混合音声，ソース音声，エンロール音声，話者インデックス
        音声データはtorch.Tensor
'''
class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, enroll_path:str, segment=10) -> None:
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

        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=,
                                                              nfft=,
                                                              win_length=,
                                                              hop_length=,
                                                              n_mels=,
                                                              window_fn=torch.hamming_window
                                                              )
        self.mean=0.
        self.std=1.

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.df.iloc[idx]
        
        source_path = row['source']
        source, _ = torchaudio.load(source_path, normalize=True)
        log_melspec = torch.log(self.transorm(source)+self.eps) # (1, n_mels, time)
        log_melspac = (log_melspec - self.mean)/self.std

        label_str = row['label']
        label = [ int(x) for x in label_str.split() ]
        
        return source

'''
# On-the-fly ミキシング
class SpeechDatasetLive(SpeechDataset):
    def __init__(self, csv_path:str, noise_csv_path:str, enroll_path:str, sample_rate=16000, segment=None) -> None:
        super().__init__(csv_path, enroll_path, sample_rate=16000, segment=None)
        self.noise_df = pd.read_csv(noise_csv_path)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.df.iloc[idx]
        source_path = row['source']
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = -1
        assert os.path.exists(source_path)
        source, sr = torchaudio.load(source_path)
        source = source[start:stop]

        source_speaker = int(row['index'])
        filtered = self.enroll_df.query('index == @source_speaker')
        enroll = filtered.iloc[randint(0, len(filtered)-1)]['source']
        enroll = enroll[start:stop]

        noise_row = self.noise_df[random.randint(0, len(self.noise_df))]
        self.noise_path = noise_row['noise']
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + sele.seg_len
        else:
            start = random.randint(0, row["length"] - len(source))
            stop = start + len(source)
        if os.path.exists(noise_path):
            noise, sr = torchaudio.load(noise_path)
            noise = noise[start:stop]
        else:
            noise=None

        return source, noise, enroll, source_speaker
'''

def data_processing(data:Tuple[Tensor,Tensor,Tensor,Tensor]) -> Tuple[Tensor, Tensor, Tensor, list, list]:
    mixtures = []
    sources = []
    enrolls = []
    lengths = []
    speakers = []

    for mixture, source, enroll, speaker in data:
        # w/o channel
        mixtures.append(mixture)
        if source is not None:
            sources.append(source)
        enrolls.append(enroll)
        lengths.append(len(mixture))
        #speakers.append(torch.from_numpy(speaker.astype(np.int)).clone())
        speakers.append(speaker)

    mixtures = nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
    if len(sources) > 0:
        sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)
    enrolls = nn.utils.rnn.pad_sequence(enrolls, batch_first=True)
    speakers = torch.from_numpy(np.array(speakers)).clone()

    mixtures = mixtures.squeeze()
    sources = sources.squeeze()
    enrolls = enrolls.squeeze()

    if mixtures.dim() == 1:
        mixtures = mixtures.unsqueeze(0)
        sources = sources.unsqueeze(0)
        enrolls = enrolls.unsqueeze(0)
        
    return mixtures, sources, enrolls, lengths, speakers
