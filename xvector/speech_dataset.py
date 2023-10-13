import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from typing import Tuple, List
from torch import Tensor
import torchaudio
import scipy.signal as signal
from einops import rearrange

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, sample_rate=16000) -> None:
        super().__init__()

        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int):
        row = self.df.iloc[idx]

        source_path = row['source']
        source, sr = torchaudio.load(source_path)
        std, mean = torch.std_mean(source, dim=-1)
        source = (source - mean)/std

        source_speaker = int(row['index'])
        
        return torch.t(source), source_speaker
    
def data_processing(data):
    sources = []
    lengths = []
    speakers = []

    for source, speaker in data:
        sources.append(source)
        lengths.append(len(source))
        speakers.append(speaker)

    sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)
    speakers = torch.from_numpy(np.array(speakers)).clone()

    sources = sources.squeeze()

    if sources.dim() == 1:
        sources = sources.unsqueeze(0)
        
    return sources, lengths, speakers
