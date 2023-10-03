import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import pandas as pd
import more_itertools
import torch
import torchaudio

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.src_keys)

    def normalize(self, signal):
        std, mean = torch.std_mean(signal)
        return (signal - mean)/std
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source,_ = torchaudio.load(row['source'])
        source = self.normalize(source)
        target,_ = torchaudio.load(row['target'])
        target = self.normalize(target)

        source_key = row['source_key']
        target_key = row['target_key']
        
        source_empbed = torch.load(row['source_embed'])
        target_empbed = torch.load(row['target_embed'])

        return torch.t(source), torch.t(target), source_embed, target_embed, source_key, target_key
    
def data_processing(data, data_type="train"):

    _src, _tgt = [], []
    _src_embed, _tgt_embed= [], []
    _src_keys, _tgt_keys = [], []
    _src_lengths, _tgt_lengths = [], []

    for source, target, source_embed, target_embed, source_key, target_key in data:
        T, C = source.shape
        _src_lengths.append(T)
        source_embed = source_embed.repeat((T, 1)) # (T, F)
        _src.append(source)
        _src_embed.append(source_embed)
        _src_keys.append(source_key)

        T, C = target.shape
        _tgt_lengths.append(T)
        target_embed = target_embed.repeat((T, 1)) # (T, F)
        _tgt.append(target)
        _tgt_embed.append(target_embed)
        _tgt_keys.append(target_key)
        
    _src = nn.utils.rnn.pad_sequence(_src, batch_first=True)
    _tgt = nn.utils.rnn.pad_sequence(_tgt, batch_first=True)
    _src_embed = nn.utils.rnn.pad_sequence(_src_embed, batch_first=True)
    _tgt_embed = nn.utils.rnn.pad_sequence(_tgt_embed, batch_first=True)

    return _src, _tgt, _src_embed, _tgt_embed, _src_lengths, _tgt_lengths, _src_keys, _tgt_keys
