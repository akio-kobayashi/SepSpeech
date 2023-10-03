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

    def __init__(self, source_path, target_path, rate=0.1):
        super().__init__()

        self.src_df = pd.read_csv(source_path)
        self.tgt_df = pd.read_csv(target_path)
        self.rate = rate

    def __len__(self):
        return len(self.src_keys)

    def normalize(self, signal):
        std, mean = torch.std_mean(signal)
        return (signal - mean)/std
    
    def __getitem__(self, idx):
        row = self.src_df.iloc[idx]
        source,_ = torchaudio.load(row['source'])
        source = self.normalize(source)
        source_key = row['source_key']
        source_embed = torch.load(row['source_embed'])
        source_utterance = row['utterance']

        if np.random.rand() > self.rate :
            row = self.tgt_df.query('utterance=@source_utterance').sample()
            target,_ = torchaudio.load(row['target'])
            target = self.normalize(target)
            target_key = row['target_key']   
            target_embed = torch.load(row['target_embed'])
        else:
            target = source
            target_key = source_key
            target_embed = source_embed
        
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
