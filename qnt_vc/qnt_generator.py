import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import pandas as pd
import more_itertools
import torch
import torchaudio
import json
from einops import rearrange

class QntSpeechDataset(torch.utils.data.Dataset):

    def __init__(self, source_path, target_path, speaker_path, rate=0.1):
        super().__init__()

        self.src_df = pd.read_csv(source_path)
        self.tgt_df = pd.read_csv(target_path)
        with open(speaker_path, 'r') as f:
            self.speaker2id = json.load(f)
        self.rate = rate

    def __len__(self):
        return len(self.src_df)

    def __getitem__(self, idx):
        row = self.src_df.iloc[idx]
        source,_ = torch.load(row['source'])
        source = rearrange(source, 'f c t -> c t f')
        source_id = self.speaker2id[row['speaker']]
        source_utterance = row['utterance']

        if np.random.rand() > self.rate :
            row = self.tgt_df.query('utterance=@source_utterance').sample()
            target,_ = torch.load(row['target'])
            target = rearrange(target, 'f c t -> c t f')
            target_id = self.speaker2id(row['speaker'])
        else:
            target = source
            target_id = source_id

        return source, target, source_id, target_id
    
def data_processing(data):
    _src, _tgt = [], []
    _src_id, _tgt_id= [], []

    for source, target, source_id, target_id in data:
        C, N, T = source.shape
        _src.append(source)
        _src_id.append(source_id)

        C, N, T = target.shape
        _tgt.append(target)
        _tgt_id.append(target_id)

    return _src, _src_id, _tgt, _tgt_id

if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml
    import pandas as pd

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpus', nargs='*', type=int)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    
    df = pd.read_csv(config['dataset']['train']['source_path'])
    for index, row in df.iterrows():
        print(row['source'])
        data = torch.load(row['source'])
    df = pd.read_csv(config['dataset']['train']['target_path'])
    for index, row in df.iterrows():
        print(row['source'])
        data = torch.load(row['source'])
