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
import json

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, speaker_path=None) -> None:
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.speaker2id={}
        if speaker_path is not None:
            with open(speaker_path, 'r') as f:
                self.speaker2id = json.load(f)
                
    def count_speakers(self):
        self.speaker2id['<UNK>']=0
        counter = 1
        for index, row in self.df.iterrows():
            speaker = row['speaker']
            if speaker not in self.speaker2id.keys():
                self.speaker2id[speaker] = counter
                counter += 1
            
    def dump_speakers(self, path):
        with open(path, 'w') as wf:
            json.dump(self.speaker2id, wf)

    def num_speakers(self):
        return len(self.speaker2id.keys())
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int):
        row = self.df.iloc[idx]

        source_path = row['source']
        source, sr = torchaudio.load(source_path)
        std, mean = torch.std_mean(source, dim=-1)
        source = (source - mean)/std

        source_speaker = int(self.speaker2id[row['speaker']])
        
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

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_json', type=str, default=None)
    args=parser.parse_args()

    data=SpeechDataset(args.input_csv)
    data.count_speakers()
    data.dump_speakers(args.output_json)
    print(data.num_speakers())
    
