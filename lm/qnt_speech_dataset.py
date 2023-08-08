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
from lm.qnt_encode import EncodecQuantizer

class QntSpeechDataSet(torch.utils.data.Dataset):
    def __init__(self, csv_path:str,
                 enroll_path:str,
                 sample_rate=16000,
                 segment=0,
                 enroll_segment=0,
                 padding_value=0) -> None:
        super().__init__()

        self.df = pd.read_csv(csv_path)
        if segment > 0:
            max_len = len(self.df)
            seg_len = int(segment * sample_rate)
            self.df = self.df[self.df['length'] <= seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )

        self.enroll_df = pd.read_csv(enroll_path)
        if enroll_segment > 0:
            max_len = len(self.enroll_df)
            seg_len = int(enroll_segment * sample_rate)
            self.enroll_df = self.enroll_df[self.enroll_df['length'] <= seg_len]
            print(
                f"Drop {max_len - len(self.enroll_df)} utterances from {max_len} "
                f"(shorter than {enroll_segment} seconds)"
            )
        
        self.quantizer = EncodecQuantizer()

    def get_padded_value(x):
        B, C, T = x.shape
        v = self.padding_value - T % self.padding_value
        pad = torch.zeros((1, C, 1))
        pad[:, :, 0] = mixture[:, :, -1]
        pad = pad.repeat(1, 1, v)
        return torch.concat ([x, pad], dim=-1)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.df.iloc[idx]
        start, stop = 0, -1
        
        # quantized data = B, C, T
        mixture_path = row['mixture']
        assert os.path.exists(mixture_path)
        mixture = self.quantizer.encode_from_file(mixture_path)

        source_path = row['source']
        assert os.path.exists(source_path)
        source = self.quantizer.encode_from_file(source_path)
            
        source_speaker = int(row['index'])
        filtered = self.enroll_df.query('index == @source_speaker')
        enroll_path = filtered.iloc[random.randint(0, len(filtered)-1)]['source']
        assert os.path.exists(enroll_path)
        enroll = self.quantizer.encode_from_file(enroll_path)

        if self.padding_value > 1:
            mixture = self.get_padded_value(mixture)
            source = self.get_padded_value(source)

        speaker = row['speaker']
        
        return mixture, source, enroll, speaker

def pad_sequence_3d(xs):
    B, C, T = xs[0]
    xs = [ rearrange(x, 'b c t -> b (c t)') for x in xs ]
    seq = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    seq = rearrange(seq, 'b (c t) -> b c t', c=C)
    return seq

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
        speakers.append(speaker)

    mixtures = pad_sequence_3d(mixtures)
    sources = pad_sequence_3d(sources)
    enrolls = pad_sequence_3d(enrolls)
    speakers = torch.from_numpy(np.array(speakers)).clone()

    mixtures = mixtures.squeeze()
    sources = sources.squeeze()
    enrolls = enrolls.squeeze()

    if mixtures.dim() == 2:
        mixtures = mixtures.unsqueeze(0)
        sources = sources.unsqueeze(0)
        enrolls = enrolls.unsqueeze(0)
        
    return mixtures, sources, enrolls, lengths, speakers

if __name__ == '__main__':
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    dataset = QntSpeechDataSet(
        **config['dataset']['valid'], 
        **config['dataset']['segment']
    )
    loader = data.DataLoader(dataset=dataset,
                             **config['dataset']['process'],
                             pin_memory=True,
                             shuffle=False, 
                             collate_fn=lambda x: data_processing(x)
                            )
    mixture, source, enroll, speaker = loader.__getitem__(0)
    print(mixture.shape)
    print(source.shape)
    print(enroll.shape)
    