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
from augment.opus_augment_simulate import OpusAugment
from augment.reverb_augment import ReverbAugment
from models.qnt_encode import EncodecQuantizer
from einops import rearrange

'''
    音声強調用データの抽出
    入力: 音声CSV，エンロールCSV
    出力: 混合音声，ソース音声，エンロール音声，話者インデックス
        音声データはtorch.Tensor
'''
class QntSpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, enroll_path:str, sample_rate=16000, segment=0, enroll_segment=0) -> None:
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.segment = segment if segment > 0 else 0
        self.sample_rate = sample_rate
        if self.segment > 0:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            self.df = self.df[self.df['length'] <= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = 0

        self.enroll_df = pd.read_csv(enroll_path)
        self.enroll_segment = enroll_segment if enroll_segment > 0 else 0
        if self.enroll_segment > 0:
            max_len = len(self.enroll_df)
            self.seg_len = int(self.enroll_segment * self.sample_rate)
            self.enroll_df = self.enroll_df[self.enroll_df['length'] <= self.seg_len]
            print(
                f"Drop {max_len - len(self.enroll_df)} utterances from {max_len} "
                f"(shorter than {enroll_segment} seconds)"
            )

        self.quantizer = EnocdecQuantizer()
        self.embed = nn.Embedding(num_embedding, embedding_dim)
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.df.iloc[idx]
        # mixture path
        mixture = self.embed(self.quantizer.encode_from_file(row['mixture']))
        source = self.embed(self.quantizer.encode_from_file(row['source']))
        source_speaker = int(row['index'])
        filtered = self.enroll_df.query('index == @source_speaker')
        enroll = self.embed(self.quantizer.encode_from_file(filtered.iloc[np.random.randint(0, len(filtered))]['source']))
        return mixture, source, enroll, source_speaker
    
def data_processing(data:Tuple[Tensor,Tensor,Tensor,Tensor]) -> Tuple[Tensor, Tensor, Tensor, List, List]:
    mixtures = []
    sources = []
    enrolls = []
    lengths = []
    speakers = []

    Q = 1024
    for mixture, source, enroll, speaker in data:
        # c = 1
        mixtures.append(rearrange(mixture, 'c q t h -> t (c q h)'))
        sources.append(rearrange(source, 'c q t h -> t (c q h)'))
        enrolls.append(rearrange(enroll, 'c q t h -> t (c q h)'))
        lengths.append(len(mixture))
        speakers.append(speaker)

    mixtures = nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
    sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)
    enrolls = nn.utils.rnn.pad_sequence(enrolls, batch_first=True)
    speakers = torch.from_numpy(np.array(speakers)).clone()

    assert mixtures.dim() == 3

    mixtures = rearrange(mixtures, 'b t (q h) -> b q t h', q=Q)
    sources = rearrange(sources, 'b t (q h) -> b q t h', q=Q)
    enrolls = rearrange(enrolls, 'b t (q h) -> b q t h', q=Q)

    return mixtures, sources, enrolls, lengths, speakers

if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()
    
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    csv_path = config['dataset']['train']['csv_path']
    noise_csv_path = config['dataset']['train']['noise_csv_path']
    enroll_csv_path = config['dataset']['train']['enroll_csv_path']

    dataset = QntSpeechDataset(csv_path,
                               enroll_csv_path
                               )
    mixture, source, enroll, speaker = dataset.__getitem__(10)

