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

'''
    音声強調用データの抽出
    入力: 音声CSV，エンロールCSV
    出力: 混合音声，ソース音声，エンロール音声，話者インデックス
        音声データはtorch.Tensor
'''
class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, enroll_path:str, sample_rate=16000, segment=0, enroll_segment=0, padding_value=0) -> None:
        super(SpeechDataset, self).__init__()

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

        self.enroll_df = pd.read_csv(enroll_path)
        self.enroll_segment = enroll_segment if enroll_segment > 0 else None
        if self.enroll_segment is not None:
            max_len = len(self.enroll_df)
            self.seg_len = int(self.enroll_segment * self.sample_rate)
            self.enroll_df = self.enroll_df[self.enroll_df['length'] <= self.seg_len]
            print(
                f"Drop {max_len - len(self.enroll_df)} utterances from {max_len} "
                f"(shorter than {enroll_segment} seconds)"
            )
        self.padding_value = padding_value

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.df.iloc[idx]
        # mixture path
        self.mixture_path = row['mixture']
        start = 0
        stop = -1

        source_path = row['source']
        if os.path.exists(source_path):
            source, sr = torchaudio.load(source_path)
            std, mean = torch.std_mean(source, dim=-1)
            source = (source - mean)/std
            source = source[:, start:stop]
        else:
            source = None
        mixture, sr = torchaudio.load(self.mixture_path)
        if self.padding_value > 0:
            mixture = self.get_padded_value(mixture)
        
        std, mean = torch.std_mean(mixture, dim=-1)
        mixture = (mixture - mean)/std
        mixture = mixture[:, start:stop]
        source_speaker = int(row['index'])
        filtered = self.enroll_df.query('index == @source_speaker')
        enroll_path = filtered.iloc[np.random.randint(0, len(filtered))]['source']
        assert os.path.exists(enroll_path)
        enroll, sr = torchaudio.load(enroll_path)
        std, mean = torch.std_mean(enroll, dim=-1)
        enroll = (enroll - mean)/std
        return torch.t(mixture), torch.t(source), torch.t(enroll), source_speaker
    
    def get_padded_value(self, x):
        v = self.padding_value - x.shape[-1] % self.padding_value
        x = F.pad(x, pad=(1, v), value=0.)
        return x

# On-the-fly ミキシング
class SpeechDatasetOTFMix(SpeechDataset):
    def __init__(self, csv_path:str, noise_csv_path:str, enroll_csv_path:str, 
                 mixing:dict, augment:dict,
                 sample_rate=16000,
                 segment=0,
                 enroll_segment=0,
                 padding_value=0) -> None:
        super().__init__(csv_path, enroll_csv_path, sample_rate, segment, enroll_segment, padding_value)
        self.noise_df = pd.read_csv(noise_csv_path)
        self.min_snr=mixing['min_snr']
        self.max_snr=mixing['max_snr']
        self.opus, self.source_reverb, self.noise_reverb = None, None, None
        if augment['opus']['use']:
            self.opus = OpusAugment(**augment['opus'])
        if augment['reverb']['use']:
            self.source_reverb = ReverbAugment(**augment['reverb']['params'],
                                               source_loc=augment['reverb']['source_loc'],
                                               loc_range=augment['reverb']['source_loc_range'],
            )
            self.noise_reverb = ReverbAugment(**augment['reverb']['params'],
                                              source_loc=augment['reverb']['noise_loc'],
                                              loc_range=augment['reverb']['noise_loc_range'],
            )
        
    def rms(self, wave):
        return torch.sqrt(torch.mean(torch.square(wave)))

    def adjusted_rms(self, _rms, snr):
        return _rms / (10**(float(snr) / 20))

    def mix(self, source, noise, snr):
        source_rms = self.rms(source)
        noise_rms = self.rms(noise)

        adj_noise_rms = self.adjusted_rms(source_rms, snr)
        adj_noise = noise * (adj_noise_rms / noise_rms)

        return source+adj_noise # Not need de-clipping, because of float values 
    
    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.df.iloc[idx]
        source_path = row['source']
        if self.seg_len is not None:
            start = np.random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = -1
        assert os.path.exists(source_path)
        source, sr = torchaudio.load(source_path)
        source = source[:, start:stop]
        if self.padding_value > 0:
            source = self.get_padded_value(source)
            
        reverb_source = None
        if self.source_reverb:
            reverb_source = self.source_reverb(source)[0]

        source_speaker = int(row['index'])
        filtered = self.enroll_df.query('index == @source_speaker')
        assert len(filtered) > 0
        enroll_path = filtered.iloc[np.random.randint(0, len(filtered))]['source']
        enroll, sr = torchaudio.load(enroll_path)

        noise_row = self.noise_df.iloc[np.random.randint(0, len(self.noise_df))]
        noise_path = noise_row['noise']
        if self.seg_len is not None:
            start = np.random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = np.random.randint(0, row["length"] - source.shape[-1])
            stop = start + source.shape[-1]
        if os.path.exists(noise_path):
            noise, sr = torchaudio.load(noise_path)
            noise = noise[:, start:stop]
        else:
            noise=None
        reverb_noise = None
        if self.noise_reverb:
            reverb_noise = self.noise_reverb(noise)[0]

        # mixing
        snr = np.random.rand() * (self.max_snr-self.min_snr) + self.min_snr
        if reverb_source is not None:
            mixture = self.mix(reverb_source, reverb_noise, snr)
        else:
            mixture = self.mix(source, noise, snr)

        # opus encode/decode
        if self.opus:
            mixture = self.opus(mixture)[0]
        # normalize
        std, mean = torch.std_mean(mixture, dim=-1)
        mixture = (mixture - mean)/std
        std, mean = torch.std_mean(source, dim=-1)
        source = (source - mean)/std
        std, mean = torch.std_mean(enroll, dim=-1)
        enroll = (enroll - mean)/std

        return torch.t(mixture), torch.t(source), torch.t(enroll), source_speaker
    
def data_processing(data:Tuple[Tensor,Tensor,Tensor,Tensor]) -> Tuple[Tensor, Tensor, Tensor, List, List]:
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

    dataset = SpeechDatasetOTFMix(csv_path, noise_csv_path, enroll_csv_path, 
                 config['augment']['mixing'], 
                 config['augment'])
    mixture, source, enroll, speaker = dataset.__getitem__(10)

class SpeechDatasetCTC(SpeechDataset):
    def __init__(self, csv_path:str, enroll_path:str, sample_rate=16000, segment=0, enroll_segment=0, padding_value=0, tokenizer=None):
        super.__init__(csv_path, enroll_path, sample_rate, segment, enroll_segment, padding_value)
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx:int):
        mixture, source, enroll, source_speaker = super.__getitem__(idx)
        row = self.df.iloc[idx]
        label_path = row['label']
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            label = self.tokenizer.text2token(line) # List[int]

        return mixture, source, enroll, source_speaker, label

def data_processing_ctc(data:Tuple[Tensor,Tensor,Tensor,int, List]) -> Tuple[Tensor, Tensor, Tensor, list, list, Tensor, list]:
    mixtures = []
    sources = []
    enrolls = []
    lengths = []
    speakers = []
    labels = []
    label_legnths = []

    for mixture, source, enroll, speaker in data:
        # w/o channel
        mixtures.append(mixture)
        if source is not None:
            sources.append(source)
        enrolls.append(enroll)
        lengths.append(len(mixture))
        #speakers.append(torch.from_numpy(speaker.astype(np.int)).clone())
        speakers.append(speaker)

        label_lengths.append(len(label))
        labels.append(torch.from_numpy(label))
        
    mixtures = nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
    if len(sources) > 0:
        sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)
    enrolls = nn.utils.rnn.pad_sequence(enrolls, batch_first=True)
    speakers = torch.from_numpy(np.array(speakers)).clone()

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    mixtures = mixtures.squeeze()
    sources = sources.squeeze()
    enrolls = enrolls.squeeze()

    if mixtures.dim() == 1:
        mixtures = mixtures.unsqueeze(0)
        sources = sources.unsqueeze(0)
        enrolls = enrolls.unsqueeze(0)
        
    return mixtures, sources, enrolls, lengths, speakers, labels, label_lengths
