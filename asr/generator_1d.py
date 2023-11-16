import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import pandas as pd
import torchaudio
from einops import rearrange
from asr_tokenizer import ASRTokenizer
import yaml

class SpeechDataset1D(torch.utils.data.Dataset):

    def __init__(self, path, config:dict, segment=0, tokenizer=None):
        super(SpeechDataset1D, self).__init__()

        self.df = pd.read_csv(path)
        
        if config['analysis']['sort_by_len']:
            self.df = self.df.sort_values('length')
        self.segment = segment
        self.sample_rate = config['analysis']['sample_rate']
        if self.segment > 0:
            max_len = len(self.df)
            max_segment_length = int(self.segment * self.sample_rate)
            self.df = self.df[self.df['length'] <= max_segment_length]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )

        self.eps = 1.e-8
        self.tokenizer = tokenizer
        if self.tokenizer == None:
            self.tokenizer = ASRTokenizer(config['dataset']['tokenizer'], config['dataset']['max_length'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        source_path = row['source']
        assert os.path.exists(source_path)

        source, _ = torchaudio.load(source_path, normalize=True)
        std, mean = torch.std_mean(source)
        source = (source - mean)/std
        source = rearrange(source, 'c t -> t c')
        
        pattern = re.compile('\s\s+')
        label_path = row['label']
        label = None
        assert os.path.exists(label_path) 
        with open(label_path, 'r') as  f:
            line = f.readline()
            line = re.sub(pattern, ' ', line.strip())
            label = self.tokenizer.text2token(line)
            label = torch.tensor(label, dtype=torch.int32)
        assert label is not None and len(label) > 0

        return source, label, row['key']

'''
    data_processing
    Return inputs, labels, input_lengths, label_lengths, outputs
'''
def data_processing(data, data_type="train"):
    inputs = []
    labels = []
    input_lengths=[]
    label_lengths=[]
    keys = []

    for input, label, key in data:
        """ inputs : (batch, time, feature) """
        # w/o channel
        inputs.append(input)
        labels.append(label)
        input_lengths.append(input.shape[0])
        label_lengths.append(len(label))
        keys.append(key)

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return inputs, labels, input_lengths, label_lengths, keys

if __name__=="__main__":

    with open('config.yaml', 'r') as yf:
        config =  yaml.safe_load(yf)
        
    tokenizer=ASRTokenizer(config['dataset']['tokenizer'])
    dataset=SpeechDataset(config['dataset']['train']['csv_path'],
                          config,
                          20,
                          tokenizer
                          )
    
