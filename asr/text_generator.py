import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import pandas as pd
import torchaudio
from asr_tokenizer import ASRTokenizer
import yaml


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, path, config:dict, tokenizer=None, output_tokenizer=None, upsample=4):
        super().__init__()

        self.upsample = upsample
        #pattern = re.compile('\s\s+')
        self.pattern = re.compile(r'⠲$')
        
        self.df = pd.read_csv(path)

        self.check_source_target_lengths()
        
        if config['analysis']['sort_by_len']:
            self.df = self.df.sort_values('input_length')
        self.max_length = config['dataset']['max_length']
        
        self.tokenizer = tokenizer
        if self.tokenizer == None:
            self.tokenizer = ASRTokenizer(config['dataset']['tokenizer'], config['dataset']['max_length'])
        self.output_tokenizer = output_tokenizer
        if self.output_tokenizer == None:
            self.output_tokenizer = ASRTokenizer(config['dataset']['output_tokenizer'],
                                                 config['dataset']['output_max_length'])
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        
        #pattern = re.compile('\s\s+')
        #pattern = re.compile(r'⠲$')
        
        label_path = row['input_label']
        label = None
        assert os.path.exists(label_path) 
        with open(label_path, 'r') as  f:
            line = f.readline()
            #line = re.sub(pattern, ' ', line.strip())
            label = self.tokenizer.text2token(line.strip())
            label = torch.tensor(label, dtype=torch.int32)
        assert label is not None and len(label) > 0

        output_label_path = row['output_label']
        output_label = None
        assert os.path.exists(output_label_path)
        with open(output_label_path, 'r') as f:
            line = f.readline()
            output_label = re.sub(self.pattern, '', line.strip())
            output_label = self.output_tokenizer.text2token(output_label)
            output_label = torch.tensor(output_label, dtype=torch.int32)
        assert output_label is not None and len(output_label) > 0

        return label, output_label, row['key']

    def check_source_target_lengths(self):
        max_len = len(self.df)
        remove_indices = []
        for index, row in self.df.iterrows():
            input_length = output_length = 0
            with open(row['input_label'], 'r') as  f:
                line = f.readline().strip()
                input_length = len(list(line))
                
            with open(row['output_label'], 'r') as f:
                line = f.readline().strip()
                line = re.sub(self.pattern, '', line)
                output_length = len(list(line))

            if input_length == 0 or output_length == 0:
                remove_indices.append(index)
            if self.upsample * input_length < output_length:
                remove_indices.append(index)
                
        self.df.drop(index=remove_indices, inplace=True)
        print(
                f"Drop {len(remove_indices)} utterances from {max_len} "
        )

'''
    data_processing
    Return labels, output_labels, input_lengths, output_lengths
'''
def data_processing(data, data_type="train"):
    labels = []
    output_labels = []
    input_lengths=[]
    output_lengths=[]
    keys = []

    for label, output_label, key in data:
        """ inputs : (batch, time, feature) """
        # w/o channel
        labels.append(label)
        output_labels.append(output_label)
        
        input_lengths.append(len(label))
        output_lengths.append(len(output_label))
        keys.append(key)

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    output_labels = nn.utils.rnn.pad_sequence(output_labels, batch_first=True)

    return labels, output_labels, input_lengths, output_lengths, keys


if __name__=="__main__":

    with open('config.yaml', 'r') as yf:
        config =  yaml.safe_load(yf)
        
    tokenizer=ASRTokenizer(config['dataset']['tokenizer'])
    tokenizer=ASRTokenizer(config['dataset']['output_tokenizer'])
    
    dataset=TextDataset(config['dataset']['train']['csv_path'],
                        config,
                        tokenizer,
                        output_tokenizer
                        )
    
