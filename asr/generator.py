import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import pandas as pd
import torchaudio
from asr_tokenizer import ASRTokenizer
import yaml

def compute_global_mean_std(csv_path, config):
    wav2spec = torchaudio.transforms.Spectrogram(
        n_fft=config['nfft'],
        win_length=config['win_length'],
        hop_length=config['hop_length'],
        window_fn=torch.hamming_window
    )
    spec2mel = torchaudio.transforms.MelScale(
        n_mels=config['n_mels'],
        sample_rate=config['sample_rate'],
        n_stft=config['nfft']//2+1
    )
    eps = 1.e-8

    num_frames = 0
    sum, sq_sum = torch.zeros((1, config['n_mels'])), torch.zeros((1, config['n_mels']))
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        source_path, label_path, key = row['source'], row['label'], row['key']
        source, _ = torchaudio.load(source_path, normalize=True)
        spec = wav2spec(source)
        melspec = torch.log(spec2mel(spec)+eps) # (1, n_mels, time)
        melspec = torch.t(melspec.squeeze()) # (time, n_mels)

        sum += torch.sum(melspec, 0)
        sq_sum += torch.sum(melspec*melspec, 0)
        num_frames += melspec.shape[0]
        
    mean = sum/num_frames
    sq_mean = sq_sum/num_frames
    std = sq_mean - mean*mean

    std = torch.sqrt(std)

    mean = mean.to('cpu').detach().numpy().copy()
    std = std.to('cpu').detach().numpy().copy()

    outpath = config['global_mean_std']
    np.savez(outpath, mean=mean, std=std)

    return mean, std

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, path, config:dict, segment=0, tokenizer=None, specaug=False):
        super(SpeechDataset, self).__init__()

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

        self.wav2spec = torchaudio.transforms.Spectrogram(
            n_fft=config['analysis']['nfft'],
            win_length=config['analysis']['win_length'],
            hop_length=config['analysis']['hop_length'],
            window_fn=torch.hamming_window
        )
        self.spec2mel = torchaudio.transforms.MelScale(
            n_mels=config['analysis']['n_mels'],
            sample_rate=self.sample_rate,
            n_stft=config['analysis']['nfft']//2+1
        )

        npz=np.load(config['analysis']['global_mean_std'])
        self.mean, self.std = npz['mean'], npz['std']
        
        self.eps = 1.e-8
        #self.specaug = True if config['augment']['specaug'] else False
        self.specaug = specaug
        self.wav2spec_complex=None
        if self.specaug:
            self.time_stretch = torchaudio.transforms.TimeStretch(n_freq=config['analysis']['nfft']//2+1)
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=config['augment']['freq_mask'])
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=config['augment']['time_mask'])
            self.wav2spec_complex = torchaudio.transforms.Spectrogram(
                n_fft=config['analysis']['nfft'],
                win_length=config['analysis']['win_length'],
                hop_length=config['analysis']['hop_length'],
                window_fn=torch.hamming_window,
                power=None
            )

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
        if self.specaug:
            spec = self.wav2spec_complex(source)
            rnd = np.random.rand() * 2/10.0 + 0.9 # 0.9 -- 1.1
            spec = self.time_stretch(spec, rnd)
            spec = torch.square(torch.abs(spec))
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)
        else:
            spec = self.wav2spec(source)
            
        melspec = torch.log(self.spec2mel(spec)+self.eps) # (1, n_mels, time)
        melspec = torch.t(melspec.squeeze()) # (time, n_mels)
        melspec = (melspec - self.mean)/self.std
        
        pattern = re.compile('\s\s+')
        label_path = row['label']
        label = None
        assert os.path.exists(label_path) 
        with open(label_path, 'r') as  f:
            line = f.readline()
            line = re.sub(pattern, ' ', line.strip())
            label = self.tokenizer.text2token(line)
            label = torch.tensor(label, dtype=torch.int32)
            #label = torch.tensor(label)
        assert label is not None and len(label) > 0

        return melspec, label, row['key']

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
        #inputs.append(torch.from_numpy(input.astype(np.float32)).clone())
        inputs.append(input)
        #labels.append(torch.from_numpy(label.astype(np.int)).clone())
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
    
