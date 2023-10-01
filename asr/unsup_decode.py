import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from generator import SpeechDataset
#import generator
from model import Transducer
from metric import IterMeter
import solver
import numpy as np
import argparse
from asr_tokenizer import ASRTokenizer
import yaml
import warnings
import pandas as pd
import torchaudio
from einops import rearrange

warnings.simplefilter('ignore')
os.environ['TOKENIZERS_PARALLELISM']='true'

class Wave2MelSpec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sample_rate = 16000
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

    def forward(self, path):
        source, _ = torchaudio.load(path, normalize=True)
        spec = self.wav2spec(source)
        melspec = torch.log(self.spec2mel(spec)+self.eps) # (1, n_mels, time)
        #melspec = torch.t(melspec.squeeze()) # (time, n_mels)
        melspec = rearrange(melspec, 'b c t -> b t c')
        melspec = (melspec - self.mean)/self.std
        return melspec
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    
    args = parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    tokenizer = ASRTokenizer(config['dataset']['tokenizer'], ctc_decode=True)
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda is True:
        print('use GPU')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    network=Transducer(device,
                       **config['transducer']
                       )
    print('Number of Parameters: ', sum([param.nelement() for param in network.parameters()]))

    network.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    network = network.to(device)

    df = pd.read_csv(args.input_csv)
    texts, logps = [], []
    feature_extractor = Wave2MelSpec(config)

    with torch.no_grad():
        for index, row in df.iterrows():
            x = feature_extractor(row['source'])
            decode, logp = network.greedy_decode(x.to(device))
            text = tokenizer.token2text_raw(decode)
            texts.append(text)
            logps.append(logp)
    df['logp'] = logps
    df['decode'] = texts
            
    df.to_csv(args.output_csv)

if __name__ == "__main__":
    
    main()
