import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
#from loss.stft_loss import stft
from einops import rearrange

class MFCCLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mfcc = T.MFCC(
            sample_rate = config['sample_rate'],
            n_mfcc = config['n_mfcc'],
            log_mels = True,
            melkwargs={"n_fft": config['n_fft'],
                       "hop_length": config['hop_length'],
                       "n_mels": config['n_mels'],
                       "center": True}
        )
        self.loss = nn.L1Loss(reduction='sum')
        self.n_fft = config['n_fft']
        self.hop_lengths = config['hop_lengths']
        
    def forward(self, preds, targets, lengths):
        # (b, t) -> (b f t)
        preds = self.mfcc(preds)
        targets = self.mfcc(targets)
        
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            #length = 1 + (lengths[b]-self.n_fft)//self.hop_length
            length = 1 + lengths[b]//self.hop_length # center = True
            mask[b, :, :length] = 1.
        return self.loss(preds * mask, targets * mask) / torch.sum(mask)

class LFCCLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lfcc = T.LFCC(
            sample_rate = config['sample_rate'],
            n_filter = config['n_filter'],
            n_lfcc = config['n_lfcc'],
            log_lf = True,
            speckwargs={"n_fft": config['n_fft'],
                        "hop_length": config['hop_length'],
                        "center": False}
        )
        self.loss = nn.L1Loss(reduction='sum')
        
    def forward(self, preds, targets, lengths):
        # (b, t) -> (b f t)
        preds = self.lfcc(preds)
        targets = self.lfcc(targets)
        
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            length = 1 + (lengths[b]-self.n_fft)//self.hop_length
            mask[b, :, :length] = 1.
        return self.loss(preds * mask, targets * mask) / torch.sum(mask)
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml
    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['config']['loss']['mfcc']['hop_length']=200
    config['config']['loss']['mfcc']['n_mels']=20
    loss=MFCCLoss(config['config']['loss']['mfcc'])
    
    source,_=torchaudio.load(args.source)
    std, mean = torch.std_mean(source)
    source = (source-mean)/std
    target,_=torchaudio.load(args.target)
    std, mean = torch.std_mean(target)
    target = (target-mean)/std
    lengths = [source.shape[-1]]
    _loss = loss(source, target, lengths)
    
    print(_loss)
