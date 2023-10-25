import os, sys
import torch
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from lite.radio_solver import LitDenoiser
from argparse import ArgumentParser
import yaml
import pandas as pd
from einops import rearrange

def main(config, args):

    model = LitDenoiser.load_from_checkpoint(args.checkpoint, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(args.input_csv)
    df['estimate'] = ''
    for index, row in df.iterrows():
        noisy, sr = torchaudio.load(row['noisy'])
        std, mean = torch.std_mean(noisy)
        noisy = (noisy - mean)/std
        
        with torch.no_grad():
            estimate = model(noisy.to(device))
            
        name = os.path.splitext(os.path.basename(row['noisy']))[0].replace('_noisy', '_estimate') + '.wav'
        path = os.path.join(args.output_root, name)

        df.loc[df['key']==row['key'], 'estimate'] = path
        torchaudio.save(filepath=path, src=estimate.to('cpu'), sample_rate=sr, bits_per_sample=16, encoding='PCM_S')
    df.to_csv(args.output_csv, index=False)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--output_root', type=str, default='./')
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    config = config['config']
    main(config, args)
