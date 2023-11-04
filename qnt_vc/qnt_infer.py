import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
import torch.utils.data as dat
from qnt_vc.qnt_solver import LitVoiceConversion
import qnt_vc.qnt_generator as G
from argparse import ArgumentParser
import yaml, json
import warnings
import pandas as pd
from einops import rearrange
import os, sys

warnings.filterwarnings('ignore')

def main(args):

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    with open(args.speakers, 'r') as f:
        speaker2id = json.load(f)
    
    target_speaker = speaker2id[args.target_speaker]

    model = LitVoiceConversion.load_from_checkpoint(args.checkpoint, config)
    df = pd.read_csv(args.input_csv)

    '''
        shape: torch.Size([channels, classes, lengths])
        dtype: torch.int64
    '''
    for index, row in df.iterrows():
        source_speaker = speaker2id[row['speaker']]
        source = torch.load(row['source'])
        source = rearrange(source, '(b f) c t -> b c t f', b=1)
        output = model.model.greedy_decode(source, source_speaker, target_speaker)
        #output = rearrange(output, 'b c t -> (b c) t f', b=1)

        path = os.path.join(args.output_dir, row['speaker']+'_'+args.target_speaker+_+row['utterance']+'.pt')
        torch.save(path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--target_speaker', type=str, required=True)
    parser.add_argument('--speakers', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gpus', nargs='*', type=int)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')

    main(args)
