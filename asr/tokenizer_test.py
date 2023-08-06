import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from text_generator import TextDataset
#import text_generator
#from model import Transducer
#from trans import TransTransducer
#from metric import IterMeter
#import solver_trans
import numpy as np
import argparse
from asr_tokenizer import ASRTokenizer
import yaml
import warnings
import pandas as pd
import re

warnings.simplefilter('ignore')
os.environ['TOKENIZERS_PARALLELISM']='true'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    args = parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    tokenizer = ASRTokenizer(config['dataset']['tokenizer'], ctc_decode=True)
    output_tokenizer = ASRTokenizer(config['dataset']['output_tokenizer'], ctc_decode=True)
    
    df_valid = pd.read_csv(config['dataset']['valid']['csv_path'])

    pattern=re.compile(r'\s\s+')
    for index, row in df_valid.iterrows():
        input_label = row['input_label']
        output_label = row['output_label']
        with open(row['input_label'], 'r') as  f:
            line = f.readline()
            line = re.sub(pattern, ' ', line.strip())
            label = tokenizer.text2token(line)
            label = tokenizer.token2text(label)
            print(label)
        with open(row['output_label'], 'r') as  f:
            line = f.readline()
            line = re.sub(pattern, ' ', line.strip())
            label = output_tokenizer.text2token(line)
            label = output_tokenizer.token2text(label)
            print(label)
            
if __name__ == "__main__":
    
    main()
