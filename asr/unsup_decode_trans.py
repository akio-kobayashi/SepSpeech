import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from generator import SpeechDataset
#import generator
from text_generator import TextDataset
import text_generator
from trans import TransTransducer
#from model import Transducer
from metric import IterMeter
#import solver
import solver_trans
import numpy as np
import argparse
from asr_tokenizer import ASRTokenizer
import yaml
import warnings
from einops import rearrange

warnings.simplefilter('ignore')
os.environ['TOKENIZERS_PARALLELISM']='true'

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
    output_tokenizer = ASRTokenizer(config['dataset']['output_tokenizer'], ctc_decode=True)
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda is True:
        print('use GPU')

    network=TransTransducer(device,
                            **config['transformer']
                            )
    print('Number of Parameters: ', sum([param.nelement() for param in network.parameters()]))

    network.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    network = network.to(device)

    df = pd.read_csv(args.input_csv)
    texts, logps = [], []

    with torch.no_grad():
        for index, row in df.iterrows():
            label_path = row['input_label']
            line = f.readline()
            label = tokenizer.text2token(line.strip())
            label = torch.tensor(label, dtype=torch.int32)

            
            decode, [logp, logp_seq] = network.greedy_decode(label.to(device))
                
            text, lop_seq = output_tokenizer.token2text_values(decode, logp_seq)
            logp = ' '.join([ str(lp) for lp in logp_seq ])
            texts.append(text)
            logps.append(logp)

    df['logp'] = logps
    df['decode'] = texts
            
    df.to_csv(args.output_csv)

if __name__ == "__main__":
    
    main()
