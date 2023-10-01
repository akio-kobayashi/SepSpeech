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

warnings.simplefilter('ignore')
os.environ['TOKENIZERS_PARALLELISM']='true'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    
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

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    eval_dataset=TextDataset(config['dataset']['test']['csv_path'], config, tokenizer, output_tokenizer)
    eval_loader=data.DataLoader(dataset=eval_dataset,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=lambda x: text_generator.data_processing(x, 'eval'))
    network=TransTransducer(device,
                            **config['transformer']
                            )
    print('Number of Parameters: ', sum([param.nelement() for param in network.parameters()]))

    if args.model is not None:
        network.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    network = network.to(device)

    #model_dir = os.path.join(config['logger']['save_dir'], 'models')
    #path = os.path.join(model_dir, config['decode_output'])
    with torch.no_grad():
        #cer = solver_trans.decode(network, device, eval_loader, tokenizer, args.output, beam_search=True)
        cer = solver_trans.decode(network, device, eval_loader, tokenizer, output_tokenizer, args.output, beam_search=True)

if __name__ == "__main__":
    
    main()
