import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from text_generator import TextDataset
import text_generator
#from model import Transducer
from trans import TransTransducer
from metric import IterMeter
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
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    tokenizer = ASRTokenizer(config['dataset']['tokenizer'], ctc_decode=True)
    output_tokenizer = ASRTokenizer(config['dataset']['output_tokenizer'], ctc_decode=True)
    
    writer = SummaryWriter(log_dir=config['logger']['save_dir'])
    if not os.path.exists(config['logger']['save_dir']):
        os.makedirs(config['logger']['save_dir'])
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda is True:
        print('use GPU')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    shuffle = False if config['analysis']['sort_by_len'] is True else False
    
    train_dataset=TextDataset(config['dataset']['train']['csv_path'], config, tokenizer, output_tokenizer, upsample=config['transformer']['upsample'])
    train_loader =data.DataLoader(dataset=train_dataset,
                                  batch_size=config['dataset']['process']['batch_size'],
                                  shuffle=shuffle,
                                  collate_fn=lambda x: text_generator.data_processing(x,'train'),
                                  **kwargs)
    valid_dataset=TextDataset(config['dataset']['valid']['csv_path'], config, tokenizer, output_tokenizer, upsample=config['transformer']['upsample'])
    valid_loader=data.DataLoader(dataset=valid_dataset,
                                 batch_size=config['dataset']['process']['batch_size'],
                                 shuffle=shuffle,
                                 collate_fn=lambda x: text_generator.data_processing(x, 'valid'))
    eval_dataset=TextDataset(config['dataset']['test']['csv_path'], config, tokenizer, output_tokenizer)
    eval_loader=data.DataLoader(dataset=eval_dataset,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=lambda x: text_generator.data_processing(x, 'eval'))

    # input_size, vocab_size, hidden_size, num_layers,
    # dropout=.5, blank=0, bidirectional=False
    network=TransTransducer(device,
                            **config['transformer']
                            )
    #print(network)
    print('Number of Parameters: ', sum([param.nelement() for param in network.parameters()]))

    if args.pretrained is not None:
        network.load_state_dict(torch.load(args.pretrained, map_location=torch.device('cpu')))
    network = network.to(device)

    optimizer=optim.RAdam(network.parameters(), config['optimizer']['lr'])
    if config['max_epochs'] == 0:
        epochs=1
    else:
        epochs=config['max_epochs']

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['optimizer']['lr'],
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=epochs,
                                              anneal_strategy='linear')
    model_dir = os.path.join(config['logger']['save_dir'], 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    path=os.path.join(config['logger']['save_dir'], 'hparams.yaml')
    with open(path, 'w') as f:
        yaml.dump(config, f)
        
    iter_meter=IterMeter()
    min_cer=100.0
    for epoch in range(1, config['max_epochs']+1):
        solver_trans.train(network, device, train_loader, optimizer, scheduler,
                           epoch, iter_meter, writer)
        avg_cer = solver_trans.test(network, device, valid_loader, epoch, iter_meter, writer)
        if avg_cer < min_cer:
            min_cer = avg_cer
            print(f'Minimum CER changed to {min_cer:.3f}')
            print('Saving model to %s' % config['model_output'])
            path = os.path.join(model_dir, config['model_output'])
            torch.save(network.to('cpu').state_dict(), path)
            network.to(device)
        path=f'checkpoint_epoch={epoch}_cer={avg_cer:.3f}'
        path=os.path.join(model_dir, path)
        torch.save(network.to('cpu').state_dict(), path)
        network.to(device)

    path = os.path.join(model_dir, config['model_output'])
    network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    network.to(device)
    path = os.path.join(model_dir, config['decode_output'])

    with torch.no_grad():
        cer = solver_trans.decode(network, device, eval_loader, tokenizer, output_tokenizer, path)

if __name__ == "__main__":
    
    main()
