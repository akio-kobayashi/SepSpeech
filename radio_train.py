import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from lite.radio_solver import LitDenoiser
import torch.utils.data as dat
import conventional.radio_dataset as rd
from conventional.radio_dataset import RadioDataset
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

'''
 PyTorch Lightning用 将来変更する予定
'''
def main(config:dict, checkpoint_path=None, dict_path=None):

    if checkpoint_path is not None:
        model = LitDenoiser.load_from_checkpoint(checkpoint_path, config=config)
    elif dict_path is not None:
        model.model.to('cpu')
        model.model.load_dict(torch.load(dict_path), map_location=torch.device('cpu'))
        model.model.to('gpu')
    else:
        model = LitDenoiser(config)

    divisor = rd.get_divisor(model)
    
    train_dataset = RadioDataset(config['dataset']['train']['csv_path'],
                                 config,
                                 segment=config['dataset']['segment']['segment'],
                                 divisor=divisor
                                 )
    train_loader = data.DataLoader(dataset=train_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=True, 
                                   collate_fn=lambda x: rd.data_processing(x))
    valid_dataset = RadioDataset(config['dataset']['valid']['csv_path'],
                                 config,
                                 segment=config['dataset']['segment']['segment'],
                                 divisor=divisor
                                 )
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=False, 
                                   collate_fn=lambda x: rd.data_processing(x))
    callbacks = [
        pl.callbacks.ModelCheckpoint( **config['checkpoint'])
    ]
    logger = TensorBoardLogger(**config['logger'])
    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          devices=args.gpus,
                          **config['trainer'] )
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=valid_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dict_path', type=str, default=None)
    parser.add_argument('--gpus', nargs='*', type=int)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    if 'config' in config.keys():
        config = config['config']
    main(config, args.checkpoint, args.dict_path)
