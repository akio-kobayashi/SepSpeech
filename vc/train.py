import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from vc.solver import LitVoiceConversion
import torch.utils.data as dat
import vc
from vc.speech_dataset import SpeechDataset
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

def main(config:dict, checkpoint_path=None):

    model = LitVoiceConversion(config)
            
    train_dataset = SpeechDataset(source_path=config['dataset']['train']['source_path'],
                                    target_path=config['dataset']['train']['target_path'])
    
    train_loader = data.DataLoader(dataset=train_dataset,
                                    **config['dataset']['process'],
                                    pin_memory=True,
                                    shuffle=True, 
                                    collate_fn=lambda x: vc.speech_dataset.data_processing(x))

    valid_dataset = SpeechDataset(source_path=config['dataset']['valid']['source_path'],
                                    target_path=config['dataset']['valid']['target_path'])
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                     **config['dataset']['process'],
                                     pin_memory=True,
                                     shuffle=False, 
                                     collate_fn=lambda x: vc.speech_dataset.data_processing(x))
        
    callbacks = [
        pl.callbacks.ModelCheckpoint( **config['checkpoint'])
    ]
    logger = TensorBoardLogger(**config['logger'])

    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          devices=args.gpus,
                          **config['trainer'] )

    trainer.fit(model=model, ckpt_path=args.checkpoint, train_dataloaders=train_loader,
                val_dataloaders=valid_loader)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpus', nargs='*', type=int)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config, args.checkpoint)
