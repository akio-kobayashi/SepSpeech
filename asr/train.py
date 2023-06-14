import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from solver import LitASR
import torch.utils.data as data
import speech_dataset
from speech_dataset import SpeechDataset
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


'''
 PyTorch Lightning用 将来変更する予定
'''
def main(config:dict, args):

    if args.checkpoint is not None:
        model = LitASR.load_from_checkpoint(args.checkpoint, config=config)
    else:
        model = LitASR(config)
        
    train_dataset = SpeechDataset(config['dataset']['train']['csv_path'], 
                                  config, 
                                  config['dataset']['segment']['segment'],
                                  tokenizer=None) 
    train_loader = data.DataLoader(dataset=train_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=True, 
                                   collate_fn=lambda x: speech_dataset.data_processing(x))
    valid_dataset = SpeechDataset(config['dataset']['valid']['csv_path'],
                                  config,
                                  config['dataset']['segment']['segment'],
                                  tokenizer=None)
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=False, 
                                   collate_fn=lambda x: speech_dataset.data_processing(x))
    callbacks = [
        pl.callbacks.ModelCheckpoint( **config['checkpoint']),
        EarlyStopping(monitor='valid_loss', patience=10, mode='min')
    ]
    logger = TensorBoardLogger(**config['logger'])
    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          **config['trainer'] )
    # find inital learning rate
    if args.lr_find:
        tuner = pl.tuner.Tuner(trainer)
        lr_find_results = tuner.lr_find(
            model,
            train_dataloaders=train_loader,
            min_lr=1.e-5,
            max_lr=1.e-3
        )
        new_lr = lr_find_results.suggestion(skip_begin=20, skip_end=20)
        print("initial learning rate: %.3f" % new_lr)
        model.hparams.lr = new_lr

    # start training
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=valid_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--lr_find', action='store_true')
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config, args)
