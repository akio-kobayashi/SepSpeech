import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from lite.solver import LitSepSpeaker
import torch.utils.data as dat
import conventional
from conventional.speech_dataset import SpeechDataset, SpeechDatasetCTC
from lite.asr_tokenizer import ASRTokenizer
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

'''
 PyTorch Lightning用 将来変更する予定
'''
def main(config:dict, checkpoint_path=None):

#    if checkpoint_path is not None:
#        model = LitSepSpeaker.load_from_checkpoint(checkpoint_path, config=config)
#    else:
    model = LitSepSpeaker(config)

    '''
    padding_value = 0
    if config['model_type'] == 'tasnet':
        padding_value = model.get_padding_value()
    '''
    
    if config['ctc']['use']:
        tokenizer = ASRTokenizer(config['dataset']['tokenizer'])
        
    if config['ctc']['use'] is False:
        train_dataset = SpeechDataset(csv_path=config['dataset']['train']['csv_path'],
                                      enroll_path=config['dataset']['train']['enroll_csv_path'],
                                      sample_rate=config['dataset']['segment']['sample_rate'],
                                      segment=config['dataset']['segment']['segment'],
                                      enroll_segment=config['dataset']['segment']['enroll_segment'])
        train_loader = data.DataLoader(dataset=train_dataset,
                                       **config['dataset']['process'],
                                       pin_memory=True,
                                       shuffle=True, 
                                       collate_fn=lambda x: conventional.speech_dataset.data_processing(x))
    else:
        train_dataset = SpeechDatasetCTC(csv_path=config['dataset']['train']['csv_path'],
                                         enroll_path=config['dataset']['train']['enroll_csv_path'],
                                         sample_rate=config['dataset']['segment']['sample_rate'],
                                         segment=config['dataset']['segment']['segment'],
                                         enroll_segment=config['dataset']['segment']['enroll_segment'])
        train_loader = data.DataLoader(dataset=train_dataset,
                                       **config['dataset']['process'],
                                       pin_memory=True,
                                       shuffle=True, 
                                       collate_fn=lambda x: conventional.speech_dataset.data_processing_ctc(x))
    if config['ctc']['use'] is False:
        valid_dataset = SpeechDataset(csv_path=config['dataset']['valid']['csv_path'],
                                      enroll_path=config['dataset']['valid']['enroll_csv_path'],
                                      sample_rate=config['dataset']['segment']['sample_rate'],
                                      segment=config['dataset']['segment']['segment'],
                                      enroll_segment=config['dataset']['segment']['enroll_segment'])
        valid_loader = data.DataLoader(dataset=valid_dataset,
                                       **config['dataset']['process'],
                                       pin_memory=True,
                                       shuffle=False, 
                                       collate_fn=lambda x: conventional.speech_dataset.data_processing(x))
    else:
        valid_dataset = SpeechDataset(csv_path=config['dataset']['valid']['csv_path'],
                                      enroll_path=config['dataset']['valid']['enroll_csv_path'],
                                      sample_rate=config['dataset']['segment']['sample_rate'],
                                      segment=config['dataset']['segment']['segment'],
                                      enroll_segment=config['dataset']['segment']['enroll_segment'],
                                      tokenizer=tokenizer)
        valid_loader = data.DataLoader(dataset=valid_dataset,
                                       **config['dataset']['process'],
                                       pin_memory=True,
                                       shuffle=False, 
                                       collate_fn=lambda x: conventional.speech_dataset.data_processing_ctc(x))
        
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
