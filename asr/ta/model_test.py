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

'''
 PyTorch Lightning用 将来変更する予定
'''
def main(config:dict, checkpoint_path=None):

    model = LitASR(config)
    for n in range(680, 700):
        input=torch.ones([1, 1, n, 80])
        print(input.shape)
        output = model.model.cntf(input)
        print(output.shape)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config)
