import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from solver import LitASR
import torch.utils.data as data
import speech_dataset
from speech_dataset import SpeechDataset
from asr_tokenizer import ASRTokenizer
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

'''
 PyTorch Lightning用 将来変更する予定
'''
def main(config:dict, checkpoint_path:str):

    assert checkpoint_path is not None:

    tokenizer = ASRTokenizer(config['tokenizer'], config['max_length'])
    model = LitASR.load_from_checkpoint(checkpoint_path, config=config, tokenizer=tokenizer)
    model.model.set_special_tokens(tokenizer.text2token(config['bos'])[0], tokenizer.text2token(config['eos'])[0])

    test_dataset = SpeechDataset(**config['dataset']['test'], 
                                  **config['dataset']['segment'],
                                  tokenizer=tokenizer)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  num_workers=2,
                                  pin_memory=True,
                                  shuffle=False, 
                                  collate_fn=lambda x: speech_dataset.data_processing(x))
    model.eval()
    with open(config['decode_output'], 'w') as f:
        for batch_idx, batch in enumerate(test_loader):
            inputs, _, input_lengths, _, keys = batch
            output = LitASR.decode(inputs, input_lengths[0])

            output = tokenizer.token2text(output) # w/o special tokens
            # split text for CER computation
            output = ' '.join(list(output))
            f.write(f'{output} ({keys[0]})\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config, args.checkpoint)
