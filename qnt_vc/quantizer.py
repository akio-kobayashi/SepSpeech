import torch
import torchaudio
from lm.qnt_encode import EncodecQuantizer
import pandas as pd
from argparse import ArgumentParser
import os, sys

def main(args):
    encoder = EncodecQuantizer()
    df = pd.read_csv(args.input_csv)
    qnts=[]
    for index, row in df.iterrows():
        path=row['source']
        qnt_data = encoder.encode_from_file(path)
        path = os.path.join(args.output_root, os.path.splitext(os.path.basename(row['source']))[0] + '.pt')
        qnts.append(path)
        torch.save(qnt_data.cpu(), path)

    df['source'] = qnts
    df.to_csv(args.output_csv)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--output_root', type=str, default='./')
    args=parser.parse_args()

    main(args)
