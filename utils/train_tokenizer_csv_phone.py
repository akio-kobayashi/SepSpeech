import sys,os
from argparse import ArgumentParser
from typing import Iterator
import re
import pandas as pd
import json
    
def main(args):
    blank_token = "<blk>"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"

    df = {}
    df[blank_token] = 0
    df[bos_token] = 1
    df[eos_token] = 2
    df[unk_token] = 3
    token_id = 4
    
    if args.text is not None:
        for path in args.txt:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    for phn in line.strip().split():
                        if phn not in df.keys():
                            df[phn] = token_id
                            token_id += 1
    else:
        for path in args.csv:
            f = pd.read_csv(path)
            for idx, row in f.iterrows():
                line = row['text']
                for phn in line.strip().split():
                        if phn not in df.keys():
                            df[phn] = token_id
                            token_id += 1
    os.makedirs(args.pretrained)
    outpath = os.path.join(args.pretrained, 'vocab.json')
    with open(outpath, mode="wt", encoding="utf-8") as f:
        json.dump(df, f, ensure_ascii=False, indent=2)
            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text', type=str, nargs='*', default=None)
    parser.add_argument('--csv', type=str, nargs='*', default=None)
    parser.add_argument('--pretrained', type=str, default='pretrained')
    args=parser.parse_args()

    main(args)
