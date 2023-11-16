import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import os, sys, re

def main(args):

    pattern1=re.compile('⠲$')
    pattern2=re.compile('\s+')
    texts = {}
    df = pd.read_csv(args.csv)
    for idx, row in df.iterrows():
        path, key = row['label'], row['key']
        with open(path, 'r') as f:
            line = f.readline().strip()
            if args.braille:
                line = pattern1.sub('', line)
            else:
                line = ' '.join(list(line))
        if args.split:
            line = pattern2.sub(' ', ' '.join(list(line)))
        texts[key] = line
        
    outputs = sorted(texts.items())
    with open(args.output, 'w') as f:
        for key, label in outputs:
            f.write(label+' ('+key+')\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--braille', action='store_true')
    parser.add_argument('--split', action='store_true')
    args=parser.parse_args()

    main(args)
