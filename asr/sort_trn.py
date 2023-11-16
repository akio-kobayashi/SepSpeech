#!/usr/bin/python3

import argparse
import array
import math
import numpy as np
import random
import os, sys
import re

def main(args):
    pattern1 = re.compile(r'\s+')
    pattern2 = re.compile(r'\s\S+$')
    texts={}
    with open(args.trn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip().split()[-1]
            texts[label] = line.strip()
    outputs = sorted(texts.items())
    with open(args.output, 'w') as f:
        for label, line in outputs:
            if args.split:
                line = pattern2.sub('', line)
                line = pattern1.sub(' ', ' '.join(list(line)))
            f.write(line+' '+label+'\n')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn', type=str, required=True)
    parser.add_argument('--output', type=str)
    parser.add_argument('--split', action='store_true')
    args=parser.parse_args()

    main(args)
