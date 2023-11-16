import os, sys, glob
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)
#parser.add_argument('--output', type=str, required=True)
args=parser.parse_args()

lines = {}
for path in glob.glob(os.path.join(args.root, '**/*.txt'), recursive=True):
    basename = os.path.splitext(os.path.basename(path))[0]
    with open(path, 'r') as f:
        line = f.readline().strip()
        lines[basename] = line
keys = sorted(lines.keys())
for key in keys:
    print(f'{key} {lines[key]}')
