import os, sys
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True)
parser.add_argument('--output_csv', type=str, required=True)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--min_sec', type=int, default=3)
parser.add_argument('--max_hour', type=int, default=1)
parser.add_argument('--segment', type=int, default=30)

args=parser.parse_args()

df = pd.read_csv(args.input_csv)
min_samples = args.min_sec * args.sample_rate #3 * 16000
max_samples = args.max_hour * 3600 * 16000
seg_samples = int(args.segment * args.sample_rate)
df = df[df['length'] / 2 <= seg_samples]
df = df[df['length'] / 2 >= min_samples ]

df = df.sample(frac=1)
line=0
samples=0
for index, row in df.iterrows():
    samples += row['length']
    if samples > max_samples:
        break
    line+=1

df = df.head(line)
df.to_csv(args.output_csv)
