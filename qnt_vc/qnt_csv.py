import os, sys, re
import pandas as pd
from argparse import ArgumentParser

pattern = re.compile(r'\d+')

def main(args):
    removed_index=[]
    df = pd.read_csv(args.input)
    for index, row in df.iterrows():
        utt = row['utterance']
        if re.fullmatch(pattern, utt) is None:
            removed_index.append(index)
    df.drop(removed_index, inplace=True)
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str)
    args=parser.parse_args()

    main(args)