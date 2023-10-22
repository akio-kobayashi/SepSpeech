import os, sys
import pandas as pd
import json
from argparse import ArgumentParser

def main(args):
    speaker2id={"<UNK>": 0}

    counter = 1
    for path in args.input_csv:
        df = pd.read_csv(path)
        for index, row in df.iterrows():
            if row['speaker'] not in speaker2id.keys():
                speaker2id[row['speaker']] = counter
                counter += 1
    
    with open(args.output, 'w') as wf:
        json.dump(speaker2id, wf, indent=2)

    print(f'number of speakers : {counter}')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str, nargs='*')
    parser.add_argument('--output', type=str, required=True)
    args=parser.parse_args()

    main(args)
