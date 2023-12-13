import pandas as pd
import argparse
import os, sys

def main(args):

    df = pd.read_csv(args.input_csv)
    with open(args.output, 'w') as f:
        f.write(f'fx vx\n')
        for idx, row in df.iterrows():
            tag = row['method']+'_'+row['snr']+'_'+row['packet_loss']+'_'+row['width']
            f.write(f'{tag} {row[args.column_name]}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', nargs='*',)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--column_name', type=str, required=True)
    args=parser.parse_args()

    main(args)
