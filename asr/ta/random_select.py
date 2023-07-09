from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main(args):
    df = pd.read_csv(args.csv)
    df=df.sample(n=args.number)
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--number', type=int, default=20000)
    parser.add_argument('--output', type=str, required=True)
    args=parser.parse_args()
    
    main(args)
