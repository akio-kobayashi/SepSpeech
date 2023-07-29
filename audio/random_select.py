import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import os, sys

def main(args):

    df = pd.read_csv(args.csv)
    assert len(df) > args.num
    df_out=df.sample(args.num, replace=False)
    df_out=df_out.sort_values('key')
    df_out.to_csv(args.output, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--output', type=str, required=True)
    args=parser.parse_args()

    main(args)
