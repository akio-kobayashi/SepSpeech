import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import os, sys

def main(args):

    df_src = pd.read_csv(args.source_csv, index_col=0)
    df_tgt = pd.read_csv(args.target_csv, index_col=0)

    df_tgt_valid=None
    for n in range(args.num_valid):
        df_src_valid = df_src.sample(args.num_valid)
        df_src.drop(df_src_valid.index, inplace=True)
        for index, row in df_src_valid.iterrows():
            utt=row['utt']
            df_filt = df_tgt.query('utt==@utt')
            df_sample = df_filt.sample(1)
            df_tgt.drop(df_sample.index, inplace=True)
            if df_tgt_valid is None:
                df_tgt_valid = df_sample
            else:
                df_tgt_valid = pd.concat([df_tgt_valid, df_sample])

    df_src.to_csv(args.output_source_train, index=False)
    df_src_valid.to_csv(args.output_source_valid, index=False)
    df_tgt.to_csv(args.output_target_train, index=False)
    df_tgt_valid.to_csv(args.output_target_valid, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_csv', type=str, required=True)
    parser.add_argument('--target_csv', type=str, required=True)
    parser.add_argument('--num_valid', type=int, default=1)
    parser.add_argument('--output_source_train', type=str, default='train.csv')
    parser.add_argument('--output_source_valid', type=str, default='valid.csv')
    parser.add_argument('--output_target_train', type=str, default='train.csv')
    parser.add_argument('--output_target_valid', type=str, default='valid.csv')
    args=parser.parse_args()

    main(args)
