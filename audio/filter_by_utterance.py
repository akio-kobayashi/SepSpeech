import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd


'''
発話集合により発話を振り分ける：例 ATR Bset & Cset
'''
def main(args):

    df_remove = None
    df = pd.read_csv(args.csv)
    for tag in args.utterance_set:
        df_filt = df.query("utt.str.contains(@tag)")
        if df_remove is None:
            df_remove = df_filt
        else:
            df_remove = pd.concat([df_remove, df_filt])
    if args.remove:
        df = df[~df['utt'].isin(df_remove['utt'])]
    else:
        df = df_remove
    df.to_csv(args.output_csv)
    
    '''
        utt = tag+'01'
        df_utt = df.query('utt==@utt')
        for _, row in df_utt.iterrows():
            speaker = row['speaker']
            if args.remove is True:
                df = df[df['speaker'] != speaker]
            else:
                d = df[df['speaker'] == speaker]
                if df_new is None:
                    df_new = d
                else:
                    df_new = pd.concat([df_new, d])
    '''
    '''
    if args.remove is True:
        df.to_csv(args.output_csv, index=False)
    else:
        df_new.to_csv(args.output_csv, index=False)
    ''' 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('--utterance-set', type=str, nargs='*')
    args=parser.parse_args()

    main(args)
