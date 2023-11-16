import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import os, sys

def main(args):

    df_new = pd.DataFrame(index=None, columns=['key', 'input_label', 'output_label', 'input_length', 'output_length'])
    _key, _input_label, _output_label, _input_length, _output_length = [], [], [], [], []
    
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    for index, row in df1.iterrows():
        key, input_label = row['key'], row['label']
        output_label = df2.query('key==@key').iloc[0]['label']
        _key.append(key)
        _input_label.append(input_label)
        with open(input_label, 'r') as f:
            line = f.readline().strip()
            _input_length.append(len(list(line)))
        _output_label.append(output_label)
        with open(output_label, 'r') as f:
            line = f.readline().strip()
            _output_length.append(len(list(line)))
            
    df_new['key'] = _key
    df_new['input_label'] = _input_label
    df_new['output_label'] = _output_label
    df_new['input_length'] = _input_length
    df_new['output_length'] = _output_length

    df_new.to_csv(args.output_csv, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv1', type=str, required=True)
    parser.add_argument('--csv2', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    args=parser.parse_args()

    main(args)
