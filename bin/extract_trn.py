import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import os, sys, re

def main(args):

    pattern0 = re.compile(r'、|。|・|？|！|…|,|\?|\!')
    pattern1 = re.compile(r'\s+')
    pattern2 = re.compile(r'\S+$')    

    refs={}
    with open(args.ref, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = pattern2.search(line.strip())
            tag = result.group().replace('(', '').replace(')', '')
            refs[tag] = line.strip()

    hyps={}
    df = pd.read_csv(args.csv)
    for index, row in df.iterrows():
        key, result = row['key'], row['source_result']
        result = pattern0.sub('', result)
        result = pattern1.sub('', result)
        result = ' '.join(list(result))
        hyps[key] = result

    ref_path=os.path.join(args.output_dir, 'ref.trn')
    hyp_path=os.path.join(args.output_dir, 'hyp.trn')
    with open(ref_path, 'w') as reff:
        with open(hyp_path, 'w') as hypf:
            keys = sorted(refs.keys())
            for key in keys:
                if key in hyps.keys():
                    reff.write(refs[key]+'\n')
                    hypf.write(hyps[key]+' ('+key+')\n')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--output_dir', type=str)
    args=parser.parse_args()

    main(args)
