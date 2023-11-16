import os
import numpy as np
import argparse
import yaml
import warnings
import pandas as pd
import wave

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='config')
parser.add_argument('--output_csv', type=str, default=None)
parser.add_argument('--text_dir', type=str, default='/tmp/')
parser.add_argument('--posterior', type=float, default=1.0e-3)
args = parser.parse_args()

df = pd.read_csv(args.input_csv)

removed = []
for idx, row in df.iterrows():
    logp = row['logp']
    if type(logp) == str:
        score = np.sum( [ float(p) for p in row['logp'].split() ] )
    else:
        score = logp
    if score > args.posterior:
        removed.append(idx)

df.drop(removed, inplace=True)
data_list = {'source': [], 'label': [], 'key': [], 'length': []}
for idx, row in df.iterrows():
    '''
      source,key,length,logp,decode -> source,label,key,length
    '''
    label = row['decode']
    if type(label) == str:
        data_list['source'].append(row['source'])
        data_list['key'].append(row['key'])
        path = os.path.join(args.text_dir, row['key']+'.txt')
        with open(path, 'w') as f:
            f.write(label+'\n')
        data_list['label'].append(path)
        wav = wave.open(row['source'], 'r')
        data_list['length'].append(len(wav.readframes(wav.getnframes())))

df_out = pd.DataFrame.from_dict(data_list, orient='columns')
df_out.to_csv(args.output_csv, index=False)

