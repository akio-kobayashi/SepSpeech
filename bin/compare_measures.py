import argparse
import json
import logging
import sys, os
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np

def main(args):
    d_file, d_obj_val, d_clr_val, d_hue=[], [], [], []
    obj_df=None
    for file in args.objective:
        df = pd.read_csv(file)
        if obj_df is None:
            obj_df = df
        else:
            obj_df = pd.concat([obj_df, df])

    clr_df=None
    for file in args.clarity:
        df = pd.read_csv(file)
        if clr_df is None:
            clr_df = df
        else:
            clr_df = pd.concat([clr_df, df])

    for idx, row in obj_df.iterrows():
        if args.mixture :
            file = row['mixture']
            obj_val = row["mix_"+args.key]
            query='mixture==@file'
        else:
            file = row['estimate']
            obj_val = row['est_'+args.key]
            query='estimate==@file'
        result=clr_df.query(query)
        if result is None:
            raise ValueError('wrong file name')
        rrow=result.iloc[0]
        if args.mixture :
            clr_val=rrow['mix_hasqi']
        else:
            clr_val=rrow['est_hasqi']
        d_file.append(file)
        d_obj_val.append(obj_val)
        d_clr_val.append(clr_val)
        d_hue.append(args.hue)
        #print(f'{obj_val} {clr_val}')
    new_df=pd.DataFrame(index=None)
    new_df['filename']=d_file
    new_df['objective']=d_obj_val
    new_df['hasqi']=d_clr_val
    new_df[args.hue_name]=d_hue
    
    new_df.to_csv(args.output, index=False)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--objective', nargs='*')
    parser.add_argument('--clarity', nargs='*')
    parser.add_argument('--key', type=str, default='OVRL_raw')
    parser.add_argument('--hue', type=str)
    parser.add_argument('--mixture', action='store_true')
    parser.add_argument('--output', type=str)
    parser.add_argument('--hue_name', type=str)
    
    args = parser.parse_args()

    main(args)

