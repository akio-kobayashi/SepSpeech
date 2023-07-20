import os, glob, sys
import argparse
import pandas as pd
import re

def main(args):

    df = None
    for path in args.csv:
        df_new = pd.read_csv(path)
        if df is None:
            df = df_new
        else:
            df = pd.concat([df, df_new])
            
    df.to_csv(args.output_csv, index=False)
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', nargs='*')
    parser.add_argument('--output-csv', type=str, required=True)
    args=parser.parse_args()

    main(args)
