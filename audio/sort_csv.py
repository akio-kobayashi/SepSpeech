import os, glob, sys
import argparse
import pandas as pd
import re

def main(args):

    df = pd.read_csv(args.csv)
    df_out = df.sort_values('key')
    df_out.to_csv(args.output_csv, index=False)
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    args=parser.parse_args()

    main(args)
