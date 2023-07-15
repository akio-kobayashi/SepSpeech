import os, glob, sys
import argparse
import pandas as pd

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=str, nargs='*')
    parser.add_argument('--ouput-csv', type=str, required=True)
    args=parser.parse_args()

    df_merge=None
    for csv_path in args.input_csv:
        if df_merge is None:
            df_merge = pd.read_csv(csv_path)
        else:
            df_temp = pd.read_csv(csv_path)
            df_merge = pd.concat([df_merge, df_temp])

    df_merge.to_csv(args.output_csv, index=False)

