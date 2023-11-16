import pandas as pd
import argparse
import os, sys, re

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--upsample', type=int, default=3)
    args = parser.parse_args()
    pattern = re.compile(r'⠲$')

    df = pd.read_csv(args.input_csv)
        
    max_len = len(df)
    remove_indices = []
    for index, row in df.iterrows():
        input_length = row['input_length']
        output_length = row['output_length']
        '''
        input_length = output_length = 0
        with open(row['input_label'], 'r') as  f:
            line = f.readline().strip()
            input_length = len(list(line))
                
        with open(row['output_label'], 'r') as f:
            line = f.readline().strip()
            line = re.sub(pattern, '', line)
            output_length = len(list(line))
        '''
        if input_length == 0 or output_length == 0:
            remove_indices.append(index)
        if args.upsample * input_length < output_length:
            remove_indices.append(index)
                
    df.drop(index=remove_indices, inplace=True)
    print(
        f"Drop {len(remove_indices)} utterances from {max_len} "
    )
    df.to_csv(args.output_csv)
    
if __name__ == "__main__":
    main()
