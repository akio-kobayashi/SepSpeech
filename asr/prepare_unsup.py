import os, glob, sys
import argparse
import pandas as pd
import wave
import re
import argparse

def main(args):

    root_path = args.root #'/export/Broadcast/vad/train/'
    
    # source, label, key, length
    data_list = {'source': [], 'key': [], 'length': []}

    min_samples = args.min_sec * 16000
    max_samples = args.max_sec * 16000
    
    for path in glob.glob(os.path.join(root_path, '**/*.wav'), recursive=True):
        path = os.path.abspath(path)
        wav = wave.open(path, 'r')
        length = int(len(wav.readframes(wav.getnframes()))/2)
        if length >= min_samples and length <= max_samples:
            data_list['source'].append(path)
            data_list['length'].append(length)
            basename = os.path.splitext(os.path.basename(path))[0]
            data_list['key'].append(basename)
            
    df = pd.DataFrame.from_dict(data_list, orient='columns')
    df.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--min_sec', type=float, default=3)
    parser.add_argument('--max_sec', type=float, default=20)
    args=parser.parse_args()

    main(args)

