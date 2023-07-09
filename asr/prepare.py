import os, glob, sys
import argparse
import pandas as pd
import wave
import re
import argparse

def main(args):

    root_path = '/export/S2B_finetune/wavs/'
    kanji_path = '/export/20230515/Data/text/kanji/'
    braille_path = '/export/20230515/Data/text/braille/'
    
    # source, label, key, length
    data_list = {'source': [], 'label': [], 'key': [], 'length': []}

    for path in glob.glob(os.path.join(root_path, '**/*.wav'), recursive=True):
        path = os.path.abspath(path)
        data_list['source'].append(path)
        
        wav = wave.open(path, 'r')
        data_list['length'].append(len(wav.readframes(wav.getnframes())))

        basename = os.path.splitext(os.path.basename(path))[0]
        data_list['key'].append(basename)
        if args.braille:
            data_list['label'].append(os.path.join(braille_path, basename) + '.txt')
        else:
            data_list['label'].append(os.path.join(kanji_path, basename) + '.txt')
            
    df = pd.DataFrame.from_dict(data_list, orient='columns')
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--braille', action='store_true')
    parser.add_argument('--output', type=str)
    args=parser.parse_args()

    main(args)

