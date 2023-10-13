import os, glob, sys
import argparse
import pandas as pd
import wave
import re

def main(args):

    df=None
    spk2id={}
    id=args.speaker_index

    # recursive search for WAVE files
    for path in glob.glob(os.path.join(args.root, '**/*.wav'), recursive=True):
        if args.relative:
            path=os.path.relpath(path)
        else:
            path=os.path.abspath(path)
        wav = wave.open(path, 'r')
        wavlen = len(wav.readframes(wav.getnframes()))
        file=os.path.basename(path)
        if args.noise:
            if df is None:
                df = pd.DataFrame(index=None, columns=['noise', 'length'], data=[[path, wavlen]])
            else:
                df2 = pd.DataFrame(index=None, columns=['noise', 'length'], data=[[path, wavlen]])
                df = pd.concat([df, df2], ignore_index=True, axis=0)
        else:
            if '_DT' in file: # JNAS Desktop recordings
                file = file.replace('_DT.wav', '')
                spk = file[1:-3]
                utt = file[len(file)-3:]
            elif file.startswith('M') or file.startswith('F'): # Deaf male/female balance sentences
                file = file.replace('.wav', '')
                spk = 'D'+file[0:4]
                utt = file[len(file)-3:]
            else:
                continue
            if not spk2id.get(spk):
                spk2id[spk] = id
                id+=1
            if df is None:
                df = pd.DataFrame(index=None, columns=['source', 'length', 'speaker', 'index', 'utt'],
                                  data=[[path, wavlen, spk, spk2id[spk], utt]])
            else:
                df2 = pd.DataFrame(index=None, columns=['source', 'length', 'speaker', 'index', 'utt'],
                                   data=[[path, wavlen, spk, spk2id[spk], utt]])
                df = pd.concat([df, df2], ignore_index=True, axis=0)
    if args.noise:
        df = df.sort_values('noise')
    else:
        df = df.sort_values('source')
    df.to_csv(args.output_csv, index=False)
    
    if args.noise is False:
        print("number of unieq speakers: %d max speaker index: %d" % (len(spk2id.values()), max(spk2id.values()) ))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./', required=True)
    parser.add_argument('--relative', action='store_true')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--speaker-index', type=int, default=1)
    args=parser.parse_args()

    main(args)
