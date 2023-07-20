import os, sys, re, glob
import pandas as pd
from argparse import ArgumentParser
import wave

def main(args):
    pattern=re.compile(r'_clean|_noisy')
    clean_wavs={}
    noisy_wavs={}
    clean_wav_lengths={}
    noisy_wav_lengths={}
    
    for path in glob.glob(os.path.join(args.root, '**/*.wav'), recursive=True):
        path=os.path.abspath(path)
        key = os.path.splitext(os.path.basename(path))[0]
        key = re.sub(pattern, '', key)
        if 'clean' in path:
            clean_wavs[key] = path
            wav = wave.open(path, 'r')
            clean_wav_lengths[key] = len(wav.readframes(wav.getnframes()))
        else:
            noisy_wavs[key] = path
            wav = wave.open(path, 'r')
            noisy_wav_lengths[key] = len(wav.readframes(wav.getnframes()))

    _clean, _noisy, _key, _length = [], [], [], []
    total_frames = 0
    for key in clean_wavs.keys():
        assert clean_wav_lengths[key] == noisy_wav_lengths[key]
        if clean_wav_lengths[key] < args.cutoff_frames[0]:
            continue
        if clean_wav_lengths[key] > args.cutoff_frames[1]:
            continue
        total_frames += clean_wav_lengths[key]
        _key.append(key)
        _clean.append(clean_wavs[key])
        _noisy.append(noisy_wavs[key])
        _length.append(clean_wav_lengths[key])
        
    df = pd.DataFrame(columns=['key', 'clean', 'noisy'])
    df['key'] = _key
    df['clean'] = _clean
    df['noisy'] = _noisy
    df['length'] = _length
    
    df.to_csv(args.output_csv, index=False)

    print(f'total files: {len(df)}, total frames: {total_frames}')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--cutoff_frames', type=int, nargs=2)
    args=parser.parse_args()

    main(args)

