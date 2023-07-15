import os, sys
import argparse
import array
import math
import numpy as np
import random
import wave
import pandas as pd
import yaml
import torch
import torchaudio
import numpy as np
from augment.opus_augment_simulate import OpusAugment
from augment.reverb_augment import ReverbAugment

def rms(wave):
    return torch.sqrt(torch.mean(torch.square(wave)))

def adjusted_rms(_rms, snr):
    return _rms / (10**(float(snr) / 20))

def mix(source, noise, snr):
    source_rms = rms(source)
    noise_rms = rms(noise)

    adj_noise_rms = adjusted_rms(source_rms, snr)
    adj_noise = noise * (adj_noise_rms / noise_rms)

    return source+adj_noise

def main(args):

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
        
    reverb_source_func = ReverbAugment(**config['augment']['reverb']['params'],
                                       source_loc=config['augment']['reverb']['source_loc'],
                                       loc_range=config['augment']['reverb']['source_loc_range']
                                       )
    reverb_noise_func = ReverbAugment(**config['augment']['reverb']['params'],
                                      source_loc=config['augment']['reverb']['noise_loc'],
                                      loc_range=config['augment']['reverb']['noise_loc_range']
                                      )
    opus_func = OpusAugment(**config['augment']['opus'])

    mix_list = {'mixture': [],
                'source': [],
                'noise': [],
                'length': [],
                'speaker': [],
                'index': [],
                'snr': [],
                'rt60': [],
                'bps': [],
                'packet_loss_rate': [],
                'resample': []}
    
    df_speech = pd.read_csv(args.speech_csv)
    df_noise = pd.read_csv(args.noise_csv)
    rand_noise = df_noise.sample(len(df_speech), replace=True)

    max_snr=config['augment']['mixing']['max_snr']
    min_snr=config['augment']['mixing']['min_snr']
    
    for index, row in df_speech.iterrows():
        if index % args.split != args.part:
            continue
        source_path, speaker, id = row['source'], row['speaker'], row['index']
        noise_path = rand_noise.iloc[rand_noise.index[index]]['noise']
        mix_path = os.path.join(args.output_dir,
                                os.path.splitext(os.path.basename(source_path))[0])+ '_mix.wav'
        print(mix_path)
        source,sr = torchaudio.load(source_path)
        noise,sr = torchaudio.load(noise_path)

        start = np.random.randint(0, noise.shape[-1] - source.shape[-1])
        stop = start + source.shape[-1]
        noise = noise[:, start:stop]
        
        reverb_noise, rt60 = reverb_noise_func(noise, rt60=0.0)
        reverb_source, _ = reverb_source_func(source, rt60=rt60)
        snr = np.random.rand() * (max_snr-min_snr) + min_snr
        
        mixture = mix(reverb_source, reverb_noise, snr)
        
        mixture, bps, packet_loss_rate, resample, markov_states = opus_func(mixture, bps=0)
        #markov_states=np.array(markov_states)
        #markov_states_path = os.path.join(args.out_dir,
        #                                  os.path.splitext(os.path.basename(source_path))[0])+'.npy'
        #np.save(markov_states_path, markov_states)
        torchaudio.save(filepath=mix_path, src=mixture.to('cpu'), sample_rate=sr) # save as float
        
        mix_list['mixture'].append(mix_path)
        mix_list['source'].append(source_path)
        mix_list['noise'].append(noise_path)
        mix_list['length'].append(source.shape[-1])
        mix_list['speaker'].append(speaker)
        mix_list['index'].append(id)
        mix_list['snr'].append(snr)
        mix_list['rt60'].append(rt60)
        mix_list['bps'].append(bps)
        mix_list['packet_loss_rate'].append(bps)
        mix_list['resample'].append(resample)
        
    df_mix = pd.DataFrame.from_dict(mix_list, orient='columns')
    df_mix.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech-csv', type=str, required=True)
    parser.add_argument('--noise-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--part', type=int, required=True)
    
    args=parser.parse_args()

    main(args)
