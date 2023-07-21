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

    config['augment']['reverb']['source_loc_range'] = [0,0,0]
    config['augment']['reverb']['noise_loc_range'] = [0,0,0]
    
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
                'start': [],
                'end': [],
                'length': [],
                'speaker': [],
                'index': [],
                'snr': [],
                'rt60': [],
                'bps': [],
                'packet_loss_rate': [],
                'resample': [],
                'markov_states': []
                }
    
    df_speech = pd.read_csv(args.speech_csv)
    df_noise = pd.read_csv(args.noise_csv)
    df_mixed = None
    rand_noise = None
    if args.mixed_csv is None:
        rand_noise = df_noise.sample(len(df_speech), replace=True)
    else:
        df_mixed = pd.read_csv(args.mixed_csv)

    if df_mixed is not None:
        df_speech = df_mixed
        
    for index, row in df_speech.iterrows():
        source_path, speaker, id = row['source'], row['speaker'], row['index']
        if df_mixed is None:
            noise_path = rand_noise.iloc[rand_noise.index[index]]['noise']
        else:
            noise_path = row['noise']
        mix_path = os.path.join(args.output_dir,
                                os.path.splitext(os.path.basename(source_path))[0])+ '_mix.wav'
        print(mix_path)
        source,sr = torchaudio.load(source_path)
        noise,sr = torchaudio.load(noise_path)

        if df_mixed is None:
            start = np.random.randint(0, noise.shape[-1] - source.shape[-1])
            end = start + source.shape[-1]
        else:
            start = row['start']
            end = row['end']
        noise = noise[:, start:end]

        min_rt60, max_rt60 = reverb_noise_func.get_rt60s()
        rt60 = args.rt60
        #(max_rt60 - min_rt60) * np.random.rand() + min_rt60

        reverb_source, reverb_noise = None, None
        if rt60 > 0.2:
            reverb_noise = reverb_noise_func(noise, rt60)[0]
            reverb_source = reverb_source_func(source, rt60)[0]
        snr = args.snr
        #np.random.rand() * (max_snr-min_snr) + min_snr

        packet_loss_rate = args.packet_loss_rate
        bps = args.bps
        
        if reverb_source is not None:
            mixture = mix(reverb_source, reverb_noise, snr)
        else:
            mixture = mix(source, noise, snr)

        markov_states=None
        if df_mixed is not None:
            markov_states = np.load(row['markov_states'])
        mixture, bps, packet_loss_rate, resample, markov_states = opus_func(mixture, bps, packet_loss_rate, markov_states)
        markov_states=np.array(markov_states)
        markov_states_path = os.path.join(args.output_dir,
                                          os.path.splitext(os.path.basename(source_path))[0])+'.npy'
        np.save(markov_states_path, markov_states)
        mix_list['markov_states'].append(markov_states_path)
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
        mix_list['resample'].append(resample)
        mix_list['packet_loss_rate'].append(packet_loss_rate)
        mix_list['start'].append(start)
        mix_list['end'].append(end)

        
    df_mix = pd.DataFrame.from_dict(mix_list, orient='columns')
    df_mix.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech-csv', type=str, required=True)
    parser.add_argument('--noise-csv', type=str, required=True)
    parser.add_argument('--mixed-csv', type=str, default=None)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--snr', type=float, default=20)
    parser.add_argument('--rt60', type=float, default=0.2)
    parser.add_argument('--bps', type=int, default=16000)
    parser.add_argument('--packet_loss_rate', type=float, default=0.1)
    
    args=parser.parse_args()

    main(args)
