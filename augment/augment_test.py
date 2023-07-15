import torch
import torchaudio
import numpy as np
from augment.opus_augment_simulate import OpusAugment
from augment.reverb_augment import ReverbAugment
import argparse
import yaml

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

if __name__ == '__main__':
    max_snr=20
    min_snr=0
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--noise', type=str, required=True)
    #parser.add_argument('--enroll', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args=parser.parse_args()
    
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

    source, sr = torchaudio.load(args.source)
    noise, sr = torchaudio.load(args.noise)
    #enroll = torchaudio.load(args.enroll)

    start = np.random.randint(0, noise.shape[-1] - source.shape[-1])
    stop = start + source.shape[-1]
    noise = noise[:, start:stop]

    min_rt60, max_rt60 = reverb_noise_func.get_rt60s()
    rt60 = (max_rt60 - min_rt60) * np.random.rand() + min_rt60

    reverb_noise = reverb_noise_func(noise, rt60)[0]
    reverb_source = reverb_source_func(source, rt60)[0]

    snr = np.random.rand() * (max_snr-min_snr) + min_snr
    mixture = mix(reverb_source, reverb_noise, snr)
    #mixture = mix(source, noise, snr)
    mixture = opus_func(mixture)[0]
    torchaudio.save(filepath=args.output, src=mixture.to('cpu'), sample_rate=sr, encoding='PCM_S') # save as float
