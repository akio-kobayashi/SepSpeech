import argparse
import json
import logging
import sys, os
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics import ScaleInvariantSignalDistortionRatio
import torch
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from lite.radio_solver import LitDenoiser
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np
import whisper

def get_divisor(model):

    start = -1
    end = -1
    with torch.no_grad():
        for n in range(1000, 1100):
            x = torch.rand(4, n)
            o = model(x.cuda())
            if x.shape[-1] == o.shape[-1] :
                if start < 0:
                    start = n
                else:
                    end = n
                    break
    return end - start

def padding(x, divisor):
    pad_value = divisor - x.shape[-1] % divisor -1
    return F.pad (x, pad=(1, pad_value ), value=0.)

def read_audio(path):
    wave, sr = torchaudio.load(path)
    return wave

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    decoder = whisper.load_model("large").to(device)

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    config['model_type'] = args.model_type # do nothing
    
    assert args.checkpoint is not None
    model = LitDenoiser.load_from_checkpoint(args.checkpoint,
                                             config=config).to(device)
    divisor = get_divisor(model)
    model.eval()
    
    sample_rate = config['dataset']['segment']['sample_rate']
    _pesq = PerceptualEvaluationSpeechQuality(sample_rate, 'wb').to(device)
    _stoi = ShortTimeObjectiveIntelligibility(sample_rate, extended=False).to(device)
    _sdr = ScaleInvariantSignalDistortionRatio().to(device)
    
    df_out = pd.DataFrame(index=None, 
                          columns=['key', 'clean', 'noisy', 'denoise', 'clean_result', 'noisy_result', 'denoise_result', 'noisy_pesq', 'noisy_stoi', 'noisy_sdr', 'denoise_pesq', 'denoise_stoi', 'denoise_sdr'])
    keys = []
    denoise_pesq, denoise_stoi, denoise_sdr = [], [], []
    noisy_pesq, noisy_stoi, noisy_sdr = [], [], []
    noisy, clean, denoise = [], [], []
    noisy_decoded, clean_decoded, denoise_decoded = [], [], []

    df = pd.read_csv(args.input_csv)

    with torch.no_grad():
        for [index, row] in df.iterrows():

            key = row['key']
            keys.append(key)
            
            noisy.append(row['noisy'])
            clean.append(row['clean'])

            noisy_wav = read_audio(row['noisy'])
            clean_wav = read_audio(row['clean'])
            noisy_pesq.append(_pesq(noisy_wav.cuda(), clean_wav.cuda()).cpu().detach().numpy())
            noisy_stoi.append(_stoi(noisy_wav.cuda(), clean_wav.cuda()).cpu().detach().numpy())
            noisy_sdr.append(_sdr(noisy_wav.cuda(), clean_wav.cuda()).cpu().detach().numpy())
            
            noisy_original_length = noisy_wav.shape[-1]
            clean_original_length = clean_wav.shape[-1]
            assert noisy_original_length == clean_original_length

            # normalize and padding
            noisy_std, noisy_mean = torch.std_mean(noisy_wav)
            noisy_wav = (noisy_wav - noisy_mean)/noisy_std
            if divisor > 0 and noisy_original_length % divisor > 0:
                noisy_wav = padding(noisy_wav, divisor)

            denoise_wav = model(noisy_wav.to(device))
            denoise_wav = denoise_wav[:, :noisy_original_length]
            denoise_wav *= noisy_std
            
            denoise_pesq.append(_pesq(denoise_wav.cuda(), clean_wav.cuda()).cpu().detach().numpy())
            denoise_stoi.append(_stoi(denoise_wav.cuda(), clean_wav.cuda()).cpu().detach().numpy())
            denoise_sdr.append(_sdr(denoise_wav.cuda(), clean_wav.cuda()).cpu().detach().numpy())
            
            denoise_outdir = args.output_dir
            if not os.path.exists(denoise_outdir):
                os.path.mkdir(denoise_outdir)
            outpath = os.path.join(denoise_outdir, key) + '_denoise.wav'
            torchaudio.save(filepath=outpath, src=denoise_wav.to('cpu'),
                            sample_rate=sample_rate)
            #                            encoding='PCM_S', bits_per_sample=16 
            #                            )
            denoise.append(outpath)

            decoded = decoder.transcribe(outpath, verbose=None, language='ja')
            denoise_decoded.append(decoded['text'])
            decoded = decoder.transcribe(row['noisy'], verbose=False, language='ja')
            noisy_decoded.append(decoded['text'])
            decoded = decoder.transcribe(row['clean'], verbose=False, language='ja')
            clean_decoded.append(decoded['text'])
            
            
    df_out['key'], df_out['noisy'], df_out['clean'], df_out['denoise'] = keys, noisy, clean, denoise
    df_out['noisy_pesq'], df_out['noisy_stoi'], df_out['noisy_sdr'] = noisy_pesq, noisy_stoi, noisy_sdr
    df_out['denoise_pesq'], df_out['denoise_stoi'], df_out['denoise_sdr'] = denoise_pesq, denoise_stoi, denoise_sdr
    df_out['noisy_result'], df_out['denoise_result'], df_out['clean_result'] = noisy_decoded, denoise_decoded, clean_decoded
    
    pesq, stoi, sdr = np.mean(noisy_pesq), np.mean(noisy_stoi), np.mean(noisy_sdr)
    print("noisy:  PESQ = %.4f , STOI = %.4f, SI-SDR = %.4f" % (pesq, stoi, sdr))
    
    pesq, stoi, sdr = np.mean(denoise_pesq), np.mean(denoise_stoi), np.mean(denoise_sdr)
    print("denoise: PESQ = %.4f , STOI = %.4f, SI-SDR = %.4f" % (pesq, stoi, sdr))

    df_out.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--output_dir')
    parser.add_argument('--model_type', type=str, default='unet')
    args = parser.parse_args()

    main(args)
