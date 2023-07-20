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

def read_audio(path):
    wave, sr = torchaudio.load(path)
    return wave

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    decoder = whisper.load_model("large").to(device)
    df = pd.read_csv(args.input_csv)

    with torch.no_grad():
        for [index, row] in df.iterrows():

            key = row['key']
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
