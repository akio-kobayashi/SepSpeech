import argparse
import json
import logging
import sys, os
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTImeObjectiveIntelligibility
from torchmetrics import ScaleInvariantSignalDistortionRatio
import torch
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from lite.solver import LitSepSpeaker
import conventional.speech_dataset as sd
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np
import whisper

def read_audio(path):
    wave, sr = torchaudio.load(path)
    std, mean = torch.std_mean(wave, dim=-1)
    wave = (wave - mean)/std
    return torch.t(wave)

def main():

    decoder = whisper.load_model("large")

    decoder.eval()
    
    df_out = pd.DataFrame(index=None, 
                          columns=['key', 'mixture', 'source', 'mix_result', 'est_result', 'mix_pesq', 'mix_stoi', 'mix_sdr', 'est_pesq', 'est_stoi', 'est_sdr'])
    keys = []
    files = []
    results = []
    df = pd.read_csv(args.input_path)
    with torch.no_grad():
        for index, row in df.iterrows():

            keys.append(os.path.splitext(os.path.basename(row['source']))[0])
            files.append(row['source'])

            decoded = decoder.transcribe(row['source'], verbose=False, language='ja')
            results.append(decoded['text'])    
           
    df_out['key'], df['source'], df['source_result'] = keys, files, results
    df.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    args = parser.parse_args()

    main(args)
