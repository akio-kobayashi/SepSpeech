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
from lite.solver import LitSepSpeaker
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

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    decoder = whisper.load_model("large").to(device)

    df_out = pd.DataFrame(index=None, 
                          columns=['key', 'source', 'source_result'])
    keys = []
    sources = []
    source_decoded = []
    df = pd.read_csv(args.input_csv)
    with torch.no_grad():
        for index, row in df.iterrows():

            key = os.path.splitext(os.path.basename(row['source']))[0]
            keys.append(key)
            sources.append(row['source'])

            decoded = decoder.transcribe(row['source'], verbose=None, language='ja')
            source_decoded.append(decoded['text'])
            
    df_out['key'], df_out['source'], df_out['source_result'] = keys, sources, source_decoded
    df_out.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--model_type', type=str, default='tasnet')
    args = parser.parse_args()

    main(args)

