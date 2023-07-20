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
    df_out = pd.DataFrame(columns=['key', 'clean', 'noisy', 'length', 'decode'])
    _key, _source, _length, _decode = [], [], [], []
    with torch.no_grad():
        for [index, row] in df.iterrows():
            source = row['clean']
            length = row['length']
            decoded = decoder.transcribe(source, verbose=False, language='ja')
            key = os.path.splitext(os.path.basename(source))[0]

            _key.append(key)
            _source.append(source)
            _length.append(length)
            _decode.append(decoded['text'])
            
    df_out['key'] = _key
    df_out['length'] = _length
    df_out['source'] = _source
    df_out['decode'] = _decode
    df_out.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-csv', type=str)
    parser.add_argument('--output-csv', type=str)
    args = parser.parse_args()

    main(args)
