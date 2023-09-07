import argparse
import json
import logging
import sys, os
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
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
    return wave

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    decoder = whisper.load_model("large-v2").to(device)

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    config['model_type'] = args.model_type
    
    assert args.checkpoint is not None
    model = LitSepSpeaker.load_from_checkpoint(args.checkpoint,
                                               config=config).to(device)
    
    model.eval()
    
    sample_rate = config['dataset']['segment']['sample_rate']
    _pesq = PerceptualEvaluationSpeechQuality(sample_rate, 'wb').to(device)
    _stoi = ShortTimeObjectiveIntelligibility(sample_rate, extended=True).to(device)
    _sdr = ScaleInvariantSignalDistortionRatio().to(device)
    
    df_out = pd.DataFrame(index=None, 
                          columns=['key', 'mixture', 'source', 'enroll', 'estimate', 'mix_result', 'est_result', 'mix_pesq', 'mix_stoi', 'mix_sdr', 'est_pesq', 'est_stoi', 'est_sdr'])
    keys = []
    est_pesq, est_stoi, est_sdr = [], [], []
    mix_pesq, mix_stoi, mix_sdr = [], [], []
    mixtures, sources, enrolls, estimates = [], [], [], []
    mix_decoded, est_decoded = [], []
    df = pd.read_csv(args.input_csv)
    df_enroll = pd.read_csv(args.enroll_csv)
    rand_enroll = df_enroll.sample(len(df), replace=True)
    with torch.no_grad():
        for [index, row], [_, row_enr] in zip(df.iterrows(), rand_enroll.iterrows()):

            key = os.path.splitext(os.path.basename(row['source']))[0]
            keys.append(key)
            mixtures.append(row['mixture'])
            sources.append(row['source'])

            mixture = read_audio(row['mixture'])
            source = read_audio(row['source'])
            if not args.no_mix:
                mix_pesq.append(_pesq(mixture.cuda(), source.cuda()).cpu().detach().numpy())
                mix_stoi.append(_stoi(mixture.cuda(), source.cuda()).cpu().detach().numpy())
                mix_sdr.append(_sdr(mixture.cuda(), source.cuda()).cpu().detach().numpy())
            else:
                mix_pesq.append('0.0')
                mix_stoi.append('0.0')
                mix_sdr.append('0.0')

            mix_std, mix_mean = torch.std_mean(mixture)
            src_std, src_mean = torch.std_mean(source)
            
            mixture_original_length = mixture.shape[-1]
            source_original_length = source.shape[-1]
            assert mixture_original_length == source_original_length

            enroll_path = row_enr['source']
            enroll = read_audio(enroll_path)
            std, mean = torch.std_mean(enroll)
            enroll = (enroll - mean)/std
            enrolls.append(enroll_path)

            # normalize and padding
            mixture = (mixture - mix_mean)/mix_std
            #if divisor > 0 and mixture_original_length % divisor > 0:
            #    mixture = padding(mixture, divisor)

            estimate, _, _ = model(mixture.to(device), enroll.to(device))
            estimate = estimate[:, :mixture_original_length]
            estimate *= mix_std
            est_pesq.append(_pesq(estimate.cuda(), source.cuda()).cpu().detach().numpy())
            est_stoi.append(_stoi(estimate.cuda(), source.cuda()).cpu().detach().numpy())
            est_sdr.append(_sdr(estimate.cuda(), source.cuda()).cpu().detach().numpy())
            
            estimate_outdir = args.output_dir
            if not os.path.exists(estimate_outdir):
                os.path.mkdir(estimate_outdir)
            outpath = os.path.join(estimate_outdir, key) + '_estimate.wav'
            torchaudio.save(filepath=outpath, src=estimate.to('cpu'),
                            sample_rate=sample_rate)
#                            encoding='PCM_S', bits_per_sample=16 
#                            )
            estimates.append(outpath)

            decoded = decoder.transcribe(outpath, verbose=None, language='ja')
            est_decoded.append(decoded['text'])
            if not args.no_mix:
                decoded = decoder.transcribe(row['mixture'], verbose=False, language='ja')
                mix_decoded.append(decoded['text'])
            else:
                mix_decoded.append('none')
            
    df_out['key'], df_out['mixture'], df_out['source'], df_out['enroll'], df_out['estimate'] = keys, mixtures, sources, enrolls, estimates
    df_out['mix_pesq'], df_out['mix_stoi'], df_out['mix_sdr'] = mix_pesq, mix_stoi, mix_sdr
    df_out['est_pesq'], df_out['est_stoi'], df_out['est_sdr'] = est_pesq, est_stoi, est_sdr
    df_out['mix_result'], df_out['est_result'] = mix_decoded, est_decoded

    if not args.no_mix:
        pesq, stoi, sdr = np.mean(mix_pesq), np.mean(mix_stoi), np.mean(mix_sdr)
        print("mixture:  PESQ = %.4f , STOI = %.4f, SI-SDR = %.4f" % (pesq, stoi, sdr))
    
    pesq, stoi, sdr = np.mean(est_pesq), np.mean(est_stoi), np.mean(est_sdr)
    print("estimate: PESQ = %.4f , STOI = %.4f, SI-SDR = %.4f" % (pesq, stoi, sdr))

    df_out.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--enroll_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--output_dir')
    parser.add_argument('--model_type', type=str, default='tasnet')
    parser.add_argument('--no_mix', action='store_true')
    args = parser.parse_args()

    main(args)

