import torch
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from xvector.solver import LitXVector
from argparse import ArgumentParser
import yaml

  
def main(config, args):

    model = LitXVector.load_from_checkpoint(args.checkpoint, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    mixture, sr = torchaudio.load(args.mixture)
    #mx = torch.max(mixture)
    std, mean = torch.std_mean(mixture, dim=-1)
    mixture = (mixture-mean)/std
    enroll, sr = torchaudio.load(args.enroll)
    std, mean = torch.std_mean(enroll, dim=-1)
    enroll = (enroll - mean)/std
    length = len(mixture.t())
    with torch.no_grad():
        output, _ = model(mixture.cuda(), enroll.cuda())
        output *= std
    #std, mean = torch.std_mean(output, dim=-1)
    #output /= std
    #output = output/torch.max(output) * mx
    torchaudio.save(filepath=args.output, src=output.to('cpu'), sample_rate=sr)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mixture', type=str, required=True)
    parser.add_argument('--enroll', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.wav')
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config, args)
