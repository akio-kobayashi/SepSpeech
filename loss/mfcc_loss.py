import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
#from loss.stft_loss import stft
from einops import rearrange

'''
def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1)) #dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1) #idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

class CepstrumLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spectrogram = T.Spectrogram(
            n_fft=config['n_fft'], 
            win_length=config['win_length'],
            hop_Length=config['hop_length'],
            power=1
        )
        self.loss = nn.L1Loss()
        
    def compute_cepstrum(self, signal):
        spec = torch.log(self.spectrogram(signal))
        B, F, T = spec.shape
        spec = rearrange(spec, 'b f t -> (b t) f')
        spec = dct(spec)
        spec = rearrange(spec, '(b t) f -> b f t', t=T)
        return spec
    
    def forward(self, preds, targets, lengths):
        preds = self.compute_cepstrum(preds)
        targets = self.compute_cepstrum(targets)
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :, :lengths[b]] = 1.
        return self.loss(preds * mask, targets * mask) / torch.sum(mask)
'''

class MFCCLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mfcc = T.MFCC(
            sample_rate = config['sample_rate'],
            n_mfcc = config['n_mfcc'],
            log_mels = True,
            melkwargs={"n_fft": config['n_fft'],
                       "hop_length": config['hop_length'],
                       "n_mels": config['n_mels'],
                       "center": False}
        )
        self.loss = nn.L1Loss(reduction='sum')
        
    def forward(self, preds, targets, lengths):
        # (b, t) -> (b f t)
        preds = self.mfcc(preds)
        targets = self.mfcc(targets)
        
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :, :lengths[b]] = 1.
        return self.loss(preds * mask, targets * mask) / torch.sum(mask)

class LFCCLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lfcc = T.LFCC(
            sample_rate = config['sample_rate'],
            n_filter = config['n_filter'],
            n_lfcc = config['n_lfcc'],
            log_lf = True,
            speckwargs={"n_fft": config['n_fft'],
                        "hop_length": config['hop_length'],
                        "center": False}
        )
        self.loss = nn.L1Loss(reduction='sum')
        
    def forward(self, preds, targets, lengths):
        # (b, t) -> (b f t)
        preds = self.lfcc(preds)
        targets = self.lfcc(targets)
        
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :, :lengths[b]] = 1.
        return self.loss(preds * mask, targets * mask) / torch.sum(mask)
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml
    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['config']['loss']['mfcc']['hop_length']=200
    config['config']['loss']['mfcc']['n_mels']=20
    loss=MFCCLoss(config['config']['loss']['mfcc'])
    
    source,_=torchaudio.load(args.source)
    std, mean = torch.std_mean(source)
    source = (source-mean)/std
    target,_=torchaudio.load(args.target)
    std, mean = torch.std_mean(target)
    target = (target-mean)/std
    lengths = [source.shape[-1]]
    _loss = loss(source, target, lengths)
    
    print(_loss)
