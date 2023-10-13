import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.unet import UNet
from models.unet2 import UNet2
from models.fast_unet import FastUNet
from models.conv_tasnet import ConvTasNet
from models.e3net import E3Net
from loss.mfcc_loss import MFCCLoss, LFCCLoss
from loss.stft_loss import MultiResolutionSTFTLoss
from loss.pesq_loss import PesqLoss
from loss.sdr_loss import NegativeSISDR
from loss.stoi_loss import NegSTOILoss
from loss.plcpa import MultiResPLCPA_ASYM, PLCPA_ASYM
from typing import Tuple
from einops import rearrange
from xvector.adacos import AdaCos
from models.e3net import LearnableEncoder
from xvector.model import X_vector

class SpeakerNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = None
        if config['model_type'] == 'unet':
            self.encoder = LearnableEncoder(chhot=config['xvector']['input_dim'])

        self.classifier = X_vector(input_dim = config['xvector']['input_dim'],
                                   output_dim = config['xvector']['output_dim'])
        
    def forward(self, x):
        # (b c t)
        if self.encoder is not None:
            x = self.encoder(x)
        x = rearrange(x, 'b c t -> b t c')
        return self.classifier(x)
    
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='sum')

    def forward(self, preds, targets, lengths):
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :lengths[b]] = 1.
        return self.loss(preds * mask, targets * mask) / torch.sum(mask)

class SpeakerLoss(nn.Module):
    def __init__(self, xvec_dim, num_speakers):
        self.metric = AdaCos(xvec_dim, num_speakers)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, xvec, speakers):
        _metric, _ = self.metric(x_vec, speakers)
        return self.loss(_metric, speakers)
        
'''
 PyTorch Lightning 用 solver
'''
class LitSepSpeaker(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config
        if config['model_type'] == 'unet':
            self.model = UNet(config)
        elif config['model_type'] == 'unet2':
            self.model = UNet2(config)
        elif config['model_type'] == 'tasnet':
            self.model = ConvTasNet(config)
        elif config['model_type'] == 'e3net':
            self.model = E3Net(config['e3net'])
        elif config['model_type'] == 'fast_unet':
            self.model = FastUNet(config)
        else:
            raise ValueError('wrong parameter: '+config['model_type'])

        self.spk_model = SpeakerNetwork(config)

        # speaker clustering loss
        self.spk_loss = SpeakerLoss(config['xvector']['output_dim'],
                                    config['xvector']['num_speakers'])
        self.spk_loss_weight = config['loss']['speaker']['weight']

        self.l1_loss_weight = self.mfcc_loss_weight = self.lfcc_loss_weight = self.plca_weight = 0.
        # Mean Absolute Error (temporal domain)        
        assert 'l1_loss' in config['loss'].keys()
        if config['loss']['l1_loss']['weight'] > 0.:
            self.l1_loss = L1Loss()
            self.l1_loss_weight = config['loss']['l1_loss']['weight']
            
        # MFCC Loss
        assert 'mfcc' in config['loss'].keys()
        if config['loss']['mfcc']['weight'] > 0.:
            self.mfcc_loss = MFCCLoss(config['loss']['mfcc'])
            self.mfcc_loss_weight = config['loss']['mfcc']['weight']
            
        # LFCC Loss
        assert 'lfcc' in config['loss'].keys()
        if config['loss']['lfcc']['weight'] > 0.:
            self.lfcc_loss = LFCCLoss(config['loss']['lfcc'])
            self.lfcc_loss_weight = config['loss']['lfcc']['weight']

        # PLCPA Loss
        assert 'plcpa_asym' in config['loss'].keys()
        if config['loss']['plcpa_asym']['weight'] > 0.:
            self.plcpa_loss = PLCPA_ASYM(config['loss']['plcpa_asym'])
            self.plcpa_weight = config['loss']['plcpa_asym']['weight']

        self.stft_loss = self.pesq_loss = self.stoi_loss = self.sdr_loss = None
        self.stft_loss_weight = self.pesq_loss_weight = self.stoi_loss_weight = self.sdr_loss_weight = 0.

        assert 'stft_loss' in config['loss'].keys()
        if config['loss']['stft_loss']['weight'] > 0.:
            self.stft_loss = MultiResolutionSTFTLoss()
            self.stft_loss_weight = config['loss']['stft_loss']['weight']
        
        assert 'pesq_loss' in config['loss'].keys()
        if config['loss']['pesq_loss']['weight'] > 0.:
            self.pesq_loss = PesqLoss(factor=1.,
                                      sample_rate=16000)
            self.pesq_loss_weight = config['loss']['pesq_loss']['weight']
        
        assert 'stoi_loss' in config['loss'].keys()
        if config['loss']['stoi_loss']['weight'] > 0.:
            self.stoi_loss = NegSTOILoss()
            self.stoi_loss_weight = config['loss']['stoi_loss']['weight']

        assert 'sdr_loss' in config['loss'].keys()
        if config['loss']['sdr_loss']['weight'] > 0.:
            self.sdr_loss = NegativeSISDR()
            self.sdr_loss_weight = config['loss']['sdr_loss']['weight']

        self.save_hyperparameters()

    def forward(self, mix:Tensor, embed:Tensor) -> Tuple[Tensor, Tensor]:
        return self.model(mix, embed) # return src_hat

    def compute_loss(self, estimate, target, lengths, valid=False):
        d = {}
        
        if self.l1_loss_weight > 0.:
            #with torch.cuda.amp.autocast('cuda', torch.float32):
            with torch.cuda.amp.autocast():
                _l1_loss = self.l1_loss(estimate, target, lengths)
            _loss  += self.l1_loss_weight * _l1_loss
            if valid:
                d['valid_l1_loss'] = _l1_loss
            else:
                d['train_l1_loss'] = _l1_loss

        if self.mfcc_loss_weight > 0.:
            #with torch.cuda.amp.autocast('cuda', torch.float32):
            with torch.cuda.amp.autocast():
                _mfcc_loss = self.mfcc_loss(estimate, target, lengths)
            _loss  += self.mfcc_loss_weight * _mfcc_loss
            if valid:
                d['valid_mfcc_loss'] = _mfcc_loss
            else:
                d['train_mfcc_loss'] = _mfcc_loss

        if self.plcpa_weight > 0.:
            with torch.cuda.amp.autocast():
                _plcpa_loss, _tsos = self.plcpa_loss(estimate, target, lengths)
                _loss += self.plcpa_weight * _plcpa_loss
            
            if valid:
                d['valid_plcpa_loss'] = _plcpa_loss
                d['valid_tsos'] = _tsos
            else:
                d['train_plcpa_loss'] = _plcpa_loss
                d['train_tsos'] = _tsos
                
        if self.lfcc_loss_weight > 0.:
            #with torch.cuda.amp.autocast('cuda', torch.float32):
            with torch.cuda.amp.autocast():
                _lfcc_loss = self.lfcc_loss(estimate, target, lengths)
            _loss  += self.lfcc_loss_weight * _lfcc_loss
            if valid:
                d['valid_lfcc_loss'] = _lfcc_loss
            else:
                d['train_lfcc_loss'] = _lfcc_loss
        
        if self.stft_loss_weight > 0.:
            with torch.cuda.amp.autocast():
                _stft_loss1, _stft_loss2 = self.stft_loss(estimate, target)
            _stft_loss = _stft_loss1 + _stft_loss2
            if valid:
                d['valid_stft_loss'] = _stft_loss
            else:
                d['train_stft_loss'] = _stft_loss
            _loss += self.stft_loss_weight * _stft_loss

        if self.pesq_loss_weight > 0.:
            #with torch.cuda.amp.autocast('cuda', torch.float32):
            with torch.cuda.amp.autocast():
                _pesq_loss = torch.mean(self.pesq(target, estimate))
            if valid:
                d['valid_pesq_loss'] = _pesq_loss
            else:
                d['train_pesq_loss'] = _pesq_loss
            _loss += self.pesq_loss_weight * _pesq_loss

        if self.sdr_loss_weight > 0.:
            with torch.cuda.amp.autocast():
                #with torch.cuda.amp.autocast('cuda', torch.float32):
                _sdr_loss = torch.mean(self.sdr_loss(estimate, target))
            if valid:
                d['valid_sdr_loss'] = _sdr_loss
            else:
                d['train_sdr_loss'] = _sdr_loss
            _loss += self.sdr_loss_weight * _sdr_loss

        if self.stoi_loss_weight > 0.:
            #with torch.cuda.amp.autocast('cuda', torch.float32):
            with torch.cuda.amp.autocast():
                _stoi_loss = torch.mean(self.stoi_loss(estimate, target))
            if valid:
                d['valid_stoi_loss'] = _stoi_loss
            else:
                d['train_stoi_loss'] = _stoi_loss
            _loss += self.stoi_loss_weight * _stoi_loss

        return _loss

    def training_step(self, batch, batch_idx:int) -> Tensor:
        mixtures, sources, enrolls, lengths, speakers = batch

        xvec = self.spk_model(enrolls)
        _spk_loss = self.spk_loss(xvec)
        if self.normalize:
            xvec = F.normalize(xvec)
            
        src_hat = self.model(sources, xvec)
        _loss = self.compute_loss(src_hat, sources, lengths, valid=False)
        _loss += self.spk_loss_weight * _spk_loss

        self.log_dict({'train_loss': _loss})
        self.log_dict({'train_speaker_loss': _spk_loss})
        
        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mixtures, sources, enrolls, lengths, speakers = batch

        xvec = self.spk_model(enrolls)
        _spk_loss = self.spk_loss(xvec)
        if self.normalize:
            xvec = F.normalize(xvec)
        
        src_hat = self.model(sources, xvec)
        _loss = self.compute_loss(src_hat, sources, lengths, valid=False)
        _loss += self.spk_loss_weight * _spk_loss

        self.log_dict({'valid_loss': _loss})
        self.log_dict({'valid_speaker_loss': _spk_loss})
        
        return _loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return optimizer
