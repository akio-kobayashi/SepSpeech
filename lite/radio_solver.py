import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from models.radio_unet import UNetRadio
#from models.conv_tasnet import ConvTasNet
from models.fast_radio_unet import FastUNetRadio
from loss.stft_loss import MultiResolutionSTFTLoss
from loss.pesq_loss import PesqLoss
from loss.stoi_loss import NegSTOILoss
from loss.sdr_loss import NegativeSISDR
from loss.mfcc_loss import MFCCLoss
from typing import Tuple
import os,sys
#import utils.cooldown

class HuberLoss(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)
        
    def forward(self, preds, targets, lengths):
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :lengths[b]] = 1
        return self.loss(preds*mask, targets*mask)
    
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
        
    def forward(self, preds, targets, lengths):
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :lengths[b]] = 1
        return self.loss(preds*mask, targets*mask)
    
class LitDenoiser(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config
        if config['model_type'] == 'fast_unet':
            self.model = FastUNetRadio(config)
        else:
            self.model = UNetRadio(config)

        self.stft_loss = self.pesq_loss = self.sdr_loss = self.stoi_loss = None
        self.stft_loss_weight = self.pesq_loss_weight = self.sdr_loss_weight = self.stoi_loss_weight = 0.
        if config['loss']['stft']['use']:
            self.stft_loss = MultiResolutionSTFTLoss()
            self.stft_loss_weight = config['loss']['stft']['weight']
        if config['loss']['pesq']['use']:
            self.pesq_loss = PesqLoss(factor=1., sample_rate=16000)
            self.pesq_loss_weight = config['loss']['pesq']['weight']
        if config['loss']['sdr']['use']:
            self.sdr_loss = NegativeSISDR()
            self.sdr_loss_weight = config['loss']['sdr']['weight']
        if config['loss']['stoi']['use']:
            self.stoi_loss = NegSTOILoss(sample_rate=16000, extended=config['loss']['stoi']['extended'])
            self.stoi_loss_weight = config['loss']['stoi']['weight']

        self.l1_loss = L1Loss()
        self.l1_loss_weight = config['loss']['l1_loss']['weight']

        self.mfcc_loss = MFCCLoss(config['loss']['mfcc'])
        self.mfcc_loss_weight = config['loss']['mfcc']['weight']
        
        self.dict_path = os.path.join(config['logger']['save_dir'], config['logger']['name'])
        self.dict_path = os.path.join(self.dict_path, 'version_'+str(config['logger']['version']))
        #self.dict_path = config['logger']['save_dir']
        self.valid_step_loss=[]
        self.valid_epoch_loss=[]
        
        self.save_hyperparameters()

    def forward(self, mix:Tensor) -> Tensor:
        return self.model(mix)

    def compute_loss(self, estimates, targets, lengths, valid=False)->Tuple[Tensor, dict]:
        prefix='valid_' if valid is True else ''
        
        d = {}

        if self.l1_loss_weight > 0.:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                _l1_loss = self.l1_loss(estimates, targets, lengths)
            d[prefix + 'l1_loss'] = _l1_loss
            _loss = self.l1_loss_weight * _l1_loss

        if self.mfcc_loss_weight > 0.:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                _mfcc_loss = self.mfcc_loss(estimates, targets, lengths)
            d[prefix + 'mfcc_loss'] = _mfcc_loss
            _loss = self.mfcc_loss_weight * _mfcc_loss
    
        if self.stft_loss is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                _stft_loss1, _stft_loss2 = self.stft_loss(estimates, targets)
                _loss += self.stft_loss_weight * (_stft_loss1 + _stft_loss2)
            d[prefix+'stft_loss'] = _stft_loss1 + _stft_loss2
        if self.pesq_loss is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                _pesq_loss = torch.mean(self.pesq_loss(targets, estimates))
            _loss += self.pesq_loss_weight * _pesq_loss
            d[prefix+'pesq_loss'] = _pesq_loss
        if self.sdr_loss is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                _sdr_loss = torch.mean(self.sdr_loss(estimates, targets))
            _loss += self.sdr_loss_weight * _sdr_loss
            d[prefix+'sdr_loss'] = _sdr_loss
        if self.stoi_loss is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                _stoi_loss = torch.mean(self.stoi_loss(estimates, targets))
            _loss += self.stoi_loss_weight * _stoi_loss
            d[prefix+'stoi_loss'] = _stoi_loss
        if valid is True:
            d['valid_loss'] = _loss
        else:
            d['train_loss'] = _loss
            
        return _loss, d
    
    def training_step(self, batch, batch_idx:int) -> Tensor:
        mixtures, sources, lengths = batch

        src_hat = self.forward(mixtures)
        _loss, d = self.compute_loss(src_hat, sources, lengths, valid=False)
        self.log_dict(d)

        #utils.cooldown.coolGPU()

        return _loss

    '''
    def on_train_epoch_end(outputs:Tensor):
        #agv_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #tensorboard_logs={'loss': agv_loss}
        #return {'avg_loss': avg_loss, 'log': tensorboard_logs}
    '''

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mixtures, sources, lengths = batch

        src_hat = self.forward(mixtures)
        _loss, d = self.compute_loss(src_hat, sources, lengths, valid=True)
        self.log_dict(d)
        self.valid_step_loss.append(_loss.item())
        
        return _loss

    def on_validation_epoch_end(self):
        _loss = np.mean(self.valid_step_loss)
        self.valid_epoch_loss.append(_loss)
        '''
        if self.current_epoch > 1 and np.min(self.valid_epoch_loss) == _loss:
            path = os.path.join(self.dict_path , 'model_epoch='+str(self.current_epoch))
            torch.save(self.model.to('cpu').state_dict(), path)
            self.model.to('cuda')
        path = os.path.join(self.dict_path, 'last.pt')
        torch.save(self.model.to('cpu').state_dict(), path)
        self.model.to('cuda')
        '''
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return optimizer
    
    def get_model(self):
        return self.model
