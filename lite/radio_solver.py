import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from models.radio_unet import UNetRadio
#from models.conv_tasnet import ConvTasNet
from loss.stft_loss import MultiResolutionSTFTLoss
from loss.pesq_loss import PesqLoss
from loss.stoi_loss import NegSTOILoss
from loss.sdr_loss import NegativeSISDR
from typing import Tuple
import utils.cooldown

'''
 PyTorch Lightning 用 solver
'''
class LitDenoiser(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config
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
            
        self.save_hyperparameters()

    def forward(self, mix:Tensor) -> Tensor:
        return self.model(mix)

    def compute_loss(self, estimates, targets, valid=False)->Tuple[Tensor, dict]:
        prefix='valid_' if valid is True else ''
        
        d = {}
        _loss = 0.
        if self.stft_loss is not None:
            _stft_loss1, _stft_loss2 = self.stft_loss(src_hat, sources)
            _loss += self.stft_loss_weight * (_stft_loss1 + _stft_loss2)
            d[prefix+'stft_loss'] = _stft_loss1 + _stft_loss2]
        if self.pesq_loss is not None:
            _pesq_loss = self.pesq_loss(sources, src_hat)
            _loss += self.pesq_loss_weight * _pesq_loss
            d[prefix+'pesq_loss'] = _pesq_loss
        if self.sdr_loss is not None:
            _sdr_loss = self.sdr_loss(src_hst, sources)
            _loss += self.sdr_loss_weight * _sdr_loss
            d[prefix+'sdr_loss'] = _sdr_loss
        if self.stoi_loss is not None:
            _stoi_loss = self.stoi_loss(src_hat, sources)
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
        _loss, d = self.compute_loss(src_hat, sources, valid=False)
        self.log_dict(d)

        utils.cooldown.coolGPU()

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
        _loss, d = self.compute_loss(src_hat, sources, valid=True)
        self.log_dict(d)

        return _loss

    '''
    def on_validation_epoch_end(outputs:Tensor):
        #agv_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #tensorboard_logs={'val_loss': agv_loss}
        #return {'avg_loss': avg_loss, 'log': tensorboard_logs}
    '''
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return optimizer
    
    def get_model(self):
        return self.model
