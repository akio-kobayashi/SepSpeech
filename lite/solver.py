import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from models.unet import UNet
from models.unet2 import UNet2
from models.conv_tasnet import ConvTasNet
from loss.stft_loss import MultiResolutionSTFTLoss
from loss.pesq_loss import PesqLoss
from loss.sdr_loss import NegativeSISDR
from loss.stoi_loss import NegSTOILoss
from typing import Tuple

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
        else:
            raise ValueError('wrong parameter: '+config['model_type'])

        #self.lambda1 = config['loss']['lambda1']
        #self.lambda2 = config['loss']['lambda2']

        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

        self.stft_loss = self.pesq_loss = self.stoi_loss = self.sdr_loss = None
        self.stft_loss_weight = self.pesq_loss_weight = self.stoi_loss_weight = self.sdr_loss_weight = 0.
        if config['loss']['stft']['use']:
            self.stft_loss = MultiResolutionSTFTLoss()
            self.stft_loss_weight = config['loss']['stft']['weight']
        if config['loss']['pesq']['use']:
            self.pesq_loss = PesqLoss(factor=1.,
                                      sample_rate=16000)
            self.pesq_loss_weight = config['loss']['pesq']['weight']
        if config['loss']['stoi_loss']['use']:
            self.stoi_loss = NegSTOILoss()
            self.stoi_loss_weight = config['loss']['stoi']['weight']
        if config['loss']['sdr_loss']['use']:
            self.sdr_loss = NegativeSISDR()
            self.sdr_loss_weight = config['loss']['sdr']['weight']

        self.save_hyperparameters()

    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        return self.model(mix, enr)

    def compute_loss(self, estimate, target, valid=False):
        d = {}
        _ce_loss = self.ce_loss(estimate, target)
        _loss = self.ce_loss_weight * _ce_loss

        if valid:
            d['valid_ce_loss'] = _ce_loss
        else:
            d['train_ce_loss'] = _ce_loss

        if self.stft_loss:
            _stft_loss1, _stft_loss2 = self.stft_loss(estimate, target)
            _stft_loss = _stft_loss1 + _stft_loss2
            if valid:
                d['valid_stft_loss'] = _stft_loss
            else:
                d['train_stft_loss'] = _stft_loss
            _loss += self.stft_loss_weight * _stft_loss

        if self.pesq_loss:
            with torch.cuda.amp.autocast('cuda', torch.float32):
                _pesq_loss = torch.mean(self.pesq(target, estimate))
            if valid:
                d['valid_pesq_loss'] = _pesq_loss
            else:
                d['train_pesq_loss'] = _pesq_loss
            _loss += self.pesq_loss_weight * _pesq_loss

        if self.sdr_loss:
            with torch.cuda.amp.autocast('cuda', torch.float32):
                _sdr_loss = torch.mean(self.sdr_loss(estimate, target))
            if valid:
                d['valid_sdr_loss'] = _sdr_loss
            else:
                d['train_sdr_loss'] = _sdr_loss
            _loss += self.sdr_loss_weight * _sdr_loss

        if self.stoi_loss:
            with torch.cuda.amp.autocast('cuda', torch.float32):
                _stoi_loss = torch.mean(self.stoi_loss(estimate, target))
            if valid:
                d['valid_stoi_loss'] = _stoi_loss
            else:
                d['train_stoi_loss'] = _stoi_loss
            _loss += self.stoi_loss_weight * _stoi_loss

        if valid:
            d['valid_loss'] = _loss
        else:
            d['train_loss'] = _loss
        self.log_dict(d)

        return _loss

    def training_step(self, batch, batch_idx:int) -> Tensor:
        mixtures, sources, enrolls, lengths, speakers = batch

        src_hat, spk_hat = self.forward(mixtures, enrolls)
        _loss = self.compute_loss(src_hat, sources)

        return _loss

    '''
    def train_epoch_end(outputs:Tensor):
        #agv_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #tensorboard_logs={'loss': agv_loss}
        #return {'avg_loss': avg_loss, 'log': tensorboard_logs}
    '''

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mixtures, sources, enrolls, lengths, speakers = batch

        src_hat, spk_hat = self.forward(mixtures, enrolls)
        _loss = self.compute_loss(src_hat, sources, valid=True)

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
