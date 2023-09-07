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
from loss.stft_loss import MultiResolutionSTFTLoss
from loss.pesq_loss import PesqLoss
from loss.sdr_loss import NegativeSISDR
from loss.stoi_loss import NegSTOILoss
from typing import Tuple
from einops import rearrange

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

        # clustering loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.ce_loss_weight = config['loss']['ce_loss']['weight']

        # Mean Absolute Error (temporal domain)
        self.l1_loss = nn.L1Loss()
        self.l1_loss_weight = config['loss']['l1_loss']['weight']

        self.stft_loss = self.pesq_loss = self.stoi_loss = self.sdr_loss = None
        self.stft_loss_weight = self.pesq_loss_weight = self.stoi_loss_weight = self.sdr_loss_weight = 0.
        if config['loss']['stft_loss']['use']:
            self.stft_loss = MultiResolutionSTFTLoss()
            self.stft_loss_weight = config['loss']['stft_loss']['weight']
        if config['loss']['pesq_loss']['use']:
            self.pesq_loss = PesqLoss(factor=1.,
                                      sample_rate=16000)
            self.pesq_loss_weight = config['loss']['pesq_loss']['weight']
        if config['loss']['stoi_loss']['use']:
            self.stoi_loss = NegSTOILoss()
            self.stoi_loss_weight = config['loss']['stoi_loss']['weight']
        if config['loss']['sdr_loss']['use']:
            self.sdr_loss = NegativeSISDR()
            self.sdr_loss_weight = config['loss']['sdr_loss']['weight']

        self.ctc_loss=None
        if config['ctc']['use']:
            self.ctc_loss = nn.CTCLoss()
            self.ctc_weight = config['ctc']['weight']

        self.save_hyperparameters()

    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        return self.model(mix, enr) # est, est_spk, ctc

    def compute_loss(self, estimate, target, estimate_spk, target_spk, valid=False):
        d = {}
        _ce_loss = self.ce_loss(estimate_spk, target_spk)
        _loss = self.ce_loss_weight * _ce_loss

        _l1_loss = self.l1_loss(estimate, target)
        _loss  += self.l1_loss_weight * _l1_loss
        
        if valid:
            d['valid_ce_loss'] = _ce_loss
            d['valid_l1_loss'] = _l1_loss
        else:
            d['train_ce_loss'] = _ce_loss
            d['train_l1_loss'] = _l1_loss

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
        if self.ctc_loss is None:
            mixtures, sources, enrolls, lengths, speakers = batch
        else:
            mixtures, sources, enrolls, lengths, speakers, labels, target_lengths = batch

        logits = None
        if self.config['model_type'] == 'unet':
            src_hat, spk_hat, logits = self.forward(mixtures, enrolls)
        else:
            src_hat, spk_hat, _ = self.forward(mixtures, enrolls)
        _loss = self.compute_loss(src_hat, sources, spk_hat, speakers, valid=False)

        if logits is not None:
            valid_lengths = torch.tensor([ self.model.valid_length_ctc(l) for l in lengths ])
            target_lengths = torch.tensor(target_lengths)
            logprobs = F.log_softmax(logits)
            logprobs = rearrange('b t c -> t b c')
            with torch.cuda.amp.autocast('cuda', torch.float32):
                _loss += self.ctc_weight * self.ctc_loss(logprobs, valid_lengths, target_lengths)

        return _loss

    '''
    def train_epoch_end(outputs:Tensor):
        #agv_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #tensorboard_logs={'loss': agv_loss}
        #return {'avg_loss': avg_loss, 'log': tensorboard_logs}
    '''

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.ctc_loss is None:
            mixtures, sources, enrolls, lengths, speakers = batch
        else:
            mixtures, sources, enrolls, lengths, speakers, labels, target_lengths = batch

        logits = None
        if self.config['model_type'] == 'unet':
            src_hat, spk_hat, logits = self.forward(mixtures, enrolls)
            if self.ctc_loss is not None:
                valid_lengths = [ self.model.valid_length_ctc(l) for l in lengths ]
        else:
            src_hat, spk_hat, _ = self.forward(mixtures, enrolls)
        _loss = self.compute_loss(src_hat, sources, spk_hat, speakers, valid=True)

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
