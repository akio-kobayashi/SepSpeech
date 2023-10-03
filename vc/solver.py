import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from loss.mfcc_loss import MFCCLoss, LFCCLoss
from loss.stft_loss import MultiResolutionSTFTLoss
from typing import Tuple
from einops import rearrange
from vc.vcmodel import VoiceConversionModel

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='sum')

    def forward(self, preds, targets, lengths):
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[:, :lengths[b]] = 1.
        return self.loss(preds * mask, targets * mask) / torch.sum(mask)
    
'''
 PyTorch Lightning 用 solver
'''
class LitVoiceConversion(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config
        self.model = VoiceConversionModel(config)

        # Mean Absolute Error (temporal domain)
        self.l1_loss = L1Loss()
        self.l1_loss_weight = config['loss']['l1_loss']['weight']

        self.diag_loss_weight = config['loss']['diag']['weight']

        self.stft_loss = None
        self.stft_loss_weight = 0.
        if config['loss']['stft_loss']['use']:
            self.stft_loss = MultiResolutionSTFTLoss()
            self.stft_loss_weight = config['loss']['stft_loss']['weight']

        self.save_hyperparameters()

    def forward(self, src:Tensor, tgt:Tensor, src_embed:Tensor, tgt_embed:Tensor, src_len, tgt_len):
        return self.model(src, tgt, src_embed, tgt_embed, src_len, tgt_len)

    def compute_loss(self, estimates, sources, targets, src_lengths, tgt_lengths, valid=False):
        d = {}
       
        _loss = 0.

        with torch.cuda.amp.autocast():
            _l1_loss=self.l1_loss(estimates, targets, tgt_lengths)
            _loss += self.l1_loss_weight * _l1_loss
        if valid:
            d['valid_11_loss'] = _l1_loss
        else:
            d['train_l1_loss'] = _l1_loss

        B, S, _ = sources.shape
        _, T, _ = targets.shape
        with torch.cuda.amp.autocast():
            _diag_loss = self.model.diagonal_attention_loss(B, T, S, src_lengths, tgt_lengths)
            _loss += self.diag_loss_weight * _diag_loss
        if valid:
            d['valid_diag_loss'] = _diag_loss
        else:
            d['train_diag_loss'] = _diag_loss
        
        with torch.cuda.amp.autocast():
            _stft_loss1, _stft_loss2 = self.stft_loss(estimate, target)
            _stft_loss = _stft_loss1 + _stft_loss2
        if valid:
            d['valid_stft_loss'] = _stft_loss
        else:
            d['train_stft_loss'] = _stft_loss
        _loss += self.stft_loss_weight * _stft_loss

        if valid:
            d['valid_loss'] = _loss
        else:
            d['train_loss'] = _loss
        self.log_dict(d)

        return _loss

    def training_step(self, batch, batch_idx:int) -> Tensor:
        sources, targets, = batch

        logits = None
        if self.config['model_type'] == 'unet':
            src_hat, spk_hat, logits = self.forward(mixtures, enrolls)
        else:
            src_hat, spk_hat, _ = self.forward(mixtures, enrolls)
        _loss = self.compute_loss(src_hat, sources, lengths, spk_hat, speakers, valid=False)

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
        _loss = self.compute_loss(src_hat, sources, lengths, spk_hat, speakers, valid=True)

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
