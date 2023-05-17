import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple

class LitASR(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config

        model = ASRModel(config)
        
        self.lambda1 = config['loss']['lambda1']
        self.lambda2 = config['loss']['lambda2']

        self.save_hyperparameters()

    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        return self.model(mix, enr)

    def training_step(self, batch, batch_idx:int) -> Tensor:
        mixtures, sources, enrolls, lengths, speakers = batch

        src_hat, spk_hat = self.forward(mixtures, enrolls)
        _stft_loss1, _stft_loss2 = self.stft_loss(src_hat, sources)
        _stft_loss = _stft_loss1 + _stft_loss2
        _ce_loss = self.ce_loss(spk_hat, speakers)
        _loss = self.lambda1 * _stft_loss + self.lambda2 * _ce_loss
        self.log_dict({'train_loss': _loss,
                       'train_stft_loss': _stft_loss,
                       'train_ce_loss': _ce_loss})

        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mixtures, sources, enrolls, lengths, speakers = batch

        src_hat, spk_hat = self.forward(mixtures, enrolls)
        _stft_loss1, _stft_loss2 = self.stft_loss(src_hat, sources)
        _stft_loss = _stft_loss1 + _stft_loss2
        _ce_loss = self.ce_loss(spk_hat, speakers)
        _loss = self.lambda1 * _stft_loss + self.lambda2 * _ce_loss
        self.log_dict({'valid_loss': _loss,
                       'valid_stft_loss': _stft_loss,
                       'valid_ce_loss': _ce_loss})

        return _loss

       def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return optimizer
    