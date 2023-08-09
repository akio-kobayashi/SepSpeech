import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from lm.qnt_unet import QntUNet
from lm.qnt_loss import MCQCrossEntropyLoss, CELoss
from typing import Tuple

class LitModel(pl.LightningModule):
    def __init__(self, config:dict, model_type=None) -> None:
        super().__init__()
        
        if model_type is None:
            model_type = config['model_type']
        self.config = config
        if model_type == 'qnt_unet':
            self.model = QntUNet(config)
        else:
            raise ValueError('wrong parameter: '+config['model_type'])


        self.ce_loss = CELoss()
        self.ce_loss_weight = config['ce_loss_weight']
        self.mcq_loss = MCQCrossEntropyLoss()
        self.mcq_loss_weight = config['mcq_loss_weight']
        
        self.save_hyperparameters()

    def forward(self, mixures:Tensor, enrolls:Tensor) -> Tuple[Tensor, Tensor]:
        # tensor shape = (b q t h)
        estimates, speaker_logits = self.model(mixtures, enrolls)

        return estimates, speaker_logits

    def training_step(self, batch, batch_idx:int) -> Tensor:
        mixtures, sources, enrolls, lengths, speakers = batch
        
        estimates, speaker_logits = self.forward(mixtures, enrolls)

        _mcq_loss = self.mcq_loss(estimates, sources, lengths)
        _ce_loss = self.ce_loss(speaker_logits, speakers)
        
        _loss = self.mcq_loss_weight * _mcq_loss + self.ce_weight * _ce_loss

        self.log_dict({'train_loss': _loss,
                       'train_mcq_loss': _mcq_loss,
                       'train_ce_loss': _ce_loss})
        return _loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mixtures, sources, enrolls, lengths, speakers = batch

        estimates, speaker_logits = self.forward(mixtures, enrolls)

        _mcq_loss = self.mcq_loss(estimates, sources, lengths)        
        _ce_loss = self.ce_loss(speaker_logits, speakers)
        
        _loss = self.mcq_loss_weight * _mcq_loss + self.ce_weight * _ce_loss

        self.log_dict({'valid_loss': _loss,
                       'valid_mcq_loss': _mcq_loss,
                       'valid_ce_loss': _ce_loss})
        return _loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(),
            **self.config['optimizer'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            **self.config['scheduler']
        )
        return [optimizer], [scheduler]
