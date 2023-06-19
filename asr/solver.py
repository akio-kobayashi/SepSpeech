import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple
from model import ASRModel, CELoss, CTCLoss
from transducer import ASRTransducer

class LitASR(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config

        self.model = ASRTransducer(config)
        #self.model = ASRModel(config)
        #self.ce_loss = CELoss()
        #self.ctc_loss = CTCLoss()
        self.lr = config['optimizer']['lr']

        #self.ce_weight = config['model']['ce_weight']
        
        self.save_hyperparameters()

    def forward(self, inputs:Tensor, labels:Tensor,
                input_lengths:list, label_lengths:list) -> Tensor:
        _loss = self.model(inputs, labels, input_lengths, label_lengths)
        return _loss
        #y, _ctc = self.model(inputs, labels, input_lengths, label_lengths)
        #return y, _ctc

    def training_step(self, batch, batch_idx:int) -> Tensor:
        inputs, labels, input_lengths, label_lengths, _ = batch

        #_pred, _ctc_loss = self.forward(inputs, labels, input_lengths, label_lengths)
        #labels_out = labels[:, 1:]
        #label_lengths = [l -1 for l in label_lengths]
        #_ce_loss = self.ce_loss(_pred, labels_out, label_lengths)
        #self.log_dict({'train_ce_loss': _ce_loss})
        #self.log_dict({'train_ctc_loss': _ctc_loss})
        #_loss = self.ce_weight * _ce_loss + _ctc_loss
        _loss = self.forward(inputs, labels, input_lengths, label_lengths)
        self.log_dict({'train_loss': _loss})

        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        inputs, labels, input_lengths, label_lengths, _ = batch

        #_pred, _ctc_loss = self.forward(inputs, labels, input_lengths, label_lengths)
        #labels_out = labels[:, 1:]
        #label_lengths = [l -1 for l in label_lengths]
        #_ce_loss = self.ce_loss(_pred, labels_out, label_lengths)
        #self.log_dict({'valid_ce_loss': _ce_loss})
        #self.log_dict({'valid_ctc_loss': _ctc_loss})
        #_loss = self.ce_weight * _ce_loss + _ctc_loss
        _loss = self.forward(inputs, labels, input_lengths, label_lengths)
        self.log_dict({'valid_loss': _loss})

        #outputs, _ = self.model.model.transcribe(inputs, torch.tensor(input_lengths).cuda())
        #print(outputs.shape)
        
        outputs = outputs.cpu().detach().numpy()
        
        return _loss

    def configure_optimizers(self):
        #optimizer=torch.optim.Adam(self.parameters(),
        #                           **self.config['optimizer'])
        optimizer = torch.optim.RAdam(
            self.parameters(),
            **self.config['optimizer'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            **self.config['scheduler']
        )
        return [optimizer], [scheduler]
        #return optimizer

    def decode(self, inputs:Tensor, input_lengths:list) -> list:
        decoded = self.model.decode(inputs, input_lengths)
        return decoded
    
