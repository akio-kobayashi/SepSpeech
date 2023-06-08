import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple
from model import ASRModel, CELoss

class LitASR(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config

        self.model = ASRModel(config)
        self.ce_loss = CELoss()
        self.lr = config['optimizer']['lr']
        
        self.save_hyperparameters()

    def forward(self, inputs:Tensor, labels:Tensor, input_lengths:list, label_lengths:list) -> Tensor:
        return self.model(inputs, labels, input_lengths, label_lengths)

    def training_step(self, batch, batch_idx:int) -> Tensor:
        inputs, labels, input_lengths, label_lengths, _ = batch

        _pred = self.forward(inputs, labels, input_lengths, label_lengths)
        labels_out = labels[:, 1:]
        #label_lengths -= 1
        label_lengths = [l -1 for l in label_lengths]
        _loss = self.ce_loss(_pred, labels_out, label_lengths)

        self.log_dict({'train_loss': _loss})

        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        inputs, labels, input_lengths, label_lengths, _ = batch

        _pred = self.forward(inputs, labels, input_lengths, label_lengths)
        labels_out = labels[:, 1:]
        #label_lengths -= 1
        label_lengths = [l -1 for l in label_lengths]
        _loss = self.ce_loss(_pred, labels_out, label_lengths)
        self.log_dict({'valid_loss': _loss})

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

    def decode(self, input:Tensor, input_length:int) -> list:
        decoded = self.model.greedy_decode(input, input_length)
        return decoded
