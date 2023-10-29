import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
from einops import rearrange
import qnt_vc.qnt_utils as U
from qnt_vc.qnt_model import QntVoiceConversionModel

class LitVoiceConversion(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QntVoiceConversionModel(config['transformer'])

        # Mean Absolute Error (temporal domain)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.save_hyperparameters()

    def forward(self, src:Tensor, tgt:Tensor, src_id:Tensor, tgt_id:Tensor):
        return self.model(src, tgt, src_id, tgt_id)

    def compute_ar_loss(self, ar_outputs, targets, valid=False):
        _targets = []
        for b in range(len(targets)):
            _tgt = rearrange(targets[b][0,:,:], '(c t) f -> c t f', c=1)
            _targets.append(rearrange(U.append_special_tokens(_tgt, bos=False), 'c t f -> t c f'))
        _targets = nn.utils.rnn.pad_sequence(_targets, batch_first=True, padding_value=-1).to('cuda')
        _targets = rearrange(_targets, 'b t c f -> b c t f')            
        _loss = self.ce_loss(ar_outputs, _targets)
        if valid is True:
            self.log_dict({'valid_ar_loss': _loss})
        else:
            self.log_dict({'train_ar_loss': _loss})
        return _loss
    
    def compute_nar_loss(self, nar_outputs, targets, valid=False):
        _targets = []
        for b in range(len(targets)):
            _tgt = targets[b][1:,:,:]
            _targets.append(rearrange(_tgt, 'c t f -> t c f'))
        _targets = nn.utils.rnn.pad_sequence(_targets, batch_first=True, padding_value=-1).to('cuda')
        _targets = rearrange(_targets, 'b t c f -> b c t f')            
        _loss = self.ce_loss(nar_outputs, _targets)
        if valid is True:
            self.log_dict({'valid_nar_loss': _loss})
        else:
            self.log_dict({'train_nar_loss': _loss})
        return _loss

    def training_step(self, batch, batch_idx:int) -> Tensor:
        src, tgt, src_id, tgt_id = batch

        ar_outputs, nar_outputs = self.forward(src, tgt, src_id, tgt_id)
        _loss = self.compute_ar_loss(ar_outputs, tgt)
        _loss += self.compute_nar_loss(nar_outputs, tgt)
        self.log_dict({'train_loss': _loss})

        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        src, tgt, src_id, tgt_id = batch

        ar_outputs, nar_outputs = self.forward(src, tgt, src_id, tgt_id)
        _loss = self.compute_ar_loss(ar_outputs, tgt)
        _loss += self.compute_nar_loss(nar_outputs, tgt)
        self.log_dict({'valid_loss': _loss})

        return _loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return optimizer

