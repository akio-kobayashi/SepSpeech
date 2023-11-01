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

    def compute_ar_loss(self, outputs, targets, valid=False):
        _loss = self.ce_loss(ar_outputs, _targets)
        return _loss

    def training_step(self, batch, batch_idx:int) -> Tensor:
        src, tgt, src_id, tgt_id = batch

        outputs = self.forward(src, tgt, src_id, tgt_id)
        _loss = self.compute_loss(outputs, tgt)
        self.log_dict({'train_loss': _loss})

        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        src, tgt, src_id, tgt_id = batch

        outputs = self.forward(src, tgt, src_id, tgt_id)
        _loss = self.compute_loss(outputs, tgt)
        self.log_dict({'valid_loss': _loss})

        return _loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return optimizer

