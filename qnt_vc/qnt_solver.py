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

    def compute_loss(self, outputs, targets, valid=False):
        _loss = self.ce_loss(outputs, targets)
        return _loss

    '''
        compute label error rate
    '''
    def compute_ler(self, outputs, targets, valid=False):
        output_indices = torch.argmax(outputs, dim=-1) # b c t
        _tgt = []
        for _t in targets:
            _tgt.append(rearrange(_t, 'c t f -> t c f'))
        _tgt = nn.utils.rnn.pad_sequence(_tgt, batch_first=True, padding_value=-1).to(outputs.device)
        _tgt = rearrange(_tgt, 'b t c f -> b c (t f)')
        num_labels = torch.sum(torch.where(_tgt >= 0, 1., 0.))
        num_hits = torch.sum(torch.where(outputs == targets, 1., 0.))
        ler = ((num_labels - num_hits)/num_labels).cpu().detach().numpy()[0]
        return ler
    
    def training_step(self, batch, batch_idx:int) -> Tensor:
        src, tgt, src_id, tgt_id = batch

        outputs = self.forward(src, tgt, src_id, tgt_id)
        _loss = self.compute_loss(rearrange(outputs, 'b c t f -> (b c t) f'), U.make_targets(tgt))
        self.log_dict({'train_loss': _loss})

        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        src, tgt, src_id, tgt_id = batch

        outputs = self.forward(src, tgt, src_id, tgt_id)
        _loss = self.compute_loss(rearrange(outputs, 'b c t f -> (b c t) f'), U.make_targets(tgt))
        _ler = self.compute_ler(rearrange(outputs, tgt))

        self.log_dict({'valid_loss': _loss})
        self.log_dict({'valid_ler': _ler})

        return _loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return optimizer

