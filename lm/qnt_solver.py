import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from lm.qnt_ddpm import UNet
from lm.qnt_loss import MCQCrossEntropyLoss, CELoss
from lm.qnt_unet import QntEncoder, QntDecoder
from lm.qnt_modules import QntSpeakerNetwork
from typing import Tuple

class LitModel(pl.LightningModule):
    def __init__(self, config:dict, model_type=None) -> None:
        super().__init__()
        
        if model_type is None:
            model_type = config['model_type']
        self.config = config
        if model_type == 'unet':
            self.model = UNet(dim=config['dim'],
                               channels=config['channels'], 
                               resnet_block_groups=config['resnet_block_groups'])
        else:
            raise ValueError('wrong parameter: '+config['model_type'])

        self.encoder = QntEncoder(sym_dim=config['encodec_codebook_size'], 
                                  emb_dim=config['channels'], 
                                  in_channels=config['dim'], 
                                  out_channels=config['dim'], 
                                  kernel_size=config['endec_kernel_size']
                                  )
        self.decoder = QntDecoder(sym_dim=config['encodec_codebook_size'], 
                                  emb_dim=config['channels'], 
                                  in_channels=config['dim'], 
                                  out_channels=config['dim'], 
                                  kernel_size=config['endec_kernel_size']
                                  )
        self.speaker_embedder = QntSpeakerNetwork(width=config['channels'], 
                                                  in_channels=config['dim'], 
                                                  out_channels=config['dim'], 
                                                  kernel_size=3, 
                                                  stride=1, 
                                                  num_speakers=config['num_speakers']
                                                )

        self.ce_loss = CELoss()
        self.ce_loss_weight = config['ce_loss_weight']
        self.mcq_loss = MCQCrossEntropyLoss()
        self.mcq_loss_weight = config['mcq_loss_weight']
        
        self.save_hyperparameters()

    def forward(self, mixures:Tensor, enrolls:Tensor) -> Tuple[Tensor, Tensor]:
        mixtures = self.encoder(mixtures)
        enrolls, speaker_logits = self.speaker_embedder(self.encoder(enrolls))
        estimates = self.model(mixtures, enrolls)

        return estimates, speaker_logits

    def training_step(self, batch, batch_idx:int) -> Tensor:
        mixtures, sources, enrolls, lengths, speakers = batch

        '''
        mixtures = self.encoder(mixtures)
        enrolls, speaker_logits = self.speaker_embedder(self.encoder(enrolls))
        
        estimates = self.forward(mixtures, enrolls)
        '''
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

        '''
        mixtures = self.encoder(mixtures)
        enrolls, speaker_logits = self.speaker_embedder(self.encoder(enrolls))
        
        estimates = self.forward(mixtures, enrolls)
        '''
        
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
