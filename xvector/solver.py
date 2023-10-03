import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from xvector.adacs import AdaCos 
from xvector.model import X_vector
from xvector.generator import SpeakerDataset
import numpy as np
import argparse
#from sklearn.metrics import accuracy_score
import pytorch_lightning as pl

class LitXVector(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.model = X_vector(
            input_dim = config['input_dim'],
            dim = config['dim'],
            dim1 = config['dim1'],
            dim2 = config['dim2'],
            output_dim = config['output_dim']
        )
        self.metric = AdaCos(config['output_dim'], config['class_num'])
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, inputs, speakers, lengths):
        return self.model(inputs, speakers, lengths)
        
    def training_step(self, batch, batch_index):
        inputs, lengths, speakers = batch
            
        _xvec = self.forward(inputs, speakers, lengths)
        _metric, _ = self.metric(_xvec, speakers)
        _loss = self.loss(_metric, speakers)

        self.log_dict({'train_loss': _loss})

        return _loss
    
    def validation_step(self, batch, batch_idx):
        inputs, lengths, speakers = batch
            
        _xvec = self.forward(inputs, speakers, lengths)
        _metric, _ = self.metric(_xvec, speakers)
        _loss = self.loss(_metric, speakers)

        self.log_dict({'valid_loss': _loss})

        return _loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return optimizer
    

