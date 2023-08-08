# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from typing import Union, List

'''
    Cross Entropy Loss for Multi-Channel Quantized Data Stream
'''
class MCQCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, channel_weights=None, ignore_index=-1) -> None:
        super().__init__(ignore_index = ignore_index, reduction='none')

        self.channel_weights = channel_weights
        self.ignore_index = ignore_index
        
    def forward(self, preds:Tensor, targets:Tensor, lengths:Union[Tensor, List]) -> Tensor:
        B, C, T = targets.shape
        if isinstance(lengths, Tensor):
            lengths = lengths.cpu().detach().numpy().to_list()
        for b in range(B):
            targets[:, :, lengths[b]:] = self.ignore_index
        
        preds = rearrange(preds, 'b c t h -> (b c t) h')
        targets = rearrange(targets, 'b c t h -> (b c t) h')

        entropy = super.forward(preds, targets)
        entropy = rearrange(entropy, '(b c t) h -> c (b t h)', c=C)
        if self.channel_weights is not None:
            entropy = entropy * self.channel_weights
        entropy = torch.mean(entropy)
        
        return entropy
    
class CELoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=-1) -> None:
        super().__init__(self, ignore_index = ignore_index)

        self.ignore_index = ignore_index
        
    def forward(self, preds:Tensor, targets:Tensor, lengths:Union[Tensor, List]) -> Tensor:
        B, C = targets.shape
        if isinstance(lengths, Tensor):
            lengths = lengths.cpu().detach().numpy().to_list()
        for b in range(B):
            targets[:, :, lengths[b]:] = self.ignore_index
        
        entropy = super.forward(preds, targets)
        
        return entropy