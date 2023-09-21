import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange

class OnnxModel(nn.Module):
    def __init__(self, mdl) -> None:
        super().__init__()
        self.model = mdl
        
    def forward(self, mix:Tensor, enr:Tensor) -> Tuple[Tensor, Tensor]:
        est, est_spk, _ = self.model(mix, enr) # est, est_spk, ctc
        return est, est_spk
    
