import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchaudio.models import Conformer
from torch import Tensor
from typing import Tuple
import networkx

class TokenEndNode(Object):
    def __init__(self, token:int, score:float, bkptr:int):
        self.entry = {'token': token, 'score': score, 'bkptr': bkptr}

class BeamSearch():
    def __init__(self):
        self.start_node = 0
        self.end_node = 1
        self.node_num = 2
        
    def step(self, curr_nodes):
        for nd in curr_nodes:
            history = []
            while True:
                history.append(self.G[nd]['token'])
                if self.G[nd]['bkptr'] == 1:
                    break
                nd = self.G[nd]['bkptr']
            with torch.no_grad():
                ys = torch.Tensor(history.reverse()).unqueeze(0)
                mask = generate_squre_subsequent_mask(ys.shape[1])
                z = self.dec_pe(self.dec_embed(ys))
                z = self.decoder(z, memory, tgt_mask=mask)
                z = F.log_softmax(self.fc(z), dim=-1)

