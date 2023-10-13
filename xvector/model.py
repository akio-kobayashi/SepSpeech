import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xvector.tdnn import TDNN
from xvector.adacos import AdaCos
from models.e3net import LearnableEncoder
from einops import rearrange

def get_mask(tensor, lengths):
    mask = np.zeros((tensor.shape[0], tensor.shape[1], 1))

    for n in range(len(lengths)):
        l=lengths[n]
        if l > tensor.shape[1]:
            l=tensor.shape[1]
        mask[n:, :l, :] = 1.

    return torch.from_numpy(mask.astype(np.float32)).clone()

def stats_with_mask(tensor, mask):
    mean = torch.div(torch.sum(tensor*mask, dim=1, keepdim=True),torch.sum(mask, dim=1, keepdim=True))
    var = torch.square(tensor-mean)
    var = torch.sum(var*mask, dim=1, keepdim=True)
    var = torch.div(var, torch.sum(mask, dim=1, keepdim=True)+1.0e-8)
    std = torch.sqrt(var)
    if mean.shape[0] < 2:
        mean, std = mean.squeeze(), std.squeeze()
        mena,std = mean.unsqueeze(0), std.unsqueeze(0)
    else:
        mean, std = mean.squeeze(), std.squeeze()
    return mean, std

class X_vector(nn.Module):
    def __init__(self, input_dim = 40, dim=512, dim1=1500, dim2=3000, output_dim=256):
        super(X_vector, self).__init__()

        self.encoder = LearnableEncoder(chout=input_dim)
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=dim, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=dim, output_dim=dim, context_size=3, dilation=2,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=dim, output_dim=dim, context_size=3, dilation=3,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=dim, output_dim=dim, context_size=4, dilation=4,dropout_p=0.5)
        self.segment5 = nn.Linear(dim, dim1)
        self.segment6 = nn.Linear(dim2, dim)
        self.segment7 = nn.Linear(dim, output_dim)

        #self.criterion = AdaCos(output_dim, class_num)

    def forward(self, inputs, speakers=None, lengths=None):
        # (B, T) -> (B, 1, T)
        inputs = rearrange(inputs, '(b c) t -> b c t', c=1)
        encode = self.encoder(inputs)
        # (B, C, T) -> (B, T, C)
        encode = rearrange(encode, 'b c t -> b t c')
        tdnn1_out = self.tdnn1(encode)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        out = self.segment5(tdnn4_out) # (b, t, f)
        if lengths is None:
            mean = torch.mean(out,1) # (b, f)
            std = torch.std(out,1) # (b, f)
        else:
            mask=get_mask(out, lengths).cuda()
            mean, std = stats_with_mask(out, mask)
        stat_pooling = torch.cat((mean,std),1) # (b, fx2)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)

        return x_vec
        #if xvec_outputs is True:
        #    x_vec = F.normalize(x_vec)
        #    return x_vec

        #loss, logits = self.criterion(x_vec, speakers)
        #return loss, logits

    '''
    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file, map_location=torch.device('cpu')), strict=False)

    def get_speaker_matrix(self):
        # return (speaker x num_feats)-matrix
        return self.criterion.get_speaker_matrix()
    '''
