import numpy as np
import copy
import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from einops import rearrange

class ASRModel(nn.Module):
    def __init__(self):
        self.preprocess = 
        self.conformer=
        
    def forward(self, x):
        
'''
 RNN Transducer
    from https://github.com/HawkAaron/E2E-ASR/blob/91d6b96bd605a9bc8e62b4a5903ec2056fa0f88d/model.py#L37
'''
class Transducer(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size=144,
                 cell_size=320,
                 dropout=.5,
                 blank=0)
        super(Transducer, self).__init__()

        self.blank = blank
        self.vocab_size = vocab_size
        
        self.fc0 = nn.Linear(hidden_size, cell_size)

        self.embed = nn.Embedding(vocab_size, vocab_size-1, padding_idx=blank)
        self.embed.weight.data[1:] = torch.eye(vocab_size-1)
        self.embed.weight.requires_grad = False
        self.decoder = nn.LSTM(vocab_size-1, cell_size, 1,
                            batch_first=True, dropout=dropout)
        
        self.fc1 = nn.Linear(2*cell_size, cell_size)
        self.fc2 = nn.Linear(cell_size, vocab_size)

    def joint(self, f, g):
        ''' `f`: encoder output (B,T,U,2H)
        `g`: decoder lstm output (B,T,U,H)
        NOTE f and g must have the same size except the last dim'''
        dim = len(f.shape) - 1
        out = torch.cat((f, g), dim=dim)
        out = torch.tanh(self.fc1(out))
        return self.fc2(out)

    def forward(self, xs, ys, xlen, ylen):
        
        # concat first zero
        #zero = autograd.Variable(torch.zeros((ys.shape[0], 1)).long()).cuda()
        zero = torch.zeros(ys.shape[0], 1, dtype=torch.int32).cuda()
        ymat = torch.cat((zero, ys), dim=1)
        # forward pm
        ymat = self.embed(ymat)
        ymat, _ = self.decoder(ymat)
        xs = xs.unsqueeze(dim=2)
        ymat = ymat.unsqueeze(dim=1)

        # expand
        sz = [max(i, j) for i, j in zip(xs.size()[:-1], ymat.size()[:-1])]
        xs = xs.expand(torch.Size(sz+[xs.shape[-1]]));
        # f : encoder lstm output (B,T,U,2H)
        # g : decoder lstm output (B,T,U,H)
        ymat = ymat.expand(torch.Size(sz+[ymat.shape[-1]]))
        out = self.joint(xs, ymat)

        loss = torchaudio.functional.rnnt_loss(out,
                                               ys.cuda(),
                                               xlen.cuda(),
                                               ylen.cuda(),
                                               blank=self.blank,
                                               )

        return loss

    def greedy_decode(self, x, ff=True):
        vy = autograd.Variable(torch.LongTensor([0]), volatile=True).view(1,1)
        # vector preserve for embedding
        if x.is_cuda: vy = vy.cuda()
        y, h = self.decoder(self.embed(vy)) # decode first zero
        y_seq = []; logp = 0
        for i in range(x.shape[1]):
            ytu = self.joint(torch.unsqueeze(x[0][i],0), y[0])
            out = F.log_softmax(ytu, dim=1)
            p, pred = torch.max(out, dim=1) # suppose blank = -1
            pred = int(pred); logp += float(p)
            if pred != self.blank:
                y_seq.append(pred)
                vy.data[0][0] = pred # change pm state
                y, h = self.decoder(self.embed(vy), h)
        return y_seq
