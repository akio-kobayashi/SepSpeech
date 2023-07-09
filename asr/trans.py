# conformer-based model
import copy
import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from warprnnt_pytorch import RNNTLoss
from einops import rearrange
from loss import MaskedMSELoss
import math
import numpy as np
from model import Sequence, Subsampler, DownSampler

class TransTransducer(nn.Module):
    def __init__(self, device,
                 input_vocab_size,
                 embed_size,
                 vocab_size,
                 hidden_size=144,
                 cell_size=320,
                 num_layers=16,
                 num_heads=8,
                 dropout=.5,
                 blank=0):
        super(TransTransducer, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.embed_size = embed_size
        self.blank = blank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.loss = RNNTLoss()

        # NOTE encoder & decoder only use lstm
        # instead of RNN, just use conformer block
        encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.input_embed = nn.Embedding(self.input_vocab_size, hidden_size)
        self.fc0 = nn.Linear(hidden_size, cell_size) # (B, T, H)
        
        self.embed = nn.Embedding(vocab_size, vocab_size-1, padding_idx=blank)
        self.embed.weight.data[1:] = torch.eye(vocab_size-1)
        self.embed.weight.requires_grad = False
        self.decoder = nn.LSTM(vocab_size-1, cell_size, 1,
                                batch_first=True, dropout=0., bidirectional=False)

        self.fc1 = nn.Linear(2*cell_size, cell_size)
        self.fc2 = nn.Linear(cell_size, vocab_size)

    def valid_input_lengths(self, x):
        return x
    
    def joint(self, f, g):
        ''' `f`: encoder output (B,T,U,2H)
        `g`: decoder lstm output (B,T,U,H)
        NOTE f and g must have the same size except the last dim'''
        dim = len(f.shape) - 1
        out = torch.cat((f, g), dim=dim)
        out = torch.tanh(self.fc1(out))
        return self.fc2(out)

    def ff_encoder(self, xs):
        xs = self.input_embed(xs)
        xs = self.encoder(xs)
        xs = self.fc0(xs)

        return xs

    def forward(self, xs, ys, xlen, ylen, return_loss=True):
        xs = self.input_embed(xs)
        xs = self.encoder(xs)
        xs = self.fc0(xs) # (B, time//scale, hidden_size)

        # concat first zero
        zero = autograd.Variable(torch.zeros((ys.shape[0], 1)).long())
        if ys.is_cuda: zero = zero.cuda()
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

        if return_loss is False:
            return out

        if ys.is_cuda:
            xlen = xlen.cuda()
            ylen = ylen.cuda()
        loss = self.loss(out, ys.int(), xlen, ylen)

        return loss

    def greedy_decode(self, x, ff=True):
        if ff is True:
            if self.downsampler is not None:
                x = self.downsampler(x)
                x = self.encoder(x)
                x = self.fc0(x)

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
        return y_seq, -logp


    def beam_search(self, xs, W=10, prefix=False):
        '''''
        `xs`: acoustic model outputs
        NOTE only support one sequence (batch size = 1)
        '''''
        use_gpu = xs.is_cuda
        def forward_step(label, hidden):
            ''' `label`: int '''
            label = autograd.Variable(torch.LongTensor([label]), volatile=True).view(1,1)
            if use_gpu: label = label.cuda()
            label = self.embed(label)
            pred, hidden = self.decoder(label, hidden)
            # pred ~ embedding dim
            return pred[0], hidden

        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        #xs = self.encoder(xs)[0][0]
        B = [Sequence(blank=self.blank)]
        #for i, x in enumerate(xs):
        for i in range(xs.shape[1]):
            sorted(B, key=lambda a: len(a.k), reverse=True) # larger sequence first add
            A = B
            B = []
            if prefix:
                # for y in A:
                #     y.logp = log_aplusb(y.logp, prefixsum(y, A, x))
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not isprefix(A[i].k, A[j].k): continue
                        # A[i] -> A[j]
                        pred, _ = forward_step(A[i].k[-1], A[i].h)
                        idx = len(A[i].k)
                        #ytu = self.joint(x, pred)
                        ytu = self.joint(torch.unsqueeze(xs[0][i],0), pred)
                        logp = F.log_softmax(ytu, dim=1)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])
                        for k in range(idx, len(A[j].k)-1):
                            #ytu = self.joint(x, A[j].g[k])
                            ytru = self.joint(torch.unsqueeze(xs[0][i],0), A[j].g[k])
                            logp = F.log_softmax(ytu, dim=1)
                            curlogp += float(logp[A[j].k[k+1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp)
                # y* = most probable in A
                A.remove(y_hat)
                # calculate P(k|y_hat, t)
                # get last label and hidden state
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
                ytu = self.joint(torch.unsqueeze(xs[0][i],0), pred)
                logp = F.log_softmax(ytu, dim=1) # log probability for each k
                # TODO only use topk vocab
                for k in range(self.vocab_size):
                    yk = Sequence(y_hat)
                    yk.logp += float(logp[0][k])
                    if k == self.blank:
                        B.append(yk) # next move
                        continue
                    # store prediction distribution and last hidden state
                    yk.h = hidden; yk.k.append(k);
                    if prefix: yk.g.append(pred)
                    A.append(yk)
                # sort A
                # sort B
                y_hat = max(A, key=lambda a: a.logp)
                yb = max(B, key=lambda a: a.logp)
                if len(B) >= W and yb.logp >= y_hat.logp: break

            # beam width
            sorted(B, key=lambda a: a.logp, reverse=True)
            B = B[:W]

        # return highest probability sequence
        return B[0].k, -B[0].logp

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size=80
    vocab_size=71
    model=Transducer(device, vocab_size)

    inputs = torch.zeros(1, 100, 80)
    input_lengths = torch.tensor([25,], dtype=torch.int32)
    targets = torch.tensor([[1, 2, 3, 4, 5, 2]])
    target_lengths = torch.tensor([6,], dtype=torch.int32)

    loss = model(inputs, targets, input_lengths, target_lengths)
    print(loss)
