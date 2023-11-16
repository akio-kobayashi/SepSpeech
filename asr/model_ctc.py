# conformer-based model
import copy
import torch
import torch.nn as nn
#from torch import autograd
import torch.nn.functional as F
import conformer
from warprnnt_pytorch import RNNTLoss
from conformer import ConformerConvModule, ConformerBlock
from cntf import CNTF
from einops import rearrange
from loss import MaskedMSELoss
import math
import numpy as np

class Sequence():
    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = [] # predictions of phoneme language model
            self.k = [blank] # prediction phoneme label
            # self.h = [None] # input hidden vector to phoneme model
            self.h = None
            self.logp = 0 # probability of this sequence, in log scale
        else:
            self.g = seq.g[:] # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp
        #self.transform=PhoneTransform()

    #def __str__(self):
    #    #print(self.k)
    #    return 'Prediction: {}\nlog-likelihood {:.2f}\n'.format(self.transform.int_to_text(self.k), -self.logp)
               
class Subsampler(nn.Module):
    def __init__(self):
        super(Subsampler, self).__init__()
        self.subsampler = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=64,
                            kernel_size=3, stride=1, padding=3//2, padding_mode='replicate',bias=False),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=64,out_channels=128,
                            kernel_size=3, stride=1, padding=3//2, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.feedforward=nn.Linear(20*128, 80)

    def forward(self, x):
        # add channel
        x = x.unsqueeze(dim=1)
        x = self.subsampler(x) # (B, 1, T, F) -> (B, 128, T//4, F//4)
        x = rearrange(x, 'b c t f -> b t (c f)')
        x = self.feedforward(x) # (B, T//4, 80)

        return x

class DownSampler(nn.Module):
    def __init__(self, output_dim):
        super(DownSampler, self).__init__()
        self.block1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=32,
                                    kernel_size=3, stride=1, padding=3//2,
                                    padding_mode='replicate',bias=False),
                                    nn.InstanceNorm2d(32),
                                    nn.LeakyReLU())
        # time//2, feats//2
        self.ds1=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#, ceil_mode=True)
        self.block2=nn.Sequential(nn.Conv2d(in_channels=32,out_channels=128,
                                    kernel_size=3, stride=1, padding=3//2,
                                    padding_mode='replicate', bias=False),
                                    nn.InstanceNorm2d(128),
                                    nn.LeakyReLU())
        # time//4, feats//4
        self.ds2=nn.MaxPool2d(kernel_size=2, stride=2)#, ceil_mode=True)
        self.feedforward=nn.Linear(20*128, output_dim)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.block1(x) # (B, 1, T, F) -> (B, 128, T, F)
        x = self.ds1(x)
        x = self.block2(x)
        x = self.ds2(x)
        x = rearrange(x, 'b c t f -> b t (c f)')
        x = self.feedforward(x) # (B, T//4, 80)

        return x

    def _valid_lengths(self, input_lengths, kernel_size=3, stride=1, padding=0, dilation=1.)->list:
        leng=[]
        for l in input_lengths:
            l = int(np.floor((l + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1))
            leng.append(l)
        return leng
    
    def valid_lengths(self, leng):
        leng = self._valid_lengths(leng, kernel_size=3, stride=1, padding=3//2)
        leng = self._valid_lengths(leng, kernel_size=2, stride=2, padding=0)
        leng = self._valid_lengths(leng, kernel_size=3, stride=1, padding=3//2)
        leng = self._valid_lengths(leng, kernel_size=2, stride=2, padding=0)

        return leng
        
'''
 RNN Transducer
    from https://github.com/HawkAaron/E2E-ASR/blob/91d6b96bd605a9bc8e62b4a5903ec2056fa0f88d/model.py#L37
'''
#@torch.compile
class Transducer(nn.Module):
    def __init__(self, device, vocab_size, ctc_vocab_size, hidden_size=144,
                 cell_size=320, num_layers=16, num_heads=8,
                 dropout=.5, blank=0, bos_token=1, eos_token=2,
                 ctc_weight = 1.0):
        super(Transducer, self).__init__()
        self.blank = blank
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.loss = RNNTLoss()

        self.ctc_vocab_size = ctc_vocab_size
        self.ctc_loss = nn.CTCLoss()
        
        # NOTE encoder & decoder only use lstm
        # instead of RNN, just use conformer block
        layers=[]
        for n in range(num_layers):
            layers.append(
                ConformerBlock(
                    dim = hidden_size,
                    dim_head = 64,
                    heads = self.num_heads,
                    ff_mult = 4,
                    conv_expansion_factor = 2,
                    conv_kernel_size = 31,
                    attn_dropout = 0.,
                    ff_dropout = 0.,
                    conv_dropout = 0.
                )
            )
        self.encoder = nn.Sequential(*layers) # (B, T, F)

        self.fc0 = nn.Linear(hidden_size, cell_size) # (B, T, H)
        
        #self.downsampler=DownSampler(output_dim=hidden_size)
        self.downsampler=CNTF(output_dim=hidden_size)

        self.embed = nn.Embedding(vocab_size, vocab_size-1, padding_idx=blank)
        self.embed.weight.data[1:] = torch.eye(vocab_size-1)
        self.embed.weight.requires_grad = False
        self.decoder = nn.LSTM(vocab_size-1, cell_size, 1,
                                batch_first=True, dropout=0., bidirectional=False)

        self.fc1 = nn.Linear(2*cell_size, cell_size)
        self.fc2 = nn.Linear(cell_size, vocab_size)

        self.ctc_decoder = nn.Linear(cell_size, ctc_vocab_size)
        self.ctc_weight = ctc_weight
        
    def valid_input_lengths(self, x):
        if self.downsampler is not None:
            y = self.downsampler.valid_lengths(x)
        else:
            y =x
        return y
    
    def joint(self, f, g):
        ''' `f`: encoder output (B,T,U,2H)
        `g`: decoder lstm output (B,T,U,H)
        NOTE f and g must have the same size except the last dim'''
        dim = len(f.shape) - 1
        out = torch.cat((f, g), dim=dim)
        out = torch.tanh(self.fc1(out))
        return self.fc2(out)

    def ff_encoder(self, xs):
        if self.downsampler is not None:
            xs = self.downsampler(xs)
        xs = self.encoder(xs)
        xs = self.fc0(xs)

        return xs

    def forward(self, xs, ys, zs, xlen, ylen, zlen, return_loss=True):
        if self.downsampler is not None:
            xs = self.downsampler(xs)
        xs = self.encoder(xs)
        xs = self.fc0(xs) # (B, time//scale, cell_size)
        
        ctc_out = self.ctc_decoder(xs)
        
        # concat first zero
        #zero = autograd.Variable(torch.zeros((ys.shape[0], 1)).long())
        zero = torch.zeros((ys.shape[0], 1)).long()
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

        # b t c -> t b c
        ctc_out = F.log_softmax(ctc_out, dim=-1)
        ctc_out = rearrange(ctc_out, 'b t c -> t b c')
        _ctc_loss = self.ctc_loss(ctc_out, zs, xlen, zlen)

        loss += self.ctc_weight * _ctc_loss
        
        return loss

    def greedy_decode(self, x, ff=True):
        if ff is True:
            if self.downsampler is not None:
                x = self.downsampler(x)
            x = self.encoder(x)
            x = self.fc0(x)

        #vy = autograd.Variable(torch.LongTensor([0]), volatile=True).view(1,1)
        vy = torch.LongTensor([self.bos_token]).view(1,1)
        # vector preserve for embedding
        if x.is_cuda: vy = vy.cuda()
        y, h = self.decoder(self.embed(vy)) # decode first zero
        y_seq = []; logp = 0
        logp_seq = []
        for i in range(x.shape[1]):
            ytu = self.joint(torch.unsqueeze(x[0][i],0), y[0])
            out = F.log_softmax(ytu, dim=1)
            p, pred = torch.max(out, dim=1) # suppose blank = -1
            pred = int(pred); logp += float(p)
            if pred != self.blank:
                y_seq.append(pred)
                logp_seq.append(-p.cpu().detach().numpy()[0])
                vy.data[0][0] = pred # change pm state
                y, h = self.decoder(self.embed(vy), h)
        return y_seq, [-logp, logp_seq]

    def beam_search(self, xs, W=10, ff=True, prefix=False):
        if ff is True:
            if self.downsampler is not None:
                xs = self.downsampler(xs)
            xs = self.encoder(xs)
            xs = self.fc0(xs)
        
        '''''
        `xs`: acoustic model outputs
        NOTE only support one sequence (batch size = 1)
        '''''
        
        use_gpu = xs.is_cuda
        def forward_step(label, hidden):
            ''' `label`: int '''
            #label = autograd.Variable(torch.LongTensor([label]), volatile=True).view(1,1)
            label = torch.LongTensor([label]).view(1,1)
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

    def simple_beam_search(self, xs, large_nbest=100, small_nbest=10):
        if self.downsampler is not None:
            xs = self.downsampler(xs)
        xs = self.encoder(xs)
        xs = self.fc0(xs)
        
        N, T, F = xs.shape
        
        use_gpu = xs.is_cuda
        def forward_step(label, hidden):
            label = torch.LongTensor([label]).view(1,1)
            if use_gpu: label = label.cuda()
            label = self.embed(label)
            pred, hidden = self.decoder(label, hidden)
            return pred[0], hidden

        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        B = [Sequence(blank=self.blank)]
        for i in range(T):
            sorted(B, key=lambda a: len(a.k), reverse=True) # larger sequence first add
            A = B
            B = []

            while True:
                y_hat = max(A, key=lambda a: a.logp) # y* = most probable in A
                A.remove(y_hat)

                # calculate P(k|y_hat, t)
                # get last label and hidden state
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
                ytu = self.joint(torch.unsqueeze(xs[0][i],0), pred)
                logp = F.log_softmax(ytu, dim=1) # log probability for each k
                top_k_logp, top_k_index = torch.topk(logp, small_nbest, dim=-1)
                top_k_logp, top_k_index = top_k_logp.tolist(), top_k_index.tolist()
                
                for i, k in enumerate(top_k_index):
                    yk = Sequence(y_hat)
                    yk.logp += float(top_k_logp[0][i])
                    if k == self.blank:
                        B.append(yk) # next move
                        continue
                    # store prediction distribution and last hidden state
                    yk.h = hidden; yk.k.append(k);
                    A.append(yk)
                # sort A
                # sort B
                y_hat = max(A, key=lambda a: a.logp)
                yb = max(B, key=lambda a: a.logp)
                if len(B) >= large_nbest and yb.logp >= y_hat.logp: break

            # beam width
            sorted(B, key=lambda a: a.logp, reverse=True)
            B = B[:large_nbest]

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
