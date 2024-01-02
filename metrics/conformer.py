import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class LinearAttention(nn.Module):
    def __init__(self, dim_model, heads=8, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim_model, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim_model, 1), 
                                    nn.GroupNorm(1, dim_model))

    def forward(self, x, mask):
        b, c, t = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) t -> b h c t", h=self.heads), qkv
        )

        # mask.shape = (b t)
        mask = torch.where(mask == True, 0.0, 1.e-12)
        mask = rearrange(mask, 'b t -> b h c t', h=1, c=1)
        q = q.softmax(dim=-2) # channel direction
        k = (k+mask).softmax(dim=-1) # temoral direction

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v) # (b h c c)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q) # (b h c t)
        out = rearrange(out, "b h c t -> b (h c) t")
        return self.to_out(out)
    
class GLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x.shape = (b c t)
        out, gate = torch.chunk(x, 2, dim=1)
        return out * gate.sigmoid()
    
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()

class Rearrange(nn.Module):
    def __init__(self, literal):
        super().__init__()
        self.literal = literal

    def forward(self, x):
        x = rearrange(x, self.literal)
        return x

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        #self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, padding=padding, groups = chan_in)

    def forward(self, x):
        #x = F.pad(x, self.padding)
        return self.conv(x)

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm(x)
        x = rearrange(x, 'b t c -> b c t')
        return x

class PreNorm(Norm):
    def __init__(self, dim, fn):
        super().__init__(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        x = rearrange(x, 'b t c -> b c t')
        return x

class FeedForward(nn.Module):
    def __init__(
        self,
        dim_model,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange('b c t -> b t c'),
            nn.Linear(dim_model, dim_model * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * mult, dim_model),
            Rearrange('b t c -> b c t'),
            nn.Dropout(dropout)
        )
        self.ff1 = nn.Linear(dim_model, dim_model*mult)

    def forward(self, x):
        x = rearrange(x, 'b c t -> b t c')
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim_model,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim_model * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            Rearrange('b c t -> b t c'),
            nn.LayerNorm(dim_model),
            Rearrange('b t c -> b c t'),
            nn.Conv1d(dim_model, inner_dim * 2, 1),
            GLU(),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim_model, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.norm1 = Norm(dim)
        self.ff1 = FeedForward(dim_model = dim, mult = ff_mult, dropout = ff_dropout)
        #self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.norm2 = Norm(dim)
        self.attn = LinearAttention(dim_model = dim, heads = heads, dim_head=dim_head)
        self.conv = ConformerConvModule(dim_model = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.norm3 = Norm(dim)
        self.ff2 = FeedForward(dim_model = dim, mult = ff_mult, dropout = ff_dropout)

        #self.attn = PreNorm(dim, self.attn)
        #self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        #self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = 0.5*(self.ff1(self.norm1(x))) + x
        x = self.attn(self.norm2(x), mask = mask) + x
        x = self.conv(x) + x
        x = 0.5*self.ff2(self.norm3(x)) + x
        x = rearrange(x, 'b c t -> b t c')
        x = self.post_norm(x)
        x = rearrange(x, 'b t c -> b c t')
        return x

class LearnableEncoder(nn.Module):
    def __init__(self, chin=1, chout=256, kernel_size=3, stride=1):
        super().__init__()
        padding = calc_same_padding(kernel_size)
        self.encoder = nn.Conv1d(chin, chout, kernel_size, stride, padding)
    
    def forward(self, x):
        # x.shape = (b t)
        assert x.dim() == 2
        x = rearrange('b (c t) -> b c t', c=1)

        return self.encoder(x)

class LearnableDecoder(nn.Module):
    def __init__(self, chin=256, chout=32, kernel_size=121, stride=1):
        super().__init__()
        padding = calc_same_padding(kernel_size)
        self.decoder = nn.Conv1d(chin, chout, kernel_size, stride, padding)
    
    def forward(self, x):
        # x.shape = (b c t)
        return self.decoder(x)
    
class SoundLevel(LearnableDecoder):
    def __init__(self, chin=256, chout=32, kernel_size=121, stride=1):
        super().__init__(chin=256, chout=32, kernel_size=121, stride=1)
    
    def forward(self, x):
        # x.shape = (b c t)
        x = super().forward(x)
        x = rearrange(torch.mean(x, dim=-1), 'b (c t) -> b c t', t=1)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block = ConformerBlock(dim=256).to(device)
    inputs = torch.randn(8, 256, 1024).to(device)
    outputs = block(inputs)
