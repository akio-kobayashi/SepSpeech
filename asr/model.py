import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
sys.path.append('../asr/tiny_transformer')
#from conformer import ConformerConvModule, ConformerBlock
from torchaudio.models import Conformer

class LayerScale(nn.Module):
    def __init__(self, dim, scale):
        chwise_scaler = nn.Parameter(torch.ones(dim) * scale)
        torch.register_parameter('chwise_scaler', diag)

    def forward(self, x):
        # x: b c t f
        x = rearrange('b c t f-> b t f c')
        x = torch.mul(self.diag, x)
        x = rearrange('b t f c -> b c t f')
        return x

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=bias)
    
    def forward(self, x):
        return self.depthwise(x)
    
class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(x)

class ConvNeXTBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.DepthwiseiConv(in_channels, kernel_size, padding),
            nn.LayerNorm(in_channels),
            nn.PointwiseConv(in_channels, in_channels*4),
            nn.GELU(),
            nn.PointwiseConv(in_channels*4, in_channels),
            LayerScale(in_channels)
        )
    def forward(self, x):
        return self.block(x)
    
class CNTF(nn.Module):
    def __init__(self, dim=80, cntf_channels=104, output_dim=1024, kernel_size=3) -> None:
        super().__init__()
        self.kernel_size=kernel_size
        for index in range(self.depth):
            self.cntf = nn.Sequential(
                nn.Conv2d(1, cntf_channels, kernel_size, stride=2),
                nn.LayerNorm(cntf_channels),
                ConvNeXTBlock(cntf_channels, kernel_size),
                nn.LayerNorm(cntf_channels),
                nn.Conv2d(cntf_channels, 2*cntf_channels, kernel_size, stride=2),
                ConvNeXTBlock(2*cntf_channels, kernel_size),
                nn.LayerNorm(2*cntf_channels),
                nn.Conv2d(2*cntf_channels, 3*cntf_channels, kernel_size=(3,1), stride=(2,1)),
                ConvNeXTBlock(3*cntf_channels),
                nn.LayerNorm(3*cntf_channels),
            )
            self.linear = nn.Linear(3*cntf_channels*dim, output_dim)

    def forward(self, x:Tensor) -> Tensor:
        # x (b t f -> b 1 t f)
        x = self.cntf(x.unsqueeze(1))
        # x (b c t f) -> (b t (c f))
        x = rearrange('b c t f -> b t (c f)')
        return self.linear(x)
    
    def _valid_lengths(self, input_lengths, kernel_size=3, stride=1, padding=0, dilation=0):
        leng=[]
        for l in input_lengths:
            np.floor((l + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1)
            leng.append(l)
        return leng

    def valid_lengths(self, input_lengths):
        leng = _valid_lengths(input_lengths, self.kernel_size, stride=2)
        leng = _valid_lengths(leng, self.kernel_size, stride=1)
        leng = _valid_lengths(leng, self.kernel_size, stride=2)
        leng = _valid_lengths(leng, self.kernel_size, stride=1)
        leng = _valid_lengths(leng, self.kernel_size, stride=2)
        leng = _valid_lengths(leng, self.kernel_size, stride=1)
        return leng

class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (b, t, f)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class CTCLoss(nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.ctc=nn.CTCLoss()

    def forward(self, outputs, labels, output_lengths, label_lengths):
        outputs = rearrange(outputs, 'b t c -> t b c')

        return  self.ctc(outputs, labels,
                        output_lengths, label_lengths)

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce=nn.CrossEntropyLoss()

    def forward(self, y_prd, y_ref, y_prd_len, y_ref_len):
        loss = 0.
        for b in range(y_prd.shape[0]):
            prd=y_prd[b, :y_prd_len[b], :]
            ref=y_ref[b, :y_ref_len[b]]
            loss += self.ce(prd,ref)

        return torch.mean(loss)

class ASRModel(nn.Module):
    def __init__(self, config):
        super(ASRModel, self).__init__()
        self.dim_input=config['dim_input']
        self.dim_output=config['dim_output']
        self.dim_model=config['dim_model']
        self.dim_feedforward=config['dim_feedforward']
        self.num_heads=config['num_heads']
        self.num_encoder_layers=config['num_encoder_layers']
        self.num_decoder_layers=config['num_decoder_layers']
        #self.enc_pe = PositionEncoding(self.dim_model, max_len=2000)
        self.dec_pe = PositionEncoding(self.dim_model, max_len=256)
        self.dec_embed = nn.Embedding(self.dim_output, self.dim_model)
        self.use_memory_mask = bool(config['memory_mask'])

        self.specdim=config['specdim']
        self.cntf_channels=config['cntf_channels']
        self.kernel_size=config['kernel_size']
        self.cntf = CNTF(dim=self.spcedim, cntf_channels=config['cntf_channels'], output_dim=self.dim_input, kernel_size=self.kernel_size)

        if config['model_type'] == 'conformer':
            self.encoder = Conformer(input_dim=self.dim_model, 
                                     num_heads=self.num_heads, 
                                     ffn_dim=self.dim_model, 
                                     num_layers=self.num_encoder_layers, 
                                     depthwise_conv_kernel_size=self.kernel_size,
                                     dropout = 0.0, 
                                     use_group_norm=False, 
                                     convolution_first=False)
        else:
            encoder_layer = nn.TransformerEncoderLayer(self.dim_model,
                                                       self.num_head,
                                                       self.dim_feedforward,
                                                       batch_first=True,
                                                       norm_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer,
                                                 self.num_encoder_layers,
                                                 nn.LayerNorm(self.dim_model))
            
        decoder_layer = nn.TransoformerDecoderLayer(self.dim_model,
                                                    self.num_heads,
                                                    self.dim_feedforward,
                                                    batch_first=True,
                                                    norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             self.num_decoder_layers,
                                             nn.LayerNorm(self.dim_model))

        self.ce_loss = CELoss()

    def forward(self, inputs, labels, input_lengths, label_lengths):

        labels_in = labels[:, 0:labels.shape[-1]-1] # remove eos 
        labels_out = labels[:, 1:] # remove bos

        label_lengths -= 1 # remove tag

        #y=self.enc_pe(self.input_encoder(inputs))
        y = self.cntf(inputs)
        valid_input_lengths = self.cntf.valid_lenths(input_lengths)

        z=self.dec_pe(self.dec_embed(labels_in))
        source_mask, target_mask, source_padding_mask, target_padding_mask = self.generate_masks(y, z, valid_input_lengths, label_lengths)

        memory = self.encoder(y)
        y = self.decoder(z, memory, tgt_mask=target_mask,
                         memory_mask=None,
                         tgt_key_padding_mask=target_padding_mask,
                         memory_key_padding_mask=source_padding_mask)
        
    
        return y, valid_input_lengths

    def generate_masks(self, src, tgt, src_len, tgt_len):
        B=src.shape[0]
        S=src.shape[1]
        src_mask=torch.zeros((S,S), dtype=bool)
        T=tgt.shape[1]
        tgt_mask=self.generate_square_subsequent_mask(T)

        src_padding_mask=torch.ones((B, S), dtype=bool)
        tgt_padding_mask=torch.ones((B, T), dtype=bool)
        for b in range(B):
            src_padding_mask[b, :src_len[b]]=False
            tgt_padding_mask[b, :tgt_len[b]]=False

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def generate_square_subsequent_mask(self, seq_len):
        mask = (torch.triu(torch.ones((seq_len, seq_len))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def update_memory_mask(self, src, src_len, ys, memory_mask):
        # make memory mask for monotonic decoding
        weights=self.decoder._get_attention_weights() # [(1, T, S),...]
        sum=weights[0]
        for n in range(1, len(weights)):
            sum+=weights[n]
        max_pos = torch.argmax(torch.from_numpy(sum), dim=2).to('cpu').detach().numpy().copy() # (1, T)
        start = max_pos[0, -1]-5
        if start < 0:
            start = 0
        end = max_pos[0, -1]+10
        if end > src_len:
            end = src_len
        if memory_mask is None:
            memory_mask=torch.zeros(self.num_heads, ys.shape[1], src.shape[1])
            memory_mask[:, 0, :]=1.
            memory_mask[:, 1, start:end]=1.
            memory_mask=memory_mask.float().masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask == 1, float(0.0))
        else:
            temp = torch.zeros(self.num_heads, 1, src.shape[1])
            temp[:, -1, start:end] = 1.
            temp=temp.float().masked_fill(temp == 0, float('-inf')).masked_fill(temp == 1, float(0.0))
            memory_mask = torch.cat((memory_mask, temp), dim=1)

        return memory_mask

    def greedy_decode(self, src, src_len, max_len):
        with torch.no_grad():
            src_padding_mask = torch.ones(1, src.shape[1], dtype=bool)
            src_padding_mask[:, :src_len]=False
            #y = self.enc_pe(self.prenet(src))
            y = self.cntf(src)
            #memory = self.transformer.encoder(y, src_key_padding_mask=src_padding_mask)
            memory = self.encoder(y)
            ys=torch.ones((1, 1), dtype=torch.int)
            ys*=2
            memory_mask=None
        for i in range(max_len - 1):
            with torch.no_grad():
                mask=self.generate_square_subsequent_mask(ys.shape[1])
                z = self.dec_pe(self.dec_embed(ys))
                z = self.decoder(z, memory, tgt_mask=mask, memory_mask=memory_mask)
                z = F.log_softmax(self.fc(z), dim=-1)
                z = torch.argmax(z[:, -1, :]).reshape(1, 1)

                ys = torch.cat((ys, z), dim=1) #(1, T+1)

                if self.use_memory_mask:
                    memory_mask = self.update_memory_mask(src, src_len, ys, memory_mask)

                if z == 3:
                    break

            torch.cuda.empty_cache()

        ys = ys.to('cpu').detach().numpy().copy().squeeze()
        return ys.tolist()
