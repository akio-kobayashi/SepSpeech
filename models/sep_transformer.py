import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.custom_transformer import CustomTransformer
form models.speaker import SpeakerNetwork
import custom_transformer
from einops import rearrange

class Rearrange(nn.Module):
    def __init__(self, pattern:str):
        super().__init__()
        self.pattern = pattern

    def foward(self, x):
        return rearrange(x, pattern)
    
class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=3000):
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
        B, T, C = x.shape
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class WaveEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            Rearrange('b t c -> b c t'),
            nn.Conv1d(in_channels = input_channels,
                      out_channels = out_channels,
                      kernel_size = 3,
                      stride = 1,
                      padding = 3//2),
            nn.ReLU(),
            Rearrange('b c t -> b t c'),
            nn.LayerNorm(out_channels)
        )
    def forward(self, x):
        return self.block(x)
    
class WaveDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            Rearrange('b t c -> b c t'),
            nn.Conv1d(in_channels = input_channels,
                      out_channels = out_channels,
                      kernel_size = 3,
                      stride = 1,
                      padding = 3//2),
            nn.ReLU(),
            Rearrange('b c t -> b t c'),
            nn.LayerNorm(out_channels)
        )
        
    def forward(self, x):
        return self.block(x)
    
class SepTransformer(nn.Module):
    def __init__(self, ,d_model, d_embed, nhead, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.d_model=d_model
        self.d_embed=d_embed
        self.nhead=nhead
        self.pos_enc = PositionEncoding(d_model=self.d_model)
        self.encoder = WaveEncoder(1, self.d_model)
        self.decoder = WaveDecoder(self.d_model, 1)
        self.transformer = CustomTransformer(d_model=self.d_model, d_embed=self.d_embed,
                                             nhead=self.nhead,
                                             num_encoder_layers=num_encoder_layers,
                                             num_decoder_layers=num_decoder_layers,
                                             batch_first=True,norm_first=True)
        self.speaker_encoder = SpeakerNetwork(self.encoder,
                                              in_channels=self.d_model,
                                              out_channels=self.d_embed,
                                              kernel_size=3,
                                              stride=1)
        
    def forward(self, src, tgt, src_spk, src_len, tgt_len):

        if src.dim()==2:
            src = src.unqueeze(-1)
        B, T, _ = src.shape
        zeros=torch.zeros((B, 1, 1), dtype=torch.float).cuda()
        ext_tgt=torch.cat((zeros, tgt),1)
        tgt_out = ext_tgt[:, 1:, :]
        tgt_in = ext_tgt[:, :T, :]

        encoded_src=self.pos_enc(self.encoder(src))
        encoded_tgt_in=self.pos_enc(self.encoder(tgt_in))
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = self.generate_masks(encoded_src, encoded_tgt_in, src_len, tgt_len)
        memory_mask=None

        B, T, C = encoded_src.shape
        enc_src_spk, z = self.speaker_encoder(src_spk) # (B, C) (B, S)
        ext_enc_src_spk = enc_src_spk.unsqueeze(1).repeat(1, T, C)
        
        y=self.transformer(encoded_src.cuda(),
                           encoded_tgt_in.cuda(),
                           ext_enc_src_spk.cuda(),
                           src_mask=src_mask.cuda(),
                           tgt_mask=tgt_mask.cuda(),
                           memory_mask= memory_mask,
                           src_key_padding_mask=src_pad_mask.cuda(),
                           tgt_key_padding_mask=tgt_pad_mask.cuda(),
                           memory_key_padding_mask=src_pad_mask.cuda())
        y=self.decoder(y) # (B, T)
        
        return y, z

    def generate_masks(self, src, tgt, src_len, tgt_len):
        # (B, T, F)
        B, S, _ = src.shape
        src_mask=torch.zeros((S,S), dtype=bool)
        B, T, _ = tgt.shape
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

    '''
    def get_attention_weight(self):
        return self.transformer._get_attention_weight()
    '''
    
    def greedy_decode(self, src, src_spk, src_len, max_len):
        # bathsize=1, then no need padding mask

        B, S, _ = src.shape
        with torch.no_grad():
            src_padding_mask = torch.ones(1, S, dtype=bool)
            src_padding_mask[:, :src_len]=False
            y = self.pos_enc(self.encoder(src))
            _, _, C = y.shape
            enc_src_spk, _ = self.speaker_encoder(src_spk) # (B, C) (B, S)
            ext_enc_src_spk = enc_src_spk.unsqueeze(1).repeat(1, S, C)
            
            memory = self.transformer.encoder(y.cuda(), ext_enc_src_spk.cuda(), src_key_padding_mask=src_padding_mask.cuda())
            ys=torch.zeros((1, 1, C), dtype=torch.float).cuda()
            memory_mask=None
            
        for i in range(max_len):
            with torch.no_grad():
                _, T, _ = ys.shape
                mask=self.generate_square_subsequent_mask(T).cuda()
                z = self.pos_enc(self.encoder(ys))
                z = self.transformer.decoder(z,
                                             memory,
                                             ext_enc_src_spk,
                                             tgt_mask=mask,
                                             memory_mask=memory_mask)
                z = self.decoder(z)
                z = z[:, -1, :].reshape(1, 1, -1)

                ys = torch.cat((ys, z), dim=1) #(1, T+1, C)

                '''
                # make memory mask for monotonic decoding
                weights=self.transformer._get_attention_weight() # [(1, T, S),...]
                sum=None
                for weight in weights:
                    if sum is None:
                        sum=weight
                    else:
                        sum+=weight
                max_pos = torch.argmax(sum, dim=2) # (1, T)
                max_pos = max_pos.to('cpu').detach().numpy().copy()
                start = max_pos[0, -1]-5
                if start < 0:
                    start = 0
                end = max_pos[0, -1]+10
                if end > src_len:
                    end = src_len
                if memory_mask is None:
                    memory_mask=torch.zeros(self.nhead, ys.shape[1], src.shape[1]).cuda()
                    memory_mask[:, 0, :]=1.
                    memory_mask[:, 1, start:end]=1.
                    memory_mask=memory_mask.float().masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask == 1, float(0.0))
                else:
                    temp = torch.zeros(self.nhead, 1, src.shape[1]).cuda()
                    temp[:, -1, start:end] = 1.
                    temp=temp.float().masked_fill(temp == 0, float('-inf')).masked_fill(temp == 1, float(0.0))
                    memory_mask = torch.cat((memory_mask, temp), dim=1)
                '''
                '''
                # normal masking
                diag_mask=voice_transformer.compute_diagonal_masks(memory_mask.shape[0])
                memory_mask += diag_mask
                '''

            torch.cuda.empty_cache()

        return ys

    def predict(self, src, tgt, src_spk, max_len):
        # bathsize=1, then no need padding mask
        B, S, C = src.shape
        with torch.no_grad():
            zeros=torch.zeros((B, 1, C), dtype=torch.float).cuda()
            ext_tgt=torch.cat((zeros, tgt),1)
            _, T, _ = tgt.shape
            ext_tgt_in = ext_tgt[:, :T, :]

        with torch.no_grad():
            y = self.pos_enc(self.encoder(y))
            enc_src_spk, _ = self.speaker_encoder(src_spk) # (B, C) (B, S)
            _, S, C = enc_src_spk.shape
            ext_enc_src_spk = enc_src_spk.unsqueeze(1).repeat(1, S, 1)
            
            memory = self.transformer.encoder(y, ext_enc_src_spk)

        output = None
        for i in range(max_len):
            with torch.no_grad():
                mask=self.generate_square_subsequent_mask(i+1).cuda()
                z = self.pos_enc(self.encoder(tgt_in[:, :i+1, :]))
                z = self.transformer.decoder(z, memory, ext_enc_src_spk, tgt_mask=mask)
                z = self.decoder(z)
                z = z[:, -1, :].reshape(1, 1, -1)
                if output is None:
                    output = z
                else:
                    output = torch.cat((output, z), dim=1)

            torch.cuda.empty_cache()

        return output

if __name__ == '__main__':
    with open("config.yaml", 'r') as yf:
        config = yaml.safe_load(yf)

    model = SepTransformer(config['transformer'])
    length = 161231
    x = torch.rand(4, length)
    s = torch.rand(4, length)
    y, _, = model(x, s)
    print(x.shape)
    print(y.shape)

