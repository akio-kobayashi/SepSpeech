import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from voice_transformer import VoiceTransformer
import voice_transformer

class LearnableEncoder(nn.Module):
    def __init__(self, chin=1, chout=2048, kernel_size=400, stride=160, padding=200) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(chin, chout, kernel_size, stride, padding)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.encoder(x)
    
class LearnableDecoder(nn.Module):
    def __init__(self, chin=2048, chout=1, kernel_size=400, stride=160, padding=0, output_padding=0) -> None:
        super().__init__()
        self.decoder = nn.ConvTranspose1d(chin, chout, kernel_size, stride, padding, output_padding)

    def forward(self, x:Tensor) -> Tensor:
        return self.decoder(x)

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
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.l1loss=nn.L1Loss()

    def forward(self, y_pred, y_true, y_pred_len, y_true_len):
        zeros=torch.zeros((y_pred.shape[0], 1, y_pred.shape[2]), dtype=torch.float).cuda()
        y_pred_sft=torch.cat([zeros, y_pred], 1)
        delta_pred=y_pred[:, :, :] - y_pred_sft[:, 0:-1, :]
        y_true_sft=torch.cat([zeros, y_true], 1)
        delta_true=y_true[:,:,:] - y_true_sft[:, 0:-1,:]

        loss=0.
        for b in range(y_pred.shape[0]):
            l1 = self.l1loss(torch.unsqueeze(delta_pred[b, :y_pred_len[b], :],0),
                            torch.unsqueeze(delta_true[b, :y_true_len[b], :],0))
            loss += l1
        return torch.mean(loss)

class BMSELoss(nn.Module):
    def __init__(self):
        super(BMSELoss, self).__init__()
        self.mse=nn.MSELoss()

    def forward(self, y_pred, y_true, y_pred_len, y_true_len):
        B=y_pred.shape[0]

        loss = 0.
        for b in range(B):
            mse = self.mse(torch.unsqueeze(y_pred[b, :y_pred_len[b], :],0),
                     torch.unsqueeze(y_true[b, :y_true_len[b], :],0))
            loss += mse

        return torch.mean(loss)

class BL1Loss(nn.Module):
    def __init__(self):
        super(BL1Loss, self).__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, y_pred, y_true, y_pred_len, y_true_len):
        B=y_pred.shape[0]

        loss = 0.
        for b in range(B):
            l1 = self.l1loss(torch.unsqueeze(y_pred[b, :y_pred_len[b], :],0),
                            torch.unsqueeze(y_true[b, :y_true_len[b], :],0))
            loss += l1

        return torch.mean(loss)

class VCModel(nn.Module):
    def __init__(self, config):
        super(VCModel, self).__init__()
        self.fdim=config['fdim']
        self.d_model=config['d_model']
        self.d_embed=config['embdim']
        self.nhead=config['nhead']
        self.pos_enc = PositionEncoding(d_model=self.fdim)
        self.encoder = LearnableEncoder()
        self.decoder = LearnableDecoder()
        self.transformer = VoiceTransformer(d_model=self.d_model,
                                            d_embed=self.d_embed,
                                            nhead=self.nhead,
                                            num_encoder_layers=config['num_encoder_layers'],
                                            num_decoder_layers=config['num_decoder_layers'],
                                            batch_first=True,
                                            norm_first=True)
        self.dloss = BL1Loss()

    def forward(self, src, tgt, src_spk, tgt_spk, src_len, tgt_len, nu=0.3, weight=1.):

        zeros=torch.zeros((src.shape[0], 1, src.shape[2]), dtype=torch.float, device=src.device)
        tgtz=torch.cat((zeros, tgt),1)
        tgt_out = tgtz[:, 1:, :]
        tgt_in = tgtz[:, 0:tgtz.shape[1]-1, :]

        y=self.encoder(self.pos_enc(src))
        z=self.encoder(self.pos_enc(tgt_in))

        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = self.generate_masks(y, z, src_len, tgt_len)
        memory_mask=None
        y=self.transformer(y, z, src_spk, tgt_spk,
                           src_mask=src_mask,
                           tgt_mask=tgt_mask,
                           memory_mask= memory_mask,
                           src_key_padding_mask=src_pad_mask,
                           tgt_key_padding_mask=tgt_pad_mask,
                           memory_key_padding_mask=src_pad_mask)
        y=self.decoder(y)
        
        loss=None
        mloss=None
        diag_loss=None
        if tgt is not None:
            mloss=self.dloss(y, tgt_out, tgt_len, tgt_len)
            weights = voice_transformer.compute_diagonal_weights(src.shape[0], tgt_out.shape[1], src.shape[1], src_len, tgt_len, nu)
            diag_loss = self.transformer._diagonal_attention_loss(weights, src_len, tgt_len)
            loss = mloss + weight * diag_loss
        return y, loss, mloss, diag_loss

    def generate_masks(self, src, tgt, src_len, tgt_len):
        # (B, T, F)
        B=src.shape[0]
        S=src.shape[1]
        src_mask=torch.zeros((S,S), dtype=bool, device=src.device)
        T=tgt.shape[1]
        tgt_mask=self.generate_square_subsequent_mask(T, src_mask.device)

        src_padding_mask=torch.ones((B, S), dtype=bool, device=src.device)
        tgt_padding_mask=torch.ones((B, T), dtype=bool, device=tgt.device)
        for b in range(B):
            src_padding_mask[b, :src_len[b]]=False
            tgt_padding_mask[b, :tgt_len[b]]=False

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def generate_square_subsequent_mask(self, seq_len, device):
        mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_attention_weight(self):
        return self.transformer._get_attention_weight()

    def greedy_decode(self, src, src_spk, tgt_spk, src_len, max_len):
        # bathsize=1, then no need padding mask

        src = src.cuda()
        with torch.no_grad():
            src_padding_mask = torch.ones(1, src.shape[1], dtype=bool, device=src.device)
            src_padding_mask[:, :src_len]=False
            y = self.encoder(self.pos_enc(src))
            memory = self.transformer.encoder(y, src_spk, src_key_padding_mask=src_padding_mask)
            ys=torch.zeros((1, 1, src.shape[-1]), dtype=torch.float, device=src.device)
            memory_mask=None
        for i in range(max_len - 1):
            with torch.no_grad():
                mask=self.generate_square_subsequent_mask(ys.shape[1], device=src.device)
                z = self.encoder(self.pos_enc(ys)) # (1, T, F)
                tgt=tgt_spk.repeat(1, z.shape[1], 1)
                z = self.transformer.decoder(z, memory, tgt, tgt_mask=mask, memory_mask=memory_mask)
                z = self.decoder(z)
                z = z[:, -1, :].reshape(1, 1, -1)

                ys = torch.cat((ys, z), dim=1) #(1, T+1, F)

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
                diag_mask=voice_transformer.compute_diagonal_masks(memory_mask.shape[0])
                memory_mask += diag_mask
                '''

            torch.cuda.empty_cache()

        return ys

    def predict(self, src, tgt, src_spk, tgt_spk, max_len):
        # bathsize=1, then no need padding mask
        src = src
        with torch.no_grad():
            zeros=torch.zeros((src.shape[0], 1, src.shape[2]), dtype=torch.float, device=src.device)
            tgtz=torch.cat((zeros, tgt),1)
            tgt_in = tgtz[:, 0:tgtz.shape[1]-1, :]

        with torch.no_grad():
            y = self.encoder(self.pos_enc(src))
            memory = self.transformer.encoder(y, src_spk)

        output = None
        for i in range(max_len - 1):
            with torch.no_grad():
                mask=self.generate_square_subsequent_mask(i+1)
                z = self.encoder(self.pos_enc(tgt_in[:, 0:i+1, :]))
                tspk=tgt_spk.repeat(1, z.shape[1], 1)
                z = self.transformer.decoder(z, memory, tspk, tgt_mask=mask)
                z = self.decoder(z)
                z = z[:, -1, :].reshape(1, 1, -1)
                if output is None:
                    output = z
                else:
                    output = torch.cat((output, z), dim=1)

            torch.cuda.empty_cache()

        return output
