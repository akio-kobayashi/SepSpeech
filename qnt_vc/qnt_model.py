import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import math
import qnt_vc.qnt_utils as U

class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        #position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position = rearrange(torch.arange(0, max_len, dtype=torch.float), '(t f) -> t f', t=max_len)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = rearrange(pe, '(b t) f -> b t f', b=1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print(self.pe.shape)
        #print(x.shape)
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class QntVoiceConversionModel(nn.Module):
    def __init__(self, config):
        super().__init__()        
        self.speaker_embedding = nn.Embedding(config['num_speakers'], config['num_speaker_embeddings'])
        self.qnt_embedding = nn.Embedding(config['num_qnt_symbols'], config['num_symbol_embeddings'])
        self.ar_transformer = QntARTransformer(config)
        self.nar_transformer = QntNARTransformer(config)

        self.bos_token_id = 1024
        self.eos_token_id = 1025
        self.max_len = config['max_length']

    def make_batch(self, src, tgt, src_id, tgt_id, ar=True):
        if ar is True:
            # append bos/eos tokens to both-ends of targets
            _src, _tgt, _src_id, _tgt_id, _src_lengths, _tgt_lengths = U.make_batch(src, tgt, src_id, tgt_id, ar=True)
        else:
            _src, _tgt, _src_id, _tgt_id, _src_lengths, _tgt_lengths = U.make_batch(src, tgt, src_id, tgt_id, ar=False)
            _src = _tgt
            _src_id = _tgt_id
            _src_lengths = _tgt_lengths

        B, C, S, _ = _src.shape
        #print(_src.shape)
        if ar is True:
            _src = rearrange(_src[:, 0, :, :], '(b c) t f -> b c (t f)', c=1) # b c t f
            _tgt = rearrange(_tgt[:, 0, :, :], '(b c) t f -> b c (t f)', c=1) # b c t f
        else:
            _src = rearrange(_src[:, 1:, :, :], 'b c t f -> b c (t f)')  # b c t f
            _tgt = rearrange(_tgt[:, :-1, :, :], 'b c t f -> b c (t f)') # b c t f

        B, C, S = _src.shape
        _src_id = rearrange(_src_id, '(b c t) -> b c t', c=1, t=1)
        _src_embed = self.speaker_embedding(_src_id).repeat((1, C, S, 1))
        _src = self.qnt_embedding(_src)
        _src = torch.cat([_src, _src_embed], dim=-1)

        B, C, T = _tgt.shape
        _tgt_id = rearrange(_tgt_id, '(b c t) -> b c t', c=1, t=1)
        _tgt_embed = self.speaker_embedding(_tgt_id).repeat((1, C, T, 1))
        _tgt = self.qnt_embedding(_tgt)
        _tgt = torch.cat([_tgt, _tgt_embed], dim=-1)

        return _src, _tgt, _src_lengths, _tgt_lengths

    def forward(self, src, tgt, src_id, tgt_id):
        ar_src, ar_tgt, src_lengths, tgt_lengths = self.make_batch(src, tgt, src_id, tgt_id, ar=True)
        ar_outputs = self.ar_transformer(ar_src, ar_tgt, src_lengths, tgt_lengths)

        nar_src, _, src_lengths, _ = self.make_batch(src, tgt, src_id, tgt_id, ar=False)
        nar_outputs = self.nar_transformer(nar_src, src_lengths)

        #print(ar_outputs.shape)
        #print(nar_outputs.shape)
        return torch.cat([ar_outputs, nar_outputs], dim=1)

    def greedy_decode(self, src, src_id, tgt_id):
        B, C, S, _ = src.shape  
        assert B == 1

        with torch.no_grad():
            src_id = rearrange(src_id, '(b c t f) -> b c t f', c=1, t=1, f=1)
            src_embed = self.speaker_embedding(src_id).repeat((1, C, S, 1))
            src = self.qnt_embedding(src)
            src = torch.cat([src, src_embed], dim=-1)
            self.ar_transformer.position_encoding(src)
            memory = self.model.encoder(src)

            # AR
            tgt_id = rearrange(tgt_id, '(b c t f) -> b c t f', c=1, t=1, f=1)
            tgt_embed = self.speaker_embedding(tgt_id).repeat((1, 1, 1, 1))

            # list of token ids
            tgt_list = [self.bos_token_id]
            tgt = rearrange(torch.tensor(tgt_list, device=self.device), 
                           '(b c t f) -> b c t f', b=1, c=1, f=1)
            tgt = torch.cat([tgt, tgt_embed], dim=-1)
            
            for i in range(self.max_len - 1):
                _, _, T, _ = tgt.shape
                tgt_mask = self.nar_transformer.generate_square_subsequent_mask(T).to(self.device)
                out = self.ar_transformer.feedforward(self.ar_transformer.decoder(tgt, memory, tgt_mask=tgt_mask))
                index = torch.argmax(out[:, :, -1, :], dim=-1).detach().numpy()[0]
                if index == self.eos_token_id:
                    break
                tgt_list.append(index)
                tgt = rearrange(torch.tensor(tgt_list, device=self.device), '(b c t f) -> b c t f', b=1, c=1, f=1)
                _, _, T, _ = tgt.shape
                tgt = torch.cat([tgt, tgt_embed.repeat(1, 1, T, 1)], dim=-1)

            # NAR
            if tgt_list[0] == self.bos_token_id:
                tgt_list.pop(0)
            # list of token ids to tensor
            tgt = rearrange(torch.tensor(tgt_list, device=self.device), '(b c t) -> b c t', b=1, c=1)
            tgts = [tgt]
            T = tgt.shape[-1]
            tgt = torch.cat([self.qnt_embedding(tgt), tgt_embed.repeat((1, 1, T, 1))], dim=-1)
            for i in range(7):
                out = self.nar_transformer(tgt)
                index = torch.argmax(out[:, -1, :, :], dim=-1).detach().numpy() # b t
                tgt = rearrange(torch.tensor(index, device=self.device), '(b c) t -> b c t', c=1)
                tgts.append(tgt)
                tgt = torch.cat(tgts, dim=1)
                T = tgt.shape[-1]
                tgt = torch.cat(self.qnt_embedding(tgt), tgt_embed.repeat((1, 1, T, 1)), dim=-1)

            outputs = torch.cat(tgts, dim=1)
        return outputs.to(torch.int64)

class QntBaseTransformer(nn.Module):
    def __init__(self, config):
        '''
            transformer.encoder.shape = (b c t (num_symbols + num_speakers))
            transformer.decoder.shape = (b c t (num_symbols + num_speakers))
        '''
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = config['num_symbol_embeddings'] + config['num_speaker_embeddings']
        self.position_encoding = PositionEncoding(self.d_model, max_len=4096)

        d_out = config['num_qnt_symbols']
        self.feedforward = nn.Linear(self.d_model, d_out)

    def forward(self, src, tgt, src_lengths, tgt_lengths):
        raise NotImplementedError

class QntARTransformer(QntBaseTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.num_channels = 1

        self.model = nn.Transformer(
            d_model=self.d_model, 
            nhead=config['num_heads'], 
            num_encoder_layers=config['num_encoder_layers'], 
            num_decoder_layers=config['num_decoder_layers'], 
            dim_feedforward=config['dim_feedforward'], 
            dropout=config['dropout'], 
            batch_first=True, 
            norm_first=False 
        )

    def forward(self, src, tgt, src_lengths, tgt_lengths): 
        src = rearrange(src, 'b c t f -> b t (c f)') 
        tgt = rearrange(tgt, 'b c t f -> b t (c f)')

        B, S, _ = src.shape
        B, T, _ = tgt.shape
        #print(src.shape)
        src = self.position_encoding(src)
        tgt = self.position_encoding(tgt)

        tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.create_masks(B, T, S, src_lengths, tgt_lengths)

        # encoder
        memory = self.model.encoder(src, src_key_padding_mask=src_key_padding_mask)

        # decoder
        outputs = self.model.decoder(tgt, memory, tgt_mask=tgt_mask, 
                                     tgt_key_padding_mask=tgt_key_padding_mask)
                                     #tgt_is_causal=True)
        outputs = self.feedforward(outputs)

        # to (b c t f)
        outputs = rearrange(outputs, 'b t (c f) -> b c t f', c=1)

        return outputs
    
    def create_masks(self, B, T, S, src_lengths, tgt_lengths):
        tgt_mask = self.model.generate_square_subsequent_mask(T).to(self.device)

        src_key_padding_mask=torch.ones((B, S), dtype=bool, device=self.device)
        tgt_key_padding_mask=torch.ones((B, T), dtype=bool, device=self.device)
        for b in range(B):
            src_key_padding_mask[b, :src_lengths[b]]=False
            tgt_key_padding_mask[b, :tgt_lengths[b]]=False

        return tgt_mask, src_key_padding_mask, tgt_key_padding_mask
    
class QntNARTransformer(QntBaseTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.num_channels = config['num_channels']

        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=config['num_heads'],
                dim_feedforward=config['dim_feedforward'], 
                dropout=config['dropout'],
                batch_first=True,
                norm_firat=False
            ),
            num_layers=config['num_nar_layers'], 
        )
    def create_mask(self, B, S):
        src_mask = self.model.generate_square_subsequent_mask(S).to(self.device)

        return src_mask
    
    def forward(self, src, src_lengths):
        # from (b c t f) 
        B,C,_,_= src.shape
        mask, _ = self.create_mask(B, C)
        src = rearrange(src, 'b c t f -> (b t) c f')
        src = self.position_encoding(src)

        # encoder
        outputs = self.model.encoder(src, mask=mask)
        outputs = self.feedforward(outputs)

        # to (b c t f)
        outputs = rearrange(outputs, '(b t) c f -> b c t f', b=B)

        return outputs
    
        
