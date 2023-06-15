import torch
from torch import Tensor
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torchaudio.transforms import RNNTLoss
from torchaudio.prototype.models import rnnt_conformer_model
from einops import rearrange

class ASRTransducer(nn.Module):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.model = rnnt_conformer_model(
            input_dim=config['transducer']['input_dim'], # dimension of frames to the transcription network
            encoding_dim=config['transducer']['encoding_dim'], # dimension of features to joint network
            time_reduction_stride = config['transducer']['time_reduction_stride'],
            conformer_input_dim=config['transducer']['conformer_input_dim'],
            conformer_ffn_dim=config['transducer']['conformer_ffn_dim'],
            conformer_num_layers=config['transducer']['conformer_num_layers'],
            conformer_num_heads=config['transducer']['conformer_num_heads'],
            conformer_depthwise_conv_kernel_size=config['transducer']['conformer_depthwise_conv_kernel_size'],
            conformer_dropout=config['conformer_dropout'],
            num_symbols=config['transducer']['num_symbols'],
            sysmbol_embedding_dim=config['transducer']['symbol_embedding_dim'],
            num_lstm_layers=config['transducer']['num_lstm_layers'],
            lstm_hidden_dim=config['transducer']['lstm_hidden_dim'],
            lstm_layer_norm=False,
            lstm_layer_norm_epsilon=1.e-8,
            lstm_dropout=config['transducer']['lstm_dropout'],
            joiner_activation="relu",
        )
    
    def forward(self, sources:Tensor, source_lengths:list, targets:Tensor, target_lengths:list):
        source_lengths = torch.tensor(source_lengths)
        target_lengths = torch.tensor(target_lengths)
        # _joint_output : (B, S, T, F)
        # _output_source_lengths : (B,)
        # _output_target_lengths : (B, )
        # _output_states : list[list[Tensor]] 
        _joint_output, _output_source_lengths, _output_target_lengths, _ = self.model(
            sources=sources,
            source_lengths=source_lengths,
            targets=targets,
            target_lengths=target_lengths
        )
        _loss = F.rnnt_loss(_joint_output, targets, _output_source_lengths, _output_target_lengths, blank=0)

        return _loss
    
    def decode(self, sources:Tensor, source_lengths:list):
        source_lengths = torch.tensor(source_lengths)
        outputs, output_lengths = self.model.transcribe(sources, source_lengths)
        return outputs, output_lengths