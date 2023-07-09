import torch
from torch import Tensor
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torchaudio.transforms import RNNTLoss
import torchaudio.prototype.models as M
from torchaudio.prototype.models import conformer_rnnt_model
from einops import rearrange
import numpy as np

class ASRTransducer(nn.Module):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.model = conformer_rnnt_model(
            input_dim=config['transducer']['input_dim'], # dimension of frames to the transcription network
            encoding_dim=config['transducer']['encoding_dim'], # dimension of features to joint network
            time_reduction_stride = config['transducer']['time_reduction_stride'],
            conformer_input_dim=config['transducer']['conformer_input_dim'],
            conformer_ffn_dim=config['transducer']['conformer_ffn_dim'],
            conformer_num_layers=config['transducer']['conformer_num_layers'],
            conformer_num_heads=config['transducer']['conformer_num_heads'],
            conformer_depthwise_conv_kernel_size=config['transducer']['conformer_depthwise_conv_kernel_size'],
            conformer_dropout=config['transducer']['conformer_dropout'],
            num_symbols=config['transducer']['num_symbols'],
            symbol_embedding_dim=config['transducer']['symbol_embedding_dim'],
            num_lstm_layers=config['transducer']['num_lstm_layers'],
            lstm_hidden_dim=config['transducer']['lstm_hidden_dim'],
            lstm_layer_norm=False,
            lstm_layer_norm_epsilon=1.e-8,
            lstm_dropout=config['transducer']['lstm_dropout'],
            joiner_activation="relu",
        )
    
    def forward(self, sources:Tensor, targets:Tensor, source_lengths:list, target_lengths:list):
        source_lengths = torch.tensor(source_lengths, dtype=torch.int32)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32)
        # targets (B, U) -> (B, U+1)
        padded_targets = torch.cat([torch.zeros(len(targets), 1, dtype=torch.int32).cuda(), targets], dim=-1)
        padded_target_lengths = torch.tensor([ l+1 for l in target_lengths ], dtype=torch.int32)
        # _joint_output : (B, T, U+1, C)
        # _output_source_lengths : (B,)
        # _output_target_lengths : (B,)
        # _output_states : list[list[Tensor]] 
        _joint_output, _output_source_lengths, _output_target_lengths, _ = self.model(
            sources=sources.cuda(),
            source_lengths=source_lengths.cuda(),
            targets=padded_targets.cuda(),
            target_lengths=padded_target_lengths.cuda()
        )
        #print(_joint_output.shape)
        _loss = F.rnnt_loss(_joint_output.cuda(),
                            targets.cuda(),
                            _output_source_lengths.cuda(),
                            target_lengths.cuda(),
                            blank=0)

        return _loss
    
    def greedy_decode(self, sources:Tensor, source_lengths:list):
        assert len(source_lengths) == 1
        leng = source_lengths[0]
        
        source_lengths = torch.tensor(source_lengths, dtype=torch.int32)
        source_encodings, source_lengths = self.model.transcribe(sources.cuda(),
                                                        source_lengths.cuda())
        #print(source_encodings.shape)
        state = None
        target=torch.tensor([[0]]) # first zero
        targets = []
        target_encodings, target_lengths, state = self.model.predict(
            target.cuda(),
            torch.tensor([[1]]).cuda(),
            state
        )
        prev=-1
        for i in range(leng):
            outputs, output_source_lengths, output_target_lengths = self.model.join(source_encodings[:, i, :].unsqueeze(0),
                                                                                    torch.tensor([[1]]).cuda(),
                                                                                    target_encodings,
                                                                                    torch.tensor([[1]]).cuda()
                                                                                    )
            pred = torch.reshape(torch.argmax(outputs, dim=-1), (1,1))
            if pred != 0 and prev != pred:
                targets.append(pred.cpu().detach().numpy()[0,0])
                prev = pred.cpu().detach().numpy()[0,0]
            target = pred
            target_encodings, target_lengths, state = self.model.predict(
                target.cuda(),
                torch.tensor([[1]]).cuda(),
                state
            )

        return targets
    
if __name__=='__main__':
    import yaml
    
    path = './config.yaml'
    with open(path, 'r') as yf:
        config = yaml.safe_load(yf)
    model = ASRTransducer(config)
    inputs = torch.randn(1, 1000, 80, dtype=torch.float32)
    input_lengths = torch.tensor([1000,], dtype=torch.int32)
    print(inputs.shape)
    print(input_lengths.shape)
    outputs,_ = model.model.transcribe(inputs, input_lengths)
    print (outputs.shape)
    
