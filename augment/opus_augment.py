from pyogg import OpusEncoder
from pyogg import OpusDecoder
import numpy as np
import os
import torchaudio
import torch
import torch.nn as nn
import librosa

class OpusAugment(nn.Module):
    def __init__(self, sample_rate, frame_duration, min_bps, max_bps, 
                 min_packet_loss_rate, max_packet_loss_rate,
                 decode_missing_packet_rate, **kwargs):
        super().__init__()
        self.min_bps = min_bps
        self.max_bps = max_bps
        
        self.samples_per_second = sample_rate
        self.min_packet_loss_rate = min_packet_loss_rate
        self.max_packet_loss_rate = max_packet_loss_rate
        self.decode_missing_packet_rate = decode_missing_packet_rate

        self.channels = 1
        self.bytes_per_sample = 2
        self.desired_frame_duration = frame_duration/1000 # 20 msec
        #self.desired_frame_size = int(self.desired_frame_duration*self.samples_per_second)

    def forward(self, x):

        bps = np.random.randint(self.min_bps, self.max_bps)
        if bps < 12000:
            target_samples_per_second = 8000
        elif bps < 15000:
            target_samples_per_second = 12000
        else:
            target_samples_per_second = 16000
        self.desired_frame_size = int(self.desired_frame_duration*target_samples_per_second)
        original_length = x.shape[-1]
        x = torchaudio.functional.resample(x, self.samples_per_second, target_samples_per_second)
        
        opus_encoder = OpusEncoder()
        opus_encoder.set_application("voip") # 'voip' 'audio' 'restricted_lowdelay'
        #opus_encoder.set_sampling_frequency(self.samples_per_second)
        opus_encoder.set_sampling_frequency(target_samples_per_second)
        opus_encoder.set_channels(self.channels)
        mx = int(bps /(1000*self.desired_frame_duration*8))
        opus_encoder.set_max_bytes_per_frame(mx)

        opus_decoder = OpusDecoder()
        opus_decoder.set_channels(self.channels)
        #opus_decoder.set_sampling_frequency(self.samples_per_second)
        opus_decoder.set_sampling_frequency(target_samples_per_second)

        #original_length = x.shape[-1] # (channels, length)
        wave_samples = x.cpu().detach().numpy().squeeze()
        wave_samples = np.array([ int(s*32768) for s in wave_samples]).astype(np.int16).tobytes()

        packet_loss_rate = np.random.rand() * (self.max_packet_loss_rate - self.min_packet_loss_rate) + self.min_packet_loss_rate
        
        start, end = 0, self.desired_frame_size * self.bytes_per_sample
        decoded = []
        fec=False
        if np.random.rand() < self.decode_missing_packet_rate:
            fec=True

        #print(f'bitrate: {bps}, sample rate: {target_samples_per_second}, packet loss: {packet_loss_rate:.3f}, FEC: {fec}')
        
        while True:
            if start >= end:
                break
                
            pcm = wave_samples[start:end]
            missing = False
            if np.random.rand() < packet_loss_rate:
                missing=True

            if len(pcm) == 0:
                break

            effective_frame_size = (
                len(pcm) # bytes
                // self.bytes_per_sample
                // self.channels
            )
            if effective_frame_size < self.desired_frame_size:
                pcm += (
                    b"\x00"
                    * ((self.desired_frame_size - effective_frame_size)
                    * self.bytes_per_sample
                    * self.channels)
                )

            encoded_packet = opus_encoder.encode(pcm)
            
            if missing and start > 0:
                if fec:
                    decoded_pcm = opus_decoder.decode_missing_packet(20)
                else:
                    decoded_pcm = b"\x00" * self.desired_frame_size * self.bytes_per_sample * self.channels
            else:
                decoded_pcm = opus_decoder.decode(encoded_packet)

            decoded.append(librosa.util.buf_to_float(decoded_pcm))

            start += self.desired_frame_size*self.bytes_per_sample
            end += self.desired_frame_size*self.bytes_per_sample
            if end >= len(wave_samples):
                end = len(wave_samples)

        decoded_pcm = np.array(decoded).reshape(1, -1)[:, :original_length] # float

        decoded_pcm = torch.from_numpy(decoded_pcm)
        decoded_pcm = torchaudio.functional.resample(decoded_pcm, target_samples_per_second, self.samples_per_second)
        
        if decoded_pcm.shape[-1] < original_length:
            zeros = torch.zeros(1, original_length - decoded_pcm.shape[-1])
            decoded_pcm = torch.cat([decoded_pcm, zeros], dim=-1)
        elif decoded_pcm.shape[-1] > original_length:
            decoded_pcm = decoded_pcm[:, :original_length]
            
        #print(f'bitrate: {bps}, sample rate: {target_samples_per_second}, packet loss: {packet_loss_rate:.3f}, FEC: {fec}')
        return decoded_pcm, bps, target_samples_per_second
        
