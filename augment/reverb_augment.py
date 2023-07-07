import pyroomacoustics as pra
import numpy as np
import os
import torchaudio
import torch
import torch.nn as nn

class ReverbAugment(nn.Module):
    def __init__(self, sample_rate,  
                 room_size:list, 
                 mic_loc:list, 
                 source_loc:list,
                 loc_range:list,
                 min_rt60=0.2,
                 max_rt60=1.0,
                 snr=60) -> None: 
        
        self.snr = snr
        self.min_rt60=min_rt60
        self.max_rt60=max_rt60

        self.material = pra.make_materials(
            ceiling="ceiling_plasterboard",
            floor="audience_0.72_m2",
            east="glass_window",
            west="gypsum_board",
            north="gypsum_board",
            south="gypsum_board",
        )
        
        self.sample_rate = sample_rate
        self.room_size = room_size
        self.mic_loc = mic_loc
        self.source_loc = source_loc
        self.loc_range = loc_range

        assert self.source_loc + self.loc_range < self.room_size and self.source_loc - self.loc_range < self.room_size

    def forward(self, source):
        # tensor -> 1d-ndarray
        source = source.squeeze().cpu().detach().numpy()
        original_length = len(source)

        rt60 = (self.max_rt60 - self.min_rt60) * np.random.rand() + self.min_rt60
        _, max_order = pra.inverse_sabine(rt60, self.room_size)

        room = pra.ShoeBox(
            self.room_size, fs=self.sample_rate, 
            materials=self.material, 
            max_order=max_order
        )
        room.add_microphone(self.mic_loc)

        source_loc = self.source_loc + self.loc_range * 2 * (np.random.rand() - .5)
        room.add_source(source_loc, signal=source)

        room.simulate(self.snr)
        target = room.mic_array.signals[0, :]
        target = torch.from_numpy(target[:original_length,], torch.float32).unsqueeze(0)
        
        return target