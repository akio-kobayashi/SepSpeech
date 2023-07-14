import numpy as np
from enum import IntEnum

class StateType(IntEnum):
    Good=1
    Bad=0
    
class State:
    def __init__(self, state_type, p_trans):
        self.state_type = state_type
        self.p_trans = p_trans

    def transition(self):
        if np.random.rand() > self.p_trans:
            return self.state_type
        else:
            return StateType[StateType(1 - int(self.state_type)).name]
        
class GilbertElliotModel:
    def __init__(self, plr=0.1, lmd=0.5, p_g=0.0, p_b=0.5):
        p_alpha = (1.-lmd) * (1. - (p_b - plr)/(p_b - p_g))
        #print(f'P_alpha = {p_alpha:.3f}')
        p_beta = (1.-lmd) * (p_b - plr)/(p_b - p_g)
        #print(f'P_beta  = {p_beta:.3f}')
        
        assert p_alpha > 0 and p_beta > 0

        self.states = {}
        self.states[StateType.Good] = State(StateType.Good, p_alpha)
        self.states[StateType.Bad] = State(StateType.Bad, p_beta)
        
    def simulate(self, sequence_length):
        sequence = [StateType.Good.value]

        current_state = StateType.Good
        for n in range(sequence_length-1):
            state = self.states[current_state].transition()
            sequence.append(state.value)
            current_state = state

        assert len(sequence) == sequence_length

        return sequence
    
if __name__ == '__main__':
    model = GilbertElliotModel()
    seq = model.simulate(100)
    print(seq)
