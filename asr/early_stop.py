import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False,
                 delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_cer = None
        self.early_stop = False
        self.val_cer_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_cer, model):

        cer = val_cer
        if self.best_cer is None:
            self.best_cer = cer
            self.save_checkpoint(val_cer, model)
        elif cer > self.best_cer + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_cer = cer
            self.save_checkpoint(val_cer, model)
            self.counter = 0

        return self.early_stop
    
    def save_checkpoint(self, val_cer, model):
        if self.verbose:
            self.trace_func(f'Validation CER decreased ({self.val_cer_min:.6f} --> {val_cer:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_cer_min = val_cer
