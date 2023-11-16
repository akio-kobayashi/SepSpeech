import numpy as np
import sys, os, re, gzip, struct
import random
import h5py

def compute_norm(h5fd):
    keys=h5fd.keys()
    rows=0
    mean=None
    std=None

    for key in keys:
        if key == 'mean':
            continue
        if key == 'std':
            continue
        mat = h5fd[key+'/data'][()]
        rows += mat.shape[0]
        if mean is None:
            mean=np.sum(mat, axis=0).astype(np.float64)
            std=np.sum(np.square(mat), axis=0).astype(np.float64)
        else:
            mean=np.add(np.sum(mat, axis=0).astype(np.float64), mean)
            std=np.add(np.sum(np.square(mat), axis=0).astype(np.float64), std)

    mean = mean/rows
    std = np.sqrt(std/rows - np.square(mean))

    return mean, std

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, path, keypath=None, stats=None, augment=False,
                expand=False, cut=0,
                scale=4, crop=0, train=True):
        super(SpeechDataset, self).__init__()

        self.scale=scale
        self.augment=augment
        self.expand=expand
        self.crop=crop
        self.train=train

        '''
            Read 80-dimentional log-mel-spectal features
        '''
        self.h5fd = h5py.File(path, 'r')
        if stats is None:
            self.mean, self.var = compute_norm(self.h5fd)
        else:
            self.mean, self.var = stats
        #self.mean = self.h5fd['mean'][()]
        #self.var = self.h5fd['var'][()]
        self.keys=[]

        if keypath is not None:
            with open(keypath,'r') as f:
                lines=f.readlines()
                for l in lines:
                    l=l.strip()
                    if cut > 0:
                        mat = self.h5fd[l+'/data'][()]
                        if mat.shape[0] > cut:
                            continue
                    self.keys.append(l)
        else:
            for key in self.h5fd.keys():
                if key != 'mean' and key != 'var':
                    if cut > 0:
                        mat = self.h5fd[key+'/data'][()]
                        if mat.shape[0] > cut:
                            continue
                    self.keys.append(key)

    def get_stats(self):
        return self.mean, self.var

    def get_keys(self):
        return self.keys

    def __len__(self):
        return len(self.keys)

    def input_size(self):
        mat = self.h5fd[self.keys[0]+'/data'][()]
        return mat.shape[1]

    def get_data(self, keys):
        data=[]
        for key in keys:
            dt = self.__getitem__(self.keys.index(key))
            data.append(dt)
        _data=data_processing(data)
        return _data

    '''
        Get features from HDF5 file
        input:  80-dim log-mel
        label:  phoneme label
    '''
    def __getitem__(self, idx):
        # original or fading : normalized log-spectral features
        # (time, feature)
        input=self.h5fd[self.keys[idx]+'/data'][()]
        # expand
        if self.expand:
            length=input.shape[0]
            if length%self.scale != 0:
                length=(input.shape[0]//self.scale+1)*self.scale
            if length != input.shape[0]:
                # expand matrix, fill zeros
                mat=np.zeros((length, input.shape[1]))
                mat[0:input.shape[0], :] = input[:, :]
                # copy last flame feats
                #for l in range(length-input.shape[0]):
                #    mat[input.shape[0]+l, :] = input[input.shape[0]-1,:]
                input = mat

        # randomly crop
        if self.crop > 0:
            max_len = input.shape[0]
            if max_len < 128:
                mat=np.zeros((128, input.shape[1]))
                mat[0:input.shape[0], :] = input[:, :]
                input = mat
            else:
                if self.train:
                    start=random.randint(0, max_len-self.crop)
                    mat = input[start:start+self.crop, :]
                else:
                    mid=max_len//2
                    mat = input[mid-self.crop//2: mid+self.crop//2, :]
                input = mat

        ref=input.copy()
        if self.augment:
            x = torch.from_numpy(np.expand_dims(np.transpose(input),axis=0).astype(np.float32)).clone()
            input=spec_augment_pytorch.spec_augment(x,frequency_masking_para=8)
            input = x.to('cpu').detach().numpy().copy()
            input=np.transpose(np.squeeze(input))
            aug_mask=input==0.0
            input -= self.mean
            input /= self.var
            input[aug_mask]=0.0 # masked feature value = (0,0, 1.0)
        else:
            input -= self.mean
            input /= self.var

        label=self.h5fd[self.keys[idx]+'/label'][()]

        return input, ref, label, self.keys[idx]

def expand_storage(data, mask, length):
    # expand matrix, fill zeros
    mat=np.zeros((data.shape[0], length, data.shape[2]))
    mat[:, 0:data.shape[1], :] = data[:,:,:]
    data = mat
    mat=np.zeros((mask.shape[0], length, mask.shape[2]))
    mat[:, 0:mask.shape[1], :] = mask[:,:,:]
    mask = mat

    data=torch.from_numpy(data.astype(np.float32)).clone()
    mask=torch.from_numpy(mask.astype(np.float32)).clone()

    return data, mask

'''
    data_processing
    Return inputs, labels, input_lengths, label_lengths, outputs
'''
def data_processing(data, data_type="train"):
    inputs = []
    refs = []
    labels = []
    input_lengths=[]
    label_lengths=[]
    keys = []

    for input, ref, label, key in data:
        """ inputs : (batch, time, feature) """
        # w/o channel
        inputs.append(torch.from_numpy(input.astype(np.float32)).clone())
        refs.append(torch.from_numpy(ref.astype(np.float32)).clone())
        labels.append(torch.from_numpy(label.astype(np.int)).clone())
        input_lengths.append(input.shape[0])
        label_lengths.append(len(label))
        keys.append(key)

    #lgt = [l // 4 for l in input_lengths]
    #input_lengths = lgt

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    refs = nn.utils.rnn.pad_sequence(refs, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    masks=np.zeros(inputs.shape)
    for n in range(len(input_lengths)):
        masks[n, 0:input_lengths[n], :] = 1
    masks=torch.from_numpy(masks.astype(np.float32)).clone()

    return inputs, refs, labels, input_lengths, label_lengths, masks, keys
