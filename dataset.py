from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import json
import pickle

class CustomDataset(Dataset):
    def __init__(self, src1, src2, trg):
        self.src1 = list(src1)
        self.src2 = list(src2)
        self.trg = list(trg)
        self.num_data = len(self.src1)
        
    def __getitem__(self, index):
        src1 = self.src1[index]
        src2 = self.src2[index]
        trg = self.trg[index]
        return src1, src2, trg
    
    def __len__(self):
        return self.num_data

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, pad_index=0, dim=0):
        self.dim = dim
        self.pad_index = pad_index

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.FloatTensor(*pad_size).fill_(self.pad_index)], dim=dim)
        
        (sequences1, sequences2, trg) = zip(*batch)
        batch_size = len(sequences1)
        
        ### for input_items desc
        # find longest sequence
        input_seq_len = max(map(lambda x: len(x), sequences1))
        
        # pad according to max_len
        sequences1 = [pad_tensor(torch.FloatTensor(seq), input_seq_len, self.dim) for seq in sequences1]
        sequences1 = torch.cat(sequences1)
        sequences1 = sequences1.view(batch_size, input_seq_len)

        sequences2 = [pad_tensor(torch.FloatTensor(seq), input_seq_len, self.dim) for seq in sequences2]
        sequences2 = torch.cat(sequences2)
        sequences2 = sequences1.view(batch_size, input_seq_len)
        
        # transpose to (Text_len*Batch size)
        #input_sequences = input_sequences.transpose(0, 1)
        trg = torch.FloatTensor(trg)
        
        return sequences1, sequences2, trg

    def __call__(self, batch):
        return self.pad_collate(batch)

def getDataLoader(dataset, batch_size, shuffle):
    return DataLoader(dataset, drop_last=True, batch_size=batch_size, collate_fn=PadCollate(),
                      shuffle=shuffle, pin_memory=True)