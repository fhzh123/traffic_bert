# Import PyTorch
import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg
        self.num_data = len(self.src)
        
    def __getitem__(self, index):
        src = self.src[index]
        src_rev = self.src[::-1][index]
        trg = self.trg[index]
        return src, src_rev, trg
    
    def __len__(self):
        return self.num_data

class Transpose_tensor:
    def __init__(self, dim=1):
        self.dim = dim

    def transpose_tensor(self, batch):
        (src, src_rev, trg) = zip(*batch)
        batch_size = len(src)
        #
        # src = torch.cat(src).view(-1, batch_size).transpose(0, 1)
        # src_rev = torch.cat(src_rev, dim=self.dim).view(-1, batch_size).transpose(0, 1)
        # trg = torch.cat(trg).view(-1, batch_size).transpose(0, 1)

        return torch.tensor(src), torch.tensor(src_rev), torch.tensor(trg)

    def __call__(self, batch):
        return self.transpose_tensor(batch)

class PadCollate:
    def __init__(self, pad_index=0, dim=0, src_max_len=None, trg_max_len=None):
        self.dim = dim
        self.pad_index = pad_index
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)
        
        (src, src_rev, trg) = zip(*batch)
        batch_size = len(src)
        
        ### for input_items desc
        # find longest sequence
        src = max(map(lambda x: len(x), src))
        
        # pad according to max_len
        src = [pad_tensor(torch.LongTensor(seq), input_seq_len, self.dim) for seq in src]
        src = torch.cat(src)
        src = input_sequences.view(batch_size, src)

        src_rev = max(map(lambda x: len(x), src_rev))
        
        # pad according to max_len
        src_rev = [pad_tensor(torch.LongTensor(seq), input_seq_len, self.dim) for seq in src_rev]
        src_rev = torch.cat(src_rev)
        src_rev = src_rev.view(batch_size, input_seq_len)

        trg = max(map(lambda x: len(x), trg))
        
        # pad according to max_len
        trg = [pad_tensor(torch.LongTensor(seq), output_seq_len, self.dim) for seq in trg]
        trg = torch.cat(trg)
        trg = trg.view(batch_size, output_seq_len)
        
        return src, src_rev, trg

    def __call__(self, batch):
        return self.pad_collate(batch)

def getDataLoader(dataset, batch_size, shuffle):
    return DataLoader(dataset, drop_last=True, batch_size=batch_size, collate_fn=Transpose_tensor(),
                      shuffle=shuffle, pin_memory=True) 