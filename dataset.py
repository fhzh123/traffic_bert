# Import PyTorch
import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, src, trg):
        self.src = list(src)
        self.trg = list(trg)
        self.num_data = len(self.src1)
        
    def __getitem__(self, index):
        src = self.src[index]
        src_rev = self.src[::-1][index]
        trg = self.trg[index]
        return src, src_rev, trg
    
    def __len__(self):
        return self.num_data

def getDataLoader(dataset, batch_size, shuffle):
    return DataLoader(dataset, drop_last=True, batch_size=batch_size, shuffle=shuffle, pin_memory=True)