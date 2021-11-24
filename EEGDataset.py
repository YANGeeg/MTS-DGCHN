import torch
from torch.utils.data import Dataset, TensorDataset

class EEGDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor
    
    def __init__(self, x_tensor, y_tensor):
        
        self.x = x_tensor
        self.y = y_tensor
        
        assert self.x.size(0) == self.y.size(0)  #判断,如果不满足就抛出异常

    def __getitem__(self, index):
        return self.x[index], self.y[index]  #索引

    def __len__(self):
        return len(self.y)    #得到Y的长度
    
