import numpy as np
import pandas as pd
from torch.utils import data

#kmer频率数据集
class kmer_dataset(data.Dataset):
    def __init__(self,df,k=4):
        self.k = k
        self.x = df['code'].apply(self.myfun).values
        self.y = df['Value'].values

    def myfun(self, code):
        ans = []
        for i in range(len(code)-self.k+1):
            ans.append(code[i:i+self.k])
        return ans

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.x)


