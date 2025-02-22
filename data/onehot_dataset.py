import numpy as np
import pandas as pd
from torch.utils import data

#one-hot编码对应数据集
class onehot_dataset(data.Dataset):
    def __init__(self, df):
        self.nt_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.x = df['code'].apply(self.myfun).values
        self.y = df['Value'].values

    def myfun(self, code):
        ans = []
        for nt in code:
            ans.append(self.nt_dict[nt])
        ans = ans[:4000]
        if len(ans) < 4000:
            ans = ans + [4] * (4000 - len(ans))
        return np.array(ans)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


