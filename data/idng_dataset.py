import itertools
import random

import torch
from dgl.data import DGLDataset
import pickle
from tqdm import tqdm
from tqdm.contrib import tzip
from models.utils import get_neighbor, idng, generate_weight_by_shapelet, make_path
import pandas as pd
import numpy as np
import os

"""
interval directed neighbor-based graph lncRNADataset
"""


class idng_dataset(DGLDataset):
    def __init__(self, data_path, k, shapelet_info, data_save_path, window_size=40, reload=False):
        self.reload = reload
        self.shapelet_info = shapelet_info
        self.data_save_path = data_save_path
        self.rna_seq = []
        self.data_path = data_path
        self.k = k
        self.graph = []
        self.shape_weight = []
        self.labels = []
        self.shape_freq = []
        self.window_size = window_size
        super(idng_dataset, self).__init__(name='idng_dataset', save_dir=self.data_save_path)
        print('***Loading Data ***')

    def process(self):
        if len(self.shapelet_info) == 0:
            kmers = [''.join(p) for p in itertools.product(['A', 'C', 'G', 'T'], repeat=self.k)]
            self.shapelet_info = [(s, 1, 1) for s in kmers]  # seq,score,random

        #  获得所有rna序列
        df = pd.read_csv(self.data_path, sep='\t')
        rna_seq = df['code'].values
        self.labels = df['Value'].values
        neighbor_dict = get_neighbor(self.k)
        print('generate graphs....................................')

        for i, (lab, seq) in enumerate(tzip(self.labels, rna_seq, desc='generate graphs')):
            graph = idng(seq, self.k, neighbor_dict, window_size=self.window_size)
            sw = generate_weight_by_shapelet(seq, self.shapelet_info, k=self.k)
            #保存形状子信息
            self.shape_weight.append(sw)
            self.graph.append(graph)
            ##############################
            self.shape_freq.append([seq.count(item[0]) for item in self.shapelet_info])

    def __getitem__(self, idx):
        return self.graph[idx], self.labels[idx], torch.FloatTensor(self.shape_weight[idx]), \
               torch.FloatTensor(self.shape_freq[idx])

    def __len__(self):
        return len(self.graph)
    #数据集保存
    def save(self):
        tqdm.write('==========save processed data============')
        graph_info = {'graph': self.graph, 'labels': self.labels, 'shape_weight': self.shape_weight,
                      'shape_freq': self.shape_freq}
        make_path(self.data_save_path)
        with open(f'{self.data_save_path}/{self.data_path.split("/")[-1].split(".")[0]}_k{self.k}_idng.pkl', 'wb') as f:
            pickle.dump(graph_info, f)
        f.close()
    #数据集加载
    def load(self):
        tqdm.write('==========load processed data============')
        with open(f'{self.data_save_path}/{self.data_path.split("/")[-1].split(".")[0]}_k{self.k}_idng.pkl',
                  'rb') as f:
            graph_info = pickle.load(f)
            self.graph = graph_info['graph']
            self.labels = graph_info['labels']
            self.shape_weight = graph_info['shape_weight']
            self.shape_freq = graph_info['shape_freq']

        f.close()
    #判断是否存在
    def has_cache(self):
        return not self.reload and os.path.exists(
            f'{self.data_save_path}/{self.data_path.split("/")[-1].split(".")[0]}_k{self.k}_idng.pkl')
