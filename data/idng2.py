import itertools
import random

import torch
from dgl.data import DGLDataset
import pickle
from tqdm import tqdm
from tqdm.contrib import tzip
from models.utils import get_neighbor, idng, generate_weight_by_shapelet, make_path, de_Bruijn_graph
import pandas as pd
import numpy as np
import os

"""
GTShapelet2所用数据集
interval directed neighbor-based graph lncRNADataset
"""


class combine_graph(DGLDataset):
    def __init__(self, data_path, k, shapelet_info, data_save_path, window_size=40, reload=False):
        self.reload = reload
        self.shapelet_info = shapelet_info
        self.data_save_path = data_save_path
        self.rna_seq = []
        self.data_path = data_path
        self.k = k
        self.graph = []
        self.dbg_graph=[]
        self.shape_weight = []
        self.labels = []
        self.shape_freq = []
        self.window_size = window_size
        super(combine_graph, self).__init__(name='combine_graph', save_dir=self.data_save_path)
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
            graph2 = de_Bruijn_graph(seq, self.k)
            sw = generate_weight_by_shapelet(seq, self.shapelet_info, k=self.k)
            self.dbg_graph.append(graph2)
            self.shape_weight.append(sw)
            self.graph.append(graph)

            ##############################
            self.shape_freq.append([seq.count(item[0]) for item in self.shapelet_info])

    def __getitem__(self, idx):
        return self.graph[idx],self.dbg_graph[idx], self.labels[idx], torch.FloatTensor(self.shape_weight[idx]), \
               torch.FloatTensor(self.shape_freq[idx])

    def __len__(self):
        return len(self.graph)

    def save(self):
        tqdm.write('==========save processed data============')
        graph_info = {'graph': self.graph,'dbg_graph':self.dbg_graph, 'labels': self.labels, 'shape_weight': self.shape_weight,
                      'shape_freq': self.shape_freq}
        make_path(self.data_save_path)
        with open(f'{self.data_save_path}/{self.data_path.split("/")[-1].split(".")[0]}_k{self.k}_comb.pkl', 'wb') as f:
            pickle.dump(graph_info, f)
        f.close()

    def load(self):
        tqdm.write('==========load processed data============')
        with open(f'{self.data_save_path}/{self.data_path.split("/")[-1].split(".")[0]}_k{self.k}_comb.pkl',
                  'rb') as f:
            graph_info = pickle.load(f)
            self.graph = graph_info['graph']
            self.dbg_graph = graph_info['dbg_graph']
            self.labels = graph_info['labels']
            self.shape_weight = graph_info['shape_weight']
            self.shape_freq = graph_info['shape_freq']

        f.close()

    def has_cache(self):
        return not self.reload and os.path.exists(
            f'{self.data_save_path}/{self.data_path.split("/")[-1].split(".")[0]}_k{self.k}_comb.pkl')
