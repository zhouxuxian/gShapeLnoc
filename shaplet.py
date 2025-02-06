import pickle
from collections import defaultdict

from tqdm import trange
import pandas as pd

#得分函数1
def score_function(item):
    return (max(item[1][0], item[1][1]) + 1) / (min(item[1][0], item[1][1]) + 1)

#得分函数2
def score_function2(item):
    return (max(item[1][0], item[1][1]) + 1) / (min(item[1][0], item[1][1]) + 1) * (
            2 * int(item[1][0] < item[1][1]) - 1)

#获取shapelet,路径为数据集路径，格式为制表符分隔
def get_shapelet(path='data/rnalight/train.csv', k_=4):
    df = pd.read_csv(path, sep='\t')
    TrainSet = [seq for seq in df['code'].values]
    label = df['Value'].values
    k_dim = [k_]

    tot = defaultdict(lambda: [0, 0])
    vis = defaultdict(lambda: -1)
    for RNA_id in trange(len(TrainSet)):
        RNA = TrainSet[RNA_id]
        c_id = label[RNA_id]
        for k in k_dim:
            for start in range(len(RNA) - k + 1):
                cur_word = RNA[start:start + k]
                if vis[cur_word] == RNA_id:
                    continue
                vis[cur_word] = RNA_id
                tot[cur_word][c_id] += 1

    ans = sorted(tot.items(), key=lambda item: score_function(item), reverse=True)
    ans = [[item[0], score_function(item), int(item[1][0] < item[1][1])] for item in ans]
    with open('shapeInfo.pkl', 'wb') as f:
        pickle.dump(ans, f)
    return ans

#利用shapelet提取motif所用
def get_shapelet_motif(path='data/rnalight/train.csv', k_=[4]):
    df = pd.read_csv(path, sep='\t')
    TrainSet = [seq for seq in df['code'].values]
    label = df['Value'].values
    k_dim = k_

    tot = defaultdict(lambda: [0, 0])
    vis = defaultdict(lambda: -1)
    for RNA_id in trange(len(TrainSet)):
        RNA = TrainSet[RNA_id]
        c_id = label[RNA_id]
        for k in k_dim:
            for start in range(len(RNA) - k + 1):
                cur_word = RNA[start:start + k]
                if vis[cur_word] == RNA_id:
                    continue
                vis[cur_word] = RNA_id
                tot[cur_word][c_id] += 1

    ans = sorted(tot.items(), key=lambda item: score_function2(item), reverse=True)
    ans = [[item[0], score_function2(item), int(item[1][0] < item[1][1])] for item in ans]

    return [item[0] for item in ans if item[2] > 0], [item[0] for item in ans if item[2] == 0][::-1]
