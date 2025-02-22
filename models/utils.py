#一些工具类
import itertools
import os
from collections import Counter
import dgl
import numpy as np
# from gensim.models import Word2Vec
# from rdkit import Chem
# from rdkit.Chem import MACCSkeys
import torch
from dgl.nn.pytorch import EdgeWeightNorm
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
# import forgi
# import forgi.graph.bulge_graph as fgb
import torch.nn.functional as F

nt_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'I': 0, 'H': 1, 'M': 2, 'S': 3}
rev_dict = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A', 'A': 'U', 'U': 'A'}
nts = ['A', 'C', 'G', 'T']

def pse_normalize(pse_knc):
    mu = np.mean(pse_knc, axis=0)
    sigma = np.std(pse_knc, axis=0)
    return (pse_knc - mu) / sigma

#对比学习指标
def do_CL(X, Y, T):
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)

    criterion = torch.nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, T)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    return CL_loss, CL_acc

#对比学习损失函数
def dual_CL(X, Y, T=0.3):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, T)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, T)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2

#序列对齐函数
def pad_batch(bg, max_input_len):
    num_batch = bg.batch_size
    graphs = dgl.unbatch(bg)
    h_node = bg.ndata['h']
    max_num_nodes = max_input_len
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()
    for i, g in enumerate(graphs):
        num_node = g.num_nodes()
        padded_h_node[-num_node:, i] = g.ndata['h']
        src_padding_mask[i, :max_num_nodes - num_node] = True

    return padded_h_node, src_padding_mask


def get_mask(bg):
    """
    :param bg:
    :return: batch_size * num_node  Ture if the degree is zero
    """
    num_batch = bg.batch_size
    in_d = bg.in_degrees() > 0
    o_d = bg.out_degrees() > 0
    mask = ~in_d & ~o_d
    mask = mask.view(num_batch, -1).to(bg.device)
    return mask

#kmer字符转数字
def kmer2num(kmer):
    ans = 0
    for i in range(len(kmer)):
        d = nt_dict[kmer[i]]
        ans = d + ans * 4
    return ans

#德布鲁因图构建函数
def de_Bruijn_graph(seq, k):
    """
    :param seq:序列表示
    :param k: kmer的k
    :return:
    """
    seq2id = []  # id seq

    num_node = 1 << 2 * k
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        n_id = kmer2num(kmer)
        seq2id.append(n_id)

    edge_freq = Counter(list(zip(seq2id[:-1], seq2id[1:])))
    graph = dgl.graph(list(edge_freq.keys()), num_nodes=num_node)
    graph.ndata['mask'] = torch.LongTensor(range(num_node))
    weight = torch.FloatTensor(list(edge_freq.values()))
    norm = EdgeWeightNorm(norm='both')
    norm_weight = norm(graph, weight)
    graph.edata['weight'] = norm_weight
    return graph


# def de_Bruijn_graph_2d(seq_2d, k):
#     """
#     :param seq_2d:二级结构序列表示
#     :param k: kmer的k
#     :return:
#     """
#     seq2id = []  # id seq
#     mask = []
#     cnt = 0
#     relabel_id = {}
#     number_node = 1 << 2 * k
#     for i in range(len(seq_2d) - k + 1):
#         kmer = seq_2d[i:i + k]
#         n_id = number_node if 'F' in kmer else number_node + 1 if 'T' in kmer else kmer2num(kmer)
#         if n_id not in relabel_id:
#             relabel_id[n_id] = cnt
#             cnt += 1
#         seq2id.append(relabel_id[n_id])
#         if n_id not in mask:
#             mask.append(n_id)
#     edge_freq = Counter(list(zip(seq2id[:-1], seq2id[1:])))
#     graph = dgl.graph(list(edge_freq.keys()))
#     graph.ndata['mask'] = torch.LongTensor(mask)
#     weight = torch.FloatTensor(list(edge_freq.values()))
#     norm = EdgeWeightNorm(norm='both')
#     norm_weight = norm(graph, weight)
#     graph.edata['weight'] = norm_weight
#     return graph


# def get_secondary_from_fx(src, dst):
#     """
#     Example:
#     get_secondary('new.fx','out.txt')
#
#     the new.fx contains:
#     >1y26
#     CGCUUCAUAUAAUCCUAAUGAUAUGGUUUGGGAGUUUCUACCAAGAGCCUUAAACUCUUGAUUAUGAAGUG
#     ((((((((((..(((((((.......)))))))......).((((((.......))))))..)))))))))
#     the out.txt will be:
#     >1y26
#     CGCUUCAUAUAAUCCUAAUGAUAUGGUUUGGGAGUUUCUACCAAGAGCCUUAAACUCUUGAUUAUGAAGUG
#     sssssssssmmmssssssshhhhhhhsssssssmmmmmmmmsssssshhhhhhhssssssmmsssssssss
#
#     :param src: src file
#     :param dst: dst file
#     :return:
#     """
#     cg = forgi.load_rna(src)
#     anno = fgb.BulgeGraph
#     with open(dst, 'w') as f:
#         for c in cg:
#             second = anno.to_element_string(c)
#             f.write(str('>' + c.name + '\n'))
#             f.write(str(c.seq + '\n'))
#             f.write(str(second + '\n'))
#             print(second)
#     f.close()


# def getfx(src, dst):
#     """
#
#     :param src: source fasta file
#     :param dst: fx file for cal secondary
#     :return:
#     """
#     tmp = 'tmp.fx'
#     main = f'"E:\ViennaRNA Package\RNAfold.exe" < {src} >{tmp} --noPS'
#     os.system(main)
#     with open(tmp, 'r') as f:
#         with open(dst, 'w') as fw:
#             cn = 0
#             for line in f:
#                 if cn % 3 != 2:
#                     fw.write(line)
#                 else:
#                     fw.write(str(line.split()[0] + '\n'))
#                 cn += 1
#     os.remove('tmp.fx')


# def get_secondary(src, dst):
#     """
#
#     :param src: src fasta file
#     :param dst: dst file
#     :return:
#     """
#     pre = time.time()
#     getfx(src, src.split('.')[0] + '.fx')
#     cur = time.time()
#     print(f'get docket bracket consume {format(cur - pre, ".1f")}s')
#     pre = cur
#     get_secondary_from_fx(src.split('.')[0] + '.fx', dst)
#     cur = time.time()
#     print(f'generate secondary consume {format(cur - pre, ".1f")}s')

#idng邻居节点
def get_neighbor(k):
    """

    :param k:
    :return: dict {kmer1:[neighbor1,neighbor2,...],kmer2:[...],...}
    """
    pools = itertools.product(['0', '1', '2', '3'], repeat=k)
    ans = {}
    for kmer in pools:
        ans[int(''.join(kmer), 4)] = []
        for i in range(len(kmer)):
            for j in range(1, 4):
                new_kmer = list(kmer)
                new_kmer[i] = str((int(kmer[i]) + j) % 4)
                ans[int(''.join(kmer), 4)].append(int(''.join(new_kmer), 4))
        ans[int(''.join(kmer), 4)].append(int(''.join(kmer), 4))
    return ans

#idng图
def idng(seq, k, neighbor_dict, window_size=20):
    """
    create an interval directed neighbor-based graph
    """

    def get_intersect(l1, l2):
        res = []
        p1 = 0
        p2 = 0
        while p1 < len(l1) and p2 < len(l2):
            if l1[p1] < l2[p2]:
                p1 += 1
            elif l1[p1] > l2[p2]:
                p2 += 1
            else:
                res.append(l1[p1])
                p1 += 1
                p2 += 1
        return res

    src = []
    dst = []
    num_node = 1 << 2 * k
    for i in range(len(seq) - k + 1):
        cur_kmer = seq[i:i + k]
        window = []
        for j in range(i + 1, i + window_size + 1):
            if j + k > len(seq):
                break
            window.append(kmer2num(seq[j:j + k]))
        neighbor = neighbor_dict[kmer2num(cur_kmer)]
        tmp_dst = get_intersect(sorted(neighbor), sorted(window))
        tmp_src = [kmer2num(cur_kmer)] * len(tmp_dst)
        src.extend(tmp_src)
        dst.extend(tmp_dst)
    edge_freq = Counter(list(zip(src, dst)))
    graph = dgl.graph(list(edge_freq.keys()), num_nodes=num_node)
    graph.ndata['mask'] = torch.LongTensor(range(num_node))
    weight = torch.FloatTensor(list(edge_freq.values()))
    norm = EdgeWeightNorm(norm='both')
    norm_weight = norm(graph, weight)
    graph.edata['weight'] = norm_weight
    return graph

#数字转kmer
def num2kmer(i, k):
    kmer = ''
    nt2int = ['A', 'C', 'G', 'T']
    while k > 0:
        kmer = nt2int[int(i % 4)] + kmer
        i /= 4
        k -= 1
    return kmer


#
# def getAllKmerFeat(k, path=None):
#     """
#     :param k: kmer
#     :param path: file saved path
#     :return:
#     """
#     if path is None:
#         path = f'../ckpt/MACCS_k{k}.pkl'
#     if os.path.exists(path):
#         print(f'exists {k}mer MACCS feat')
#         with open(path, 'rb') as f:
#             feat = pickle.load(f)
#         return feat
#     feat = [seq2MACCS(num2kmer(i, k)) for i in range(1 << (2 * k))]
#     feat = torch.concat(feat).reshape((1 << (2 * k)), -1)
#     with open(path, 'wb') as f:
#         pickle.dump(feat, f, protocol=4)
#     return feat

#
# def seq2MACCS(kmer):
#     mol = Chem.MolFromSequence(kmer)
#     fp = MACCSkeys.GenMACCSKeys(mol)
#     fea = torch.tensor(list(fp))
#     return fea

#根据形状子信息过滤
def generate_weight_by_shapelet(seq, shapelet_info, k=3):  # 1
    """
    :param seq: seq of rna
    :param shapelet_info: shapelet,score,tag
    :return:
    """

    ans = [0] * (1 << 2 * k)
    n = len(seq)
    for x, y in zip(range(n), range(k, n)):
        cur_word = seq[x:y]
        for shapelet_seq, score, _ in shapelet_info:
            if cur_word in shapelet_seq and ans[kmer2num(cur_word)] == 0:
                ans[kmer2num(cur_word)] += score
    return ans

#获得频率
def get_freq(seq):
    """
    2, 3, 4 ,5 mer 1360d
    kmer freq vector
    :param seq:
    :return:
    """
    freq = []

    for k in range(1, 6):
        tmp = torch.FloatTensor([0] * (1 << 2 * k))
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j + k]
            tmp[kmer2num(kmer)] += 1
        tmp /= (len(seq) - k + 1)
        freq.extend(tmp)

    return torch.FloatTensor(freq)

#反向互补序列
def get_rev(seq):
    """
    get the rev complement
    :param seq:
    :return:
    """
    return ''.join([rev_dict[seq[i]] for i in range(len(seq) - 1, -1, -1)])

#辅助函数，递归创建文件
def make_path(path):
    if os.path.exists(path): return
    if '/' not in path:
        os.makedirs(path)
        return
    make_path(path[:path.rindex('/')])
    os.makedirs(path)


def random_pseudo_rev(seq):
    return ''.join([nts[np.random.randint(4)] for _ in range(len(seq))])

#结果输出函数
def eval_output(model_perf, path):
    with open(os.path.join(path, f"Evaluate_Result_TestSet.txt"), 'w') as f:
        f.write("AUROC=%s\tAUPRC=%s\tAccuracy=%s\tMCC=%s\tRecall=%s\tPrecision=%s\tf1_score=%s\n" %
                (model_perf["auroc"], model_perf["auprc"], model_perf["accuracy"], model_perf["mcc"],
                 model_perf["recall"], model_perf["precision"], model_perf["f1"]))
        f.write("\n######NOTE#######\n")
        f.write(
            "#According to help_documentation of sklearn.metrics.classification_report:in binary classification, recall of the positive class is also known as sensitivity; recall of the negative class is specificity#\n\n")
        f.write(model_perf["class_report"])


# Evaluate performance of model
def evaluate_performance(y_test, y_pred, y_prob):
    # AUROC
    auroc = metrics.roc_auc_score(y_test, y_prob)
    auroc_curve = metrics.roc_curve(y_test, y_prob)
    # AUPRC
    auprc = metrics.average_precision_score(y_test, y_prob)
    auprc_curve = metrics.precision_recall_curve(y_test, y_prob)
    # Accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # MCC
    mcc = metrics.matthews_corrcoef(y_test, y_pred)

    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    class_report = metrics.classification_report(y_test, y_pred, target_names=["control", "case"])

    model_perf = {"auroc": auroc, "auroc_curve": auroc_curve,
                  "auprc": auprc, "auprc_curve": auprc_curve,
                  "accuracy": accuracy, "mcc": mcc,
                  "recall": recall, "precision": precision, "f1": f1,
                  "class_report": class_report}
    return model_perf


# Plot AUROC of model
def plot_AUROC(model_perf, path):
    # get AUROC,FPR,TPR and threshold
    roc_auc = model_perf["auroc"]
    fpr, tpr, threshold = model_perf["auroc_curve"]
    # return AUROC info
    temp_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    temp_df.to_csv(os.path.join(path, f"AUROC_info.txt"), header=True, index=False, sep='\t')
    # plot
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUROC of Models")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path, f"AUROC_TestSet.pdf"), format="pdf")

#
# def train_wv(seqs, path):
#     """
#     Parameters
#     ----------
#     seqs: the rna seq list
#     path: path to save model
#
#     Returns word2vec model
#     -------
#     """
#     vector_size = 64
#     window = 10
#     min_count = 5
#     sg = 1
#     epochs = 50
#     k = 4
#
#     sentences = []
#     for seq in seqs:
#         sentence = []
#         for i in range(len(seq) - k + 1):
#             sentence.append(seq[i:i + k])
#         sentences.append(sentence)
#     model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg, epochs=epochs)
#     model.save(path)
#     return model
