import logging
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from models.utils import make_path
import torch
from dgl.dataloading import GraphDataLoader
import numpy as np
from shaplet import get_shapelet
import warnings
import sys
from models.Model import LncLoc
from data.idng_dataset import idng_dataset
from models.utils import evaluate_performance, eval_output, plot_AUROC
from config import config

warnings.filterwarnings("ignore")
sys.path.append("./models")

#损失函数
loss_fn = torch.nn.BCEWithLogitsLoss()
sigmoid = torch.nn.Sigmoid()

#训练迭代
def train_step(dataloader, model, opt, config):
    model.train()
    tot_loss = 0
    for graph, label, sw,sf in dataloader:
        opt.zero_grad()
        logit = model(graph.to(config.device), sw.to(config.device),sf.to(config.device))
        label = torch.tensor(label).float()
        label = label.unsqueeze(-1).to(config.device)
        loss = loss_fn(logit, label)
        loss.backward()
        opt.step()
        tot_loss += loss
    return tot_loss

#测试
def test_performance(dataloader, model, threshold=0.5, config=None):
    model.eval()
    prob = []
    target = []
    tot = 0
    with torch.no_grad():
        for graph, label, sw,sf in dataloader:
            logit = model(graph.to(config.device), sw.to(config.device),sf.to(config.device))
            label = torch.tensor(label).float()
            label = label.unsqueeze(-1).to(config.device)
            loss = loss_fn(logit, label)
            tot += loss
            prob.extend(sigmoid(logit).to('cpu').numpy())
            target.extend(label.to('cpu').numpy())
        pred = [int(p >= threshold) for p in prob]
        ans = evaluate_performance(np.array(target), np.array(pred), np.array(prob))

        print(f'auroc: {ans["auroc"]} auprc :{ans["auprc"]}  acc:{ans["accuracy"]} mcc:{ans["mcc"]} '
              f'f1:{ans["f1"]}')
        return ans

#训练主函数
def train_fine_tune():
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    shapelet_info = []
    if config.use_shapelet == 1: #形状子信息
        shapelet_info = get_shapelet(k_=4)[:200]

    #创建数据集
    train_dataset = idng_dataset(data_path=f'{config.data_path}/train.csv', k=config.k,
                                 data_save_path=config.data_save_path, shapelet_info=shapelet_info,
                                 reload=config.reload, window_size=config.window_size)
    
    #10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=config.seed)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=config.batch_size)
    best_auc = 0
    # k-flod
    for i, (train_idx, dev_idx) in enumerate(kf.split(train_dataset)):
        #训练、验证子数据集的采样下标
        train_sampler = SubsetRandomSampler(train_idx)
        dev_sampler = SubsetRandomSampler(dev_idx)
        #idng数据集
        train_dataloader = GraphDataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
        valid_dataloader = GraphDataLoader(train_dataset, batch_size=config.batch_size, sampler=dev_sampler)
        model = LncLoc(k=config.k, embed_dim=config.embed_dim, hidden_dim=config.hidden_dim)
        model.to(config.device)
        #优化器
        opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay)

        for epoch in range(config.epoch):
            print(f'fold:{i + 1} epoch{epoch + 1}')
            logging.info(f'fold:{i + 1} epoch{epoch + 1}')
            train_step(train_dataloader, model, opt, config)
            print('train')
            test_performance(train_dataloader, model, config=config)
            print('dev')
            ans = test_performance(valid_dataloader, model, config=config)
            if ans["auroc"] > best_auc:
                best_auc = ans["auroc"]
                save_dir = config.res_dir + f'/{best_auc:.4f}'
                make_path(save_dir)
                torch.save(model.state_dict(), f'{save_dir}/best.pt')
                # model_performance = test_performance(test_dataloader, model, config=config)
                # eval_output(model_performance, path=save_dir)
                # plot_AUROC(model_performance, path=save_dir)


if __name__ == '__main__':
    logging.basicConfig(filename="train.log",
                        filemode="a",
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S",
                        level=logging.INFO)

    config = config()
    train_fine_tune()
