#调参使用，大部分内容和main.py一致
import logging
import wandb
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
from models.utils import evaluate_performance
import argparse

warnings.filterwarnings("ignore")
sys.path.append("./models")

loss_fn = torch.nn.BCEWithLogitsLoss()

sigmoid = torch.nn.Sigmoid()


def train_step(dataloader, model, opt, config):
    model.train()
    tot_loss = 0
    for graph, label, sw, sf in dataloader:
        opt.zero_grad()
        logit = model(graph.to(config.device), sw.to(config.device), sf.to(config.device))
        label = torch.tensor(label).float()
        label = label.unsqueeze(-1).to(config.device)
        loss = loss_fn(logit, label)
        loss.backward()
        opt.step()
        tot_loss += loss
    return tot_loss


def test_performance(dataloader, model, threshold=0.5, config=None):
    model.eval()
    prob = []
    target = []
    tot = 0
    with torch.no_grad():
        for graph, label, sw, sf in dataloader:
            logit = model(graph.to(config.device), sw.to(config.device), sf.to(config.device))
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

#同main.py
def train_fine_tune():
    wandb.init()
    config = wandb.config
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    shapelet_size = 1 << 2 * config.shape_size
    shape_num = int(shapelet_size * config.shape_ratio)

    shapelet_info = get_shapelet(k_=config.shape_size)[:shape_num]

    train_dataset = idng_dataset(data_path=f'{config.data_path}/train.csv', k=config.k,
                                 data_save_path=config.data_save_path, shapelet_info=shapelet_info,
                                 reload=False, window_size=config.ws)


    kf = KFold(n_splits=10, shuffle=True, random_state=config.seed)

    best_auc = 0
    for i, (train_idx, dev_idx) in enumerate(kf.split(train_dataset)):
        train_sampler = SubsetRandomSampler(train_idx)
        dev_sampler = SubsetRandomSampler(dev_idx)
        train_dataloader = GraphDataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
        valid_dataloader = GraphDataLoader(train_dataset, batch_size=config.batch_size, sampler=dev_sampler)
        model = LncLoc(k=config.k,shape_num=shape_num, embed_dim=config.embed_dim, hidden_dim=config.hidden_dim)
        model.to(config.device)

        opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay)

        for epoch in range(config.epoch):
            print(f'fold:{i + 1} epoch{epoch + 1}')
            logging.info(f'fold:{i + 1} epoch{epoch + 1}')
            train_step(train_dataloader, model, opt, config)
            print('train')
            test_performance(train_dataloader, model, config=config)
            print('dev')
            ans = test_performance(valid_dataloader, model, config=config)

            wandb.log({
                'epoch': epoch,
                'acc': ans["accuracy"],
                'auc': ans["auroc"],
                'f1': ans["f1"],
                'mcc': ans["mcc"],
                'best_auc': best_auc
            })
            if ans["auroc"] > best_auc:
                best_auc = ans["auroc"]
                save_dir = config.res_dir + f'/{best_auc:.4f}'
                make_path(save_dir)
                torch.save(model.state_dict(), f'{save_dir}/best.pt')
            print(f'best_auc {best_auc}')


if __name__ == '__main__':
    logging.basicConfig(filename="train.log",
                        filemode="a",
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S",
                        level=logging.INFO)

    parser = argparse.ArgumentParser(description="input the window_size")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    #wandb相关的设置
    sweep_configuration = {
        'method': 'random',
        'name': 'swp_casual',
        'metric': {'goal': 'maximize', 'name': 'best_auc'},
        'parameters':
            {
                'seed': {
                    'value': 0
                },
                'shape_size': {
                    'values': [4, 5, 6, 7, 8]
                },
                'shape_ratio': {
                    'values':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
                },
                'k': {
                    'values': [3, 4]
                },
                'device': {
                    'value': f'cuda:{args.gpu}'
                },
                'batch_size': {'value': 256},
                'lr': {
                    'values': [0.001, 0.0001, 0.00001]
                },
                'epoch': {
                    'value': 200
                },
                'decay': {
                    'distribution': 'uniform',
                    'max': 0.001,
                    'min': 0.000001
                },
                'embed_dim': {
                    'values': [16, 32, 64, 128, 256, 512]
                },
                'hidden_dim': {
                    'values': [16, 32, 64, 128, 256, 512, 1024, 2048]
                },
                'ws':{
                    'values':[20,30,40,50,60,70]
                },
                'data_path': {
                    'value': 'data/rnalight'
                },
                'data_save_path': {
                    'value': 'ckpt/finetune/rnalight'
                },
                'res_dir': {
                    'value': 'result/'
                }
            },

    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project='g_withShape'
    )
    wandb.agent(sweep_id, function=train_fine_tune, count=200)
