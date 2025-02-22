# 两类边进行传播的GTShapelet2的训练代码
# import logging
# from sklearn.model_selection import KFold
# from torch.utils.data import SubsetRandomSampler
# from models.utils import make_path
# import torch
# from dgl.dataloading import GraphDataLoader
# import numpy as np
# from shaplet import get_shapelet
# import warnings
# import sys
# from models.Model2 import LncLoc
# from data.combinegraph_dataset import combine_graph
# from models.utils import evaluate_performance, eval_output, plot_AUROC
# import argparse
# warnings.filterwarnings("ignore")
# sys.path.append("./models")
# #########wandb调参
# 
# class config:
#     def __init__(self):
#         super(config, self).__init__()
# 
# 
# loss_fn = torch.nn.BCEWithLogitsLoss()
# 
# sigmoid = torch.nn.Sigmoid()
# 
# 
# def train_step(dataloader, model, opt, config):
#     model.train()
#     tot_loss = 0
#     for graph,g2, label, sw,sf in dataloader:
#         opt.zero_grad()
#         logit = model(graph.to(config.device),g2.to(config.device), sw.to(config.device),sf.to(config.device))
#         label = torch.tensor(label).float()
#         label = label.unsqueeze(-1).to(config.device)
#         loss = loss_fn(logit, label)
#         loss.backward()
#         opt.step()
#         tot_loss += loss
#     return tot_loss
# 
# 
# def test_performance(dataloader, model, threshold=0.5, config=None):
#     model.eval()
#     prob = []
#     target = []
#     tot = 0
#     with torch.no_grad():
#         for graph,g2, label, sw,sf in dataloader:
#             logit = model(graph.to(config.device),g2.to(config.device), sw.to(config.device),sf.to(config.device))
#             label = torch.tensor(label).float()
#             label = label.unsqueeze(-1).to(config.device)
#             loss = loss_fn(logit, label)
#             tot += loss
#             prob.extend(sigmoid(logit).to('cpu').numpy())
#             target.extend(label.to('cpu').numpy())
#         pred = [int(p >= threshold) for p in prob]
#         ans = evaluate_performance(np.array(target), np.array(pred), np.array(prob))
# 
#         print(f'auroc: {ans["auroc"]} auprc :{ans["auprc"]}  acc:{ans["accuracy"]} mcc:{ans["mcc"]} '
#               f'f1:{ans["f1"]}')
#         return ans
# 
# 
# def train_fine_tune():
#     torch.manual_seed(config.seed)
#     torch.cuda.manual_seed_all(config.seed)
#     shapelet_info = []
#     if config.use_shapelet == 1:
#         shapelet_info = get_shapelet(k_=4)[:200]
# 
#     train_dataset = combine_graph(data_path=f'{config.data_path}/train.csv', k=config.k,
#                                  data_save_path=config.data_save_path, shapelet_info=shapelet_info,
#                                  reload=config.reload, window_size=config.window_size)

# 
#     kf = KFold(n_splits=10, shuffle=True, random_state=config.seed)
#     test_dataloader = GraphDataLoader(test_dataset, batch_size=config.batch_size)
#     best_auc = 0
#     for i, (train_idx, dev_idx) in enumerate(kf.split(train_dataset)):
#         train_sampler = SubsetRandomSampler(train_idx)
#         dev_sampler = SubsetRandomSampler(dev_idx)
#         train_dataloader = GraphDataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
#         valid_dataloader = GraphDataLoader(train_dataset, batch_size=config.batch_size, sampler=dev_sampler)
#         model = LncLoc(k=config.k,shape_num=200, embed_dim=config.embed_dim, num_gcn=config.num_gcn,hidden_dim=config.hidden_dim)
#         model.to(config.device)
# 
#         opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay)
# 
#         for epoch in range(config.epoch):
#             print(f'fold:{i + 1} epoch{epoch + 1}')
#             logging.info(f'fold:{i + 1} epoch{epoch + 1}')
#             train_step(train_dataloader, model, opt, config)
#             print('train')
#             test_performance(train_dataloader, model, config=config)
#             print('dev')
#             ans = test_performance(valid_dataloader, model, config=config)
#             if ans["auroc"] > best_auc:
#                 best_auc = ans["auroc"]
#                 save_dir = config.res_dir + f'/{best_auc:.4f}'
#                 make_path(save_dir)
#                 torch.save(model.state_dict(), f'{save_dir}/best.pt')
# 
#                 model_pref = test_performance(valid_dataloader, model, config=config)
#                 eval_output(model_pref, path=save_dir)
#                 plot_AUROC(model_pref, path=save_dir)
# 
#             print(f'best_auc {best_auc}')
# 
# 
# if __name__ == '__main__':
#     logging.basicConfig(filename="train.log",
#                         filemode="a",
#                         format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
#                         datefmt="%d-%m-%Y %H:%M:%S",
#                         level=logging.INFO)
# 
# 
# 
#     parser = argparse.ArgumentParser(description="input the window_size")
#     parser.add_argument('--w', type=int, default=40)
#     parser.add_argument('--gpu', type=int, default=0)
# 
#     args = parser.parse_args()
#     config.device = f'cuda:{args.gpu}'
#     config.batch_size = 256
#     config.k = 3
#     config.embed_dim = 256
#     config.hidden_dim = 256
#     config.num_gcn = args.gpu+1
#     config.lr = 0.00001
#     config.decay = 0.0003676
# 
#     config.epoch = 200
#     config.seed = 0
#     config.window_size = args.w
#     config.data_path = f'data/rnalight'
#     config.data_save_path = f'ckpt/finetune/rnalight'
#     config.res_dir = 'result/'
#     config.use_shapelet = 1
#     config.reload = False
#     train_fine_tune()
# 
#     # ls = os.listdir('data/raw/CNRCI_train_data_source/transcripts_type')
#     # for d in ls:
#     #     make_path(f'ckpt/finetune/{d}')
#     #     # idng_dataset(f'data/{d}/benchmark_train.csv', 3,f'ckpt/finetune/{d}')
#     #     # idng_dataset(f'data/{d}/benchmark_dev.csv', 3, f'ckpt/finetune/{d}')
#     #     # idng_dataset(f'data/{d}/test_set.csv', 3, f'ckpt/finetune/{d}')
#     #     logging.info(f'{d} start logging....')
#     #     logging.info('=' * 100)
#     #     config.data_path = f'data/{d}'
#     #     config.data_save_path = f'ckpt/finetune/{d}'
#     #     train_fine_tune()
#     #     logging.info('=' * 100)
#     #     logging.info(f'{d} end logging....')
#     # sendMail('train.log')
