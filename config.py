#部分参数设置
class config:
    def __init__(self):
        super(config, self).__init__()
        self.device = f'cuda:0'
        self.batch_size = 256
        self.k = 3
        self.embed_dim = 256
        self.hidden_dim = 256
        self.lr = 0.00001
        self.decay = 0.0003676
        self.epoch = 200
        self.seed = 0
        self.window_size = 40
        self.data_path = f'data/rnalight'
        self.data_save_path = f'ckpt/finetune/rnalight'
        self.res_dir = 'result/'
        self.use_shapelet = 1
        self.reload = True
