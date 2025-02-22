import torch
import torch.nn as nn
from models.GTShapelet2 import GTShapelet2

#模型主函数，包含图模型和最后的形状子拼接后进行分类的逻辑
class LncLoc(nn.Module):
    def __init__(self, k, shape_num, embed_dim=128, num_gcn=2,num_heads=4, hidden_dim=1024):
        super(LncLoc, self).__init__()
        self.graph_encoder = GTShapelet2(k, embed_dim, num_gcn=num_gcn,num_heads=num_heads)
        self.classify = nn.Sequential(
            nn.Linear(embed_dim + shape_num, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, g, g2,sw, sf):
        #sw = torch.nn.functional.normalize(sw, p=2, dim=1)
        graph_repr = self.graph_encoder(g, g2)
        graph_repr = torch.concat((sf, graph_repr), dim=1)
        ###############
        # sf = torch.nn.functional.normalize(sf, p=2, dim=1)
        # graph_repr = torch.concat((sf, graph_repr), dim=1)
        # graph_repr = torch.nn.functional.normalize(graph_repr, p=2, dim=1)
        #####################

        return self.classify(graph_repr)

    def load_pretrain_encoder(self, source):
        source_state = source
        for name, param in self.named_parameters():
            if name in source_state:
                param.data = source_state[name]

    def set_parameter(self):
        for name, param in self.named_parameters():
            if 'graph_encoder' in name:
                continue
            if 'norm' in name:
                continue
            if 'bias' in name:
                nn.init.zeros_(param)
                continue
            nn.init.kaiming_uniform_(param)
