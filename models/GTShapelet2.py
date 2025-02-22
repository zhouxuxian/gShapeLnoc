from dgl.nn.pytorch import GINConv
import torch.nn as nn
import torch
from models.utils import get_mask

#与GTShapelet类似，传播时先按照第一类边传播，再按照第二类边传播。
#聚合特征后不进行读出，按照第二类边传播，再按照第一类边传播
class GTShapelet2(nn.Module):
    def __init__(self, k, embed_dim=128, num_gcn=6, num_heads=4):
        super(GTShapelet2, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = 1 << 2 * k
        self.embed = nn.Embedding(self.num_nodes, self.embed_dim)
        self.convs = nn.ModuleList()
        self.norm_gcn = nn.LayerNorm(embed_dim)
        self.convs.append(GINConv(nn.Linear(embed_dim, embed_dim//2)))
        self.convs.append(GINConv(nn.Linear(embed_dim//2, embed_dim//2)))
        self.convs.append(GINConv(nn.Linear(embed_dim//2, embed_dim)))
        self.act = nn.GELU()
        self.MHA = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)  # MultiHeadAttention
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, embed_dim], requires_grad=True))
        self.norm_after = nn.LayerNorm(embed_dim)
        self.set_parameter()

    def set_parameter(self):
        for name, param in self.named_parameters():
            if 'norm' in name:
                continue
            if 'bias' in name:
                nn.init.zeros_(param)
                continue
            nn.init.kaiming_uniform_(param)

    def forward(self, g, g2):
        h = self.embed(g.ndata['mask'])
        h2 = self.embed(g2.ndata['mask'])
        for idx, gnn in enumerate(self.convs):
            if idx % 2 == 0:
                h = gnn(g, h, edge_weight=g.edata['weight'].float())
                h = self.act(h)
            else:
                h = gnn(g2, h, edge_weight=g2.edata['weight'].float())
                h = self.act(h)
        for idx, gnn in enumerate(self.convs):
            if idx % 2 == 1:
                h2 = gnn(g, h2, edge_weight=g.edata['weight'].float())
                h2 = self.act(h2)
            else:
                h2 = gnn(g2, h2, edge_weight=g2.edata['weight'].float())
                h2 = self.act(h2)
        with g.local_scope():
            g.ndata['h'] = h+h2
            src_padding_mask = get_mask(g)
            h = h+h2
            h = h.view(-1, self.num_nodes, self.embed_dim)  # batch_size  * num_node * embed
        expand_cls_embedding = self.cls_embedding.expand(h.size(0), 1, -1)  # batchsize * 1 * embed
        h = torch.cat([h, expand_cls_embedding], dim=1)  # batch * length * dim
        zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
        src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        attn_output, _ = self.MHA(h, h, h, key_padding_mask=src_padding_mask)
        h = self.norm_after(h + attn_output)
        return h[:, -1, :]
