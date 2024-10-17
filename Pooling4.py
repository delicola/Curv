#这里面的初始feature不是onehot编码，

from collections import defaultdict, namedtuple
from typing import Optional, Callable, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_scatter import scatter
from torch.nn import Parameter, Linear
from torch_geometric.nn import GCNConv, MessagePassing, GATConv
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from torch_geometric.utils import from_networkx, add_remaining_self_loops, softmax
from torch_sparse import coalesce


from tools import edgeIndex
import tools
from Pooling1 import RicciCurvaturePooling1
from config import parser
args = parser.parse_args()


class simiConv(MessagePassing):  # self.gnn_score = simiConv(self.in_channels, 1)
    def __init__(self, in_channels, out_channels):
        super(simiConv, self).__init__(aggr='add')  #'mean'

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Linear(in_channels, in_channels, bias=False)
        self.lin2 = Linear(in_channels, out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):  #这里传入的x就是前面得到的h向量
        """"""
        a = self.lin1(x)  #对应公式7的W1，是一个线性变化即乘上W1（dxd维）
        b = self.lin2(x)  #对应公式8得W2（dx1维）参数，他这里都是先乘上参数再做运算
        out = self.propagate(edge_index, x=a, x_cluster=b)  #x_cluster就是h向量乘上W2
        # return out + b
        return out  #.sigmoid()

    def message(self, x_i, x_j, x_cluster_i):
        out = torch.cosine_similarity(x_i, x_j).reshape(-1, 1)  #获得节点相似性方法可以换，x_i表i节点本身，x_j表i节点得邻居
        print(x_i.shape, out.shape)
        return x_cluster_i * out  #这里就是公式8，x_cluster_i就是i节点的x_cluster，out是eij
        # return  out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class RicciCurvaturePooling(nn.Module):
    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster"])

    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.01, alpha=0.5, GNN: Optional[Callable] = GCNConv,
                 dropout: float = 0.0, negative_slope: float = 0.2, add_self_loops: bool = False, **kwargs):
        super(RicciCurvaturePooling, self).__init__()
        self.alpha = alpha
        self.in_channels = in_channels
        self.ratio = ratio
        self.verbose = "ERROR"
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.add_self_loops = add_self_loops
        self.heads = 6
        self.weight = Parameter(torch.Tensor(in_channels, self.heads * in_channels))
        self.attention = Parameter(torch.Tensor(1, self.heads, 2 * in_channels))
        self.use_attention = True

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, in_channels)
        self.gnn_score = simiConv(self.in_channels, 1)


        self.marginloss = nn.MarginRankingLoss(0.5)
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.CosineLoss = nn.CosineEmbeddingLoss(margin=0.2)
        self.GCN = GCNConv(self.in_channels, self.in_channels,
                                         **kwargs)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,
                                         **kwargs)
        self.reset_parameters()

    def glorot(self, tensor):  # inits.py中
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        self.glorot(self.weight)
        self.glorot(self.attention)
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def gen_subs(self, edge_index, N):  #找出节点的邻接节点并将其Conv(self.in_channels, 1)作为子图，并返回相应的索引

        edgelists = defaultdict(list)  # 用于存储每个节点的邻接节点
        match = defaultdict(list)
        for i in range(edge_index.size()[1]):
            s = int(edge_index[0][i])
            t = int(edge_index[1][i])
            if s != t:
                edgelists[s].append(t)

        start = []
        end = []
        for i in range(N):
            start.append(i)
            end.append(i)
            match[i].append(i)  #这里可以去掉就不含自身---------
            if len(match[i]) == 1:
                match[i].extend(edgelists[i])
                start.extend(edgelists[i])
                end.extend([i] * len(edgelists[i]))
#edgelists不含自身，match是包含自身
        #start = []
        #end = []
        #for i in range(N):
        #start.append(i)
        #end.append(i)
        #if i in edgelists:
        #for j in edgelists[i]:
        #start.append(j)
        #end.append(i)

        source_nodes = torch.Tensor(start).reshape((1, -1))
        target_nodes = torch.Tensor(end).reshape((1, -1))
        subindex = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long)

        return subindex, edgelists, match

    def choose(self, x, x_pool, edge_index, batch, score, match, old_index):
        nodes_remaining = set(range(x.size(0)))
        cluster = torch.empty_like(batch, device=torch.device('cpu')) #对应位置的节点所属的社团编号
        node_argsort = torch.argsort(score, descending=True)
        i = 0 #社团编号
        transfer = {} #key 社团编号 value代表其对应的中心节点，这里面孤立节点自己作为一个社团，但他们不在remainnodes中，这样对吗？
        new_node_indices = [] #中心节点

        tar_in = [] #存储社团周围节点
        tar_tar = [] #存储社团编号
        for node_idx in node_argsort.tolist():  #sort_node 这一次迭代表示将motif合并并用一个新的索引表示这个合并的节点，同时移除原有的这些节点防止重叠
            source = match[node_idx]

            if node_idx not in nodes_remaining: #中心节点被分走，则跳过
                continue
            # source.pop(0)
            if len(source) == 1 and node_idx in source:  #如果节点是一个孤立节点，跳过
                continue


            source = [c for c in source if c in nodes_remaining] #当有一个邻居被分走就跳过？,找到还剩余的邻居，这里面的source是邻居节点

            transfer[i] = node_idx  #transfer表示融合后的

            new_node_indices.append(node_idx)
            for j in source:
                cluster[j] = i  #记录哪些节点j被融合到新社团序号i中
                if j != node_idx:
                    tar_in.append(j)
                    tar_tar.append(i)

            nodes_remaining = [j for j in nodes_remaining if j not in source]

            i += 1

        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            transfer[i] = node_idx
            i += 1

        #cluster = cluster.to(torch.device('cuda'))
        cluster = cluster.to(x.device)
        index = new_node_indices + nodes_remaining  #融合后还有哪些节点
        new_x_pool = x_pool[index, :] #x_pool是所有原始节点的特征向量，new_x_pool是按照排序得到的节点的特征向量
        new_x = torch.cat([x[new_node_indices, :], x_pool[nodes_remaining, :]]) #所有社团的特征
        new_score = score[new_node_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_node_indices),))
            new_score = torch.cat([new_score, remaining_score]) #所有社团新的得分，多个节点的社团是原来的得分，单个节点的社团是1
        # new_x = new_x * new_score.view(-1, 1)
        N = new_x.size(0)
        #todo
        #不需要社团之间的连边
        # new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)  #用聚类中的序号替换原来的节点索引序号，得到的是社团之间的连接关系
        # unpool_info = self.unpool_description(edge_index=edge_index,
        #                                       cluster=cluster)

        #生成正样本和负样本
        pos = []
        anchor_pos = []
        neg = []
        anchor_neg = []
        sig = {}
        old_match = tools.edgeindex2match(old_index.tolist())#key 节点编号，  value是社团编号，key邻居的社团编号，但是这个邻居不和key在一个社团
        for idx in range(x.size(0)):
            sig[idx] = []
            if cluster[idx].item() in range(len(transfer)):  # 生成正样本  这个节点所属的社团是大社团，往下走
                pos.append(idx)
                anchor_pos.append(cluster[idx].item())
                for j in old_match[idx]:  # 遍历所有其他聚类
                    if j != idx and cluster[j] != cluster[idx] and cluster[j].item() not in sig[idx]:
                        # 生成负样本：选择与当前节点不同簇的所有聚类
                        neg.append(idx)
                        anchor_neg.append(cluster[j].item())
                        sig[idx].append(cluster[j].item())

        pos_pos = x_pool[pos]
        pos_anchor = new_x[anchor_pos]  #位置对应的是节点编号，value对应的是他所归属社团的特征
        neg_neg = x_pool[neg]
        neg_anchor = new_x[anchor_neg]

        #基于edgeindex生成match，key为节点编号，value为邻居节点编号

#TODO
#new_edge_index这里面有问题，不应该是社团之间的索引，我们不会对他融合。

        # return new_x, new_x_pool, new_edge_index, unpool_info, cluster, transfer, pos_pos, pos_anchor, neg_neg, neg_anchor
        return new_x, new_x_pool, cluster, transfer, pos_pos, pos_anchor, neg_neg, neg_anchor

    def BCEloss(self, pos_anchor, pos, neg_anchor, neg):
        n1, h1 = pos_anchor.size()
        n2, h2 = neg_anchor.size()

        TotalLoss = 0.0
        pos = torch.bmm(pos_anchor.view(n1, 1, h1), pos.view(n1, h1, 1))
        loss1 = self.BCEWloss(pos, torch.ones_like(pos))
        if neg_anchor.size()[0] != 0:
            neg = torch.bmm(neg_anchor.view(n2, 1, h2), neg.view(n2, h2, 1))
            loss2 = self.BCEWloss(neg, torch.zeros_like(neg))
        else:
            loss2 = 0

        TotalLoss += loss2 + loss1
        return TotalLoss

    def forward(self, x, edge_index, old_index, batch=None):
        # 将 edge_index 转换为 NetworkX 图
        G = tools.convert_edge_index_to_graph(edge_index)
        N = x.size(0)
        if N == 1:
            unpool_info = self.unpool_description(edge_index=edge_index,
                                                  cluster=torch.tensor([0]))
            return x, edge_index, unpool_info, torch.tensor(0.0, requires_grad=True), 0.0

        edge_index, _ = add_remaining_self_loops(edge_index, fill_value=1, num_nodes=N)

        if batch is None:
            batch = torch.LongTensor(size=([N]))

        subindex, edgelists, match = self.gen_subs(edge_index, N)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x_pool = x
        if self.GNN is not None:
            # print("GNN")
            x_pool_j = self.gnn_intra_cluster(x=x, edge_index=edge_index)  # 这里用了自注意力gnn对x处理
        # print("x_pool", x_pool.size())

        if self.use_attention: #这里面将subindex全都改成edgeindex
            x_pool_j = torch.matmul(x_pool_j, self.weight)

            x_pool_j = x_pool_j.view(-1, self.heads, self.in_channels)

            x_i = x_pool_j[subindex[0]]  # 保存原来邻居的特征向量xj

            x_j = scatter(x_i, subindex[1], dim=0, reduce='max')  # 虚拟合并节点xmi向量

            alpha = (torch.cat([x_i, x_j[subindex[1]]], dim=-1) * self.attention).sum(dim=-1)  # 对应公式4向量拼接部分

            alpha = F.leaky_relu(alpha, self.negative_slope)  # self.negative_slope=0.2
            alpha = softmax(alpha, subindex[1], num_nodes=x_pool_j.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            v_j = x_pool_j[subindex[0]] * alpha.view(-1, self.heads, 1)

            x = scatter(v_j, subindex[1], dim=0, reduce='add')

            x = x.mean(dim=1)  # 就是h向量社团特征

        fitness = self.gnn_score(x, subindex).sigmoid().view(-1)
        #x_pool = self.GCN(x_pool, subindex)

        x, new_x_pool, cluster, transfer, pos_pos, pos_anchor, neg_neg, neg_anchor = self.choose(
            x, x_pool, edge_index, batch, fitness, match, old_index)

        loss = self.BCEloss(pos_anchor, pos_pos, neg_anchor, neg_neg)



        return x_pool, subindex, fitness, loss
        #return x, new_edge_index, unpool_info, cluster, fitness, loss








class GraphNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNet, self).__init__()
        self.fc = Linear(in_channels, 16)
        self.conv1 = GCNConv(16, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.conv4 = GATConv(out_channels, out_channels)
        #self.conv4 = GATConv(out_channels, out_channels, heads=6, dropout=0.6)
        self.pool1 = RicciCurvaturePooling(alpha=0.5, in_channels=out_channels)
        #self.pool2 = RicciCurvaturePooling(alpha=0.5, in_channels=out_channels*6)
        self.pool2 = RicciCurvaturePooling(alpha=0.5, in_channels=out_channels)
        #self.lin = Linear(out_channels, 1)
        #self.final_score = simiConv(out_channels*6, 1)
        self.final_score = simiConv(out_channels, 1)

    # def edgeIndex(self, G):
    #     source_nodes = []
    #     target_nodes = []
    #     for e in list(G.edges()):
    #         n1 = int(e[0])
    #         n2 = int(e[1])
    #         source_nodes.append(n1)
    #         source_nodes.append(n2)
    #         target_nodes.append(n2)
    #         target_nodes.append(n1)
    #     source_nodes = torch.Tensor(source_nodes).reshape((1, -1))
    #     target_nodes = torch.Tensor(target_nodes).reshape((1, -1))
    #     edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long)
    #
    #     return edge_index
    def forward(self, G, x):
        #data = from_networkx(G)

        #edge_index = data.edge_index
        #print(edge_index, '\n', edge_index.shape)
        edge_index = edgeIndex(G)
        #TODU
        # print(edge_index.shape)
        # 第一层卷积
        x = F.relu(self.fc(x))
        x = F.relu(self.conv1(x, edge_index))

        # 第二层卷积
        # x = F.relu(self.conv2(x, edge_index))
        # 第三层卷积
        x = F.relu(self.conv3(x, edge_index))
        # 应用池化层，删除曲率为负的边，更新后的 edge_index 和 x 会传递给下一层
        x_pool1, new_edge_index1, fitness1, loss1 = self.pool1(x, edge_index, edge_index)
        drop_edge_index = tools.drop_edge(edge_index, methods='neg')
        x = F.relu(self.conv4(x_pool1, edge_index))

        x_pool2, new_edge_index2, fitness2, loss2 = self.pool2(x, drop_edge_index, edge_index)

        x1 = self.final_score(x_pool2, edge_index)
        #x_pool2, new_edge_index2, fitness2, loss2 = self.pool2(x_pool1, new_edge_index1, edge_index)
        #x_pool3, new_edge_index3, fitness3, loss3 = self.pool3(x_pool2, new_edge_index2, edge_index)
        #x, new_edge_index1, unpool_info, cluster, fitness, loss = self.pool(x, edge_index)

        #x1 = F.relu(self.lin(x_pool3))
        # x1 = F.relu(self.conv4(x_pool, new_edge_index1))
        x1 = torch.sigmoid(x1)
        rank_dict = {}
        i = 0
        for n in G.nodes():
            rank_dict[n] = float(x1[i])
            i += 1

        #return x1, new_edge_index1, unpool_info, cluster, fitness, loss
        return x1, new_edge_index2, loss1+loss2, rank_dict




# # 示例用法
# #G = nx.karate_club_graph()
# # G = nx.barabasi_albert_graph(100, 3)
# path = './data/BA_{}.gml'.format(100_3)
# # G.remove_nodes_from(list(nx.isolates(G)))
# # G.remove_edges_from(nx.selfloop_edges(G))
# # G.graph['path'] = path
# # tools.save_graph_gml(G)
#
# G = nx.read_gml(path, destringizer = int, label='id')
# #
# num_nodes = G.number_of_nodes()
# # 生成独热编码的特征矩阵
# feature = torch.eye(num_nodes, dtype=torch.float)
#
# # 创建模型并运行前向传播
# model = GraphNet(in_channels=num_nodes, hidden_channels=64, out_channels=32)
# #out, updated_edge_index, unpool_info1, cluster1, fitness1, loss1 = model(G, feature)
# out, updated_edge_index, loss3, rank_dict = model(G, feature)
# print(out.shape)  # 输出的特征矩阵形状
# print(out)  # 输出的特征矩阵
# print(updated_edge_index)  # 输出更新后的边索引
# #print(updated_edge_index.shape)
# #print(unpool_info1)  # 输出池化信息
# #print(cluster1) # 输出 cluster 信息
# #print(fitness1)  # 输出 fitness 信息
# print(loss3)  # 输出 loss 信息
# print(rank_dict)  # 输出 rank_dict 信息
