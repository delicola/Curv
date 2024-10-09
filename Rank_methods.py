import os

import networkx as nx

import tools
from Pooling import GraphNet
import torch
import time
import copy
from tqdm import tqdm

import torch.optim as optim


def rank_by_curv(G, ep = 50, reverse = True):

    num_nodes = G.number_of_nodes()
    feature = tools.generate_feature_matrix(G)
    # feature = torch.eye(num_nodes, dtype=torch.float)
    model = GraphNet(in_channels=4, hidden_channels=64, out_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    for epoch in tqdm(range(ep)):
        t = time.time()
        model.train()
        G0 = copy.deepcopy(G)
        # 测试模型
        out, updated_edge_index, loss3, rank_dict = model(G, feature)
        optimizer.zero_grad()
        loss3.backward()
        optimizer.step()
        print(loss3)
    # out, updated_edge_index, loss3, rank_dict = model(G, feature)
    print(rank_dict)
    rank =tools.sortbydict(rank_dict, reverse=reverse)
    rank2 = tools.sortbydict(rank_dict, reverse=False)
    print(rank)
    n = list(rank_dict.values())
    m = [float('%.4f' % i) for i in n]
    # print("网络：{}, MI of curv: {}".format(name, tools.cal_MI(m)))
    return rank, rank2

def rank_by_degree(G):
    # graph_path = os.path.join('./data/{}/{}-{}.gml'.format(dataset_name, dataset_name, high_dim))
    # G = graphgml(dataset_name, high_dim)
    name = os.path.split(G.graph['path'])[1][0:-4]
    print(G)
    rank_list_degree = tools.sortbydict(dict(G.degree()))
    n = list(dict(G.degree()).values())
    m = [float('%.4f' % i) for i in n]
    print(rank_list_degree)
    # print("网络：{}, MI of degree: {}".format(name, tools.cal_MI(m)))

    return rank_list_degree

def rank_by_PR(G):
    pr = nx.pagerank(G)
    rank_list_pr = tools.sortbydict(pr)
    name = os.path.split(G.graph['path'])[1][0:-4]
    n = list(pr.values())
    m = [float('%.4f' % i) for i in n]
    print("网络：{}, MI of PR: {}".format(name, tools.cal_MI(m)))
    return rank_list_pr