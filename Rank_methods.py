import os

import networkx as nx

import tools
from Pooling7 import GraphNet
import torch
import time
import copy
from tqdm import tqdm
from CI.python_inferrence import collective_influence_l

import torch.optim as optim

from config import parser
args = parser.parse_args()


def rank_by_curv(G, ep = args.epoch, reverse = True):

    num_nodes = G.number_of_nodes()
    drop_edge_indexs = tools.cal_curve(G, args.drop_percent, args.drop_times)
    #feature = tools.generate_feature_matrix(G)
    feature = torch.eye(num_nodes, dtype=torch.float)
    model = GraphNet(in_channels=num_nodes, hidden_channels=64, out_channels=32)
    # model = GraphNet(in_channels=4, hidden_channels=64, out_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    for epoch in tqdm(range(ep)):
        if epoch ==19:
            print(55)
        t = time.time()
        model.train()
        G0 = copy.deepcopy(G)
        # 测试模型
        out, updated_edge_index, loss3, rank_dict = model(G, feature, drop_edge_indexs)
        optimizer.zero_grad()
        loss3.backward()
        optimizer.step()
        # print(loss3)
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

def rank_by_CI(G,l=1):
    nodes = collective_influence_l(G,l)
    print("CI:",nodes)
    # nodes = [5, 10, 7, 6, 11, 15, 16, 3, 2, 12, 0, 1, 8, 4, 9, 14, 13]
    return nodes

# def rank_by_MY(G):
#     666