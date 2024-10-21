import os

import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import numpy as np
import random
import math
import argparse
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import torch
# from config import parser
from collections import Counter

from tools import edgeIndex

def cal_curve(G, drop_ratio, drop_times):
    edge_index = edgeIndex(G)#原始图索引
    # 计算原始图曲率
    orc = OllivierRicci(G, alpha=0.5, verbose='INFO')
    orc.compute_ricci_curvature()
    edge_rc_list = list(orc.G.edges.data("ricciCurvature"))#存储边曲率
    edge_rc_all = [rc[2] for rc in edge_rc_list]#取出所有边的曲率
    x = 0
    y = 0
    edge_neg = []
    edge_pos = []
    edge_zero = []
    for i in edge_rc_all:
        if i < 0:
            x += 1
            edge_neg.append(i)
        elif i > 0:
            y += 1
            edge_pos.append(i)
        else:
            edge_zero.append(i)

    drop_edge_index = [edge_index]
    original_edge_rc_list = edge_rc_list.copy()

    # 初始化一个空的列表存放每次删完后的边索引
    for i in range(drop_times):
        if not edge_neg:  # 检查是否还有负曲率边可以删除
            break

        edge_rc_list_sorted = sorted(original_edge_rc_list, key=lambda x: x[2])#根据边曲率排序
        num_edges_to_remove = int(len(edge_neg) * drop_ratio)#移除负曲率边中多少条边
        edges_to_remove = edge_rc_list_sorted[:num_edges_to_remove]#取出这些边
        print("edges_to_remove:", edges_to_remove)
        for edge in edges_to_remove:
            G.remove_edge(edge[0], edge[1])#移除这些边
            #更新original_edge_rc_list
            original_edge_rc_list.remove(edge)
        edge_index = edgeIndex(G)#获得新的边索引
        drop_edge_index.append(edge_index)#存储新的边索引
        #把drop_edge_index转化为list
        edge_neg = [rc for rc in original_edge_rc_list if rc[2] < 0]#更新负曲率边

    drop_edge_index = list(drop_edge_index)

    return drop_edge_index

G = nx.karate_club_graph()

drop_edge_index = cal_curve(G, 0.6, 4)

print(drop_edge_index)




