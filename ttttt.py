# 这个文件用于探索自己的图的曲率是多少

import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from matplotlib import pyplot as plt

# 1. 创建一个图，例如 karate club graph

edge_list = [(1, 2), (1, 3), (2, 1), (2, 3), (2, 5),
             (3, 1), (3, 2), (3, 4), (4, 3), (4, 5), (4, 8), (4, 9),
             (5, 2), (5, 4), (5, 6), (6, 5), (6, 7), (7, 6), (7, 8),
             (8, 4), (8, 7), (9, 4), (9, 10), (9, 12), (9, 13),
             (10, 9), (10, 11), (11, 10), (12, 9), (13, 9),
             (13, 14), (14, 13), (14, 15), (14, 16), (15, 14), (16, 14)]
G = nx.Graph()
G.add_edges_from(edge_list)


# 2. 初始化 Ollivier-Ricci 对象 (假设 alpha = 0.5, method = "OTD")
orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD")
orc_OTD.compute_ricci_curvature()
edge_rc_list = list(orc_OTD.G.edges.data("ricciCurvature"))
print(edge_rc_list)
# 3. 调用 compute_ricci_flow 函数，进行 Ricci 流计算 (比如迭代10次)
# orc_OTD.compute_ricci_flow(iterations=10)
# w = list(orc_OTD.G.edges.data("weight"))
# edge_rc_list = list(orc_OTD.G.edges.data("ricciCurvature"))
# original_edge_rc_list = list(orc_OTD.G.edges.data("original_RC"))
# print(w)
# print(original_edge_rc_list)
# print(edge_rc_list)


