import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from matplotlib import pyplot as plt

# 1. 创建一个图，例如 karate club graph
G = nx.barabasi_albert_graph(30,3)


# 2. 初始化 Ollivier-Ricci 对象 (假设 alpha = 0.5, method = "OTD")
orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD")

# 3. 调用 compute_ricci_flow 函数，进行 Ricci 流计算 (比如迭代10次)
orc_OTD.compute_ricci_flow(iterations=10)
w = list(orc_OTD.G.edges.data("weight"))
edge_rc_list = list(orc_OTD.G.edges.data("ricciCurvature"))
original_edge_rc_list = list(orc_OTD.G.edges.data("original_RC"))
print(w)
print(original_edge_rc_list)
print(edge_rc_list)


