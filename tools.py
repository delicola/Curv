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

from config import parser
args = parser.parse_args()



def sortbydict(dict, reverse = True):
    remove = sorted(dict.items(), key=lambda x: x[1], reverse=reverse)
    removelist = []
    for i in range(len(remove)):
        removelist.append(remove[i][0])
    return removelist


def network_dismantling(G, removelist, threshold, method = 'unknown'): #theshold 是待移除节点的比例
    componentslist = []
    components_radio = []
    # 试一试随机生成移除列表
    # # random.shuffle(G.graph['removelist'])
    num_nodes = len(G.nodes())
    # removelist = G.graph['removelist']
    g = G.copy()

    x = []
    x.append(0)
    components_radio.append(1)
    num_remove = int(num_nodes * threshold)
    logger.info('本次拆解移除{}个节点,方法是{}'.format(num_remove, method))
    for t in range(num_remove-3):
        g.remove_nodes_from(removelist[0:t + 1])
        componentslist.append(len(max(nx.connected_components(g), key=len)))
        components_radio.append(len(max(nx.connected_components(g), key=len)) / num_nodes)
        x.append((t+1)/num_nodes)

    return x, components_radio

def count_SIR(G):
    s_num, i_num, r_num = 0, 0, 0
    for node in G:
        if G.nodes[node]['status'] == 'S':
            s_num += 1
        elif G.nodes[node]['status'] == 'I':
            i_num += 1
        else:
            r_num += 1
    return s_num,i_num, r_num

def update_node_status(G, node, beta, gamma):
    if G.nodes[node]['status'] == 'I':
        p = random.random()
        if p < gamma:
            G.nodes[node]['status'] = 'R'
    elif G.nodes[node]['status'] == 'S':
        p = random.random()
        count = 0
        for neighbour in G.adj[node]:
            if G.nodes[neighbour]['status'] == 'I':
                count = count + 1
        if p < 1 - (1 - beta) ** count:
            G.nodes[node]['status'] = "I"


def simulate_SIR(G, infected_nodes_list, beta, gamma=0, step=50, infect_ratio = 0.1, times = 50):
    # store every iteration I + R
    sir_values = {}
    num_S_list_all = np.zeros(step)
    num_I_list_all = np.zeros(step)
    num_R_list_all = np.zeros(step)
    # self_influence_dict = self_influence(G)
    # gravity_enhance = gravity_plus(G)
    for i in tqdm(range(times)):
        # for node_toInfect in nodes_list:
            # 初始化每个节点的状态为S
            num_S = 0
            num_I = 0
            num_R = 0
            for node in G:
                G.nodes[node]['status'] = 'S'
            # 按顺序设定一个节点为起始已感染节点
            for node_toInfect in infected_nodes_list:
                G.nodes[node_toInfect]['status'] = 'I'

            for s in range(step):
                num_S, num_I, num_R = count_SIR(G)
                num_S_list_all[s] += num_S
                num_I_list_all[s] += num_I
                num_R_list_all[s] += num_R

                for node_temp in G:
                    update_node_status(G, node_temp, beta, gamma)
                    # print(" {}\n".format(s))

    num_S_list = num_S_list_all / times
    num_I_list = num_I_list_all / times
    num_R_list = num_R_list_all / times

    return num_S_list, num_I_list, num_R_list

def calculate(G):
    degree = dict(nx.degree(G))
    avg_degree = sum(degree.values()) / len(G.nodes)
    degrees = 0
    for i in degree.keys():
        degrees += math.pow(degree.get(i), 2)
    avg_degree_sqrt = degrees / len(G.nodes)
    beta = avg_degree / avg_degree_sqrt
    return 1.5 * beta


#TODO 画图
def plot_SIR(G, method_dict):
    beta = calculate(G)
    for method in method_dict:
        num_S, num_I, num_R = simulate_SIR(G, infected_nodes_list=method.keys(), beta=beta)
    plt.legend()
    plt.show()


def cal_area_R(l):
    area = 0
    for i in range(len(l)-1):
        area += l[i]
    area = area / len(l)
    return area

def return_index(list, threshold = 0.01): #返回列表小于阈值所对应的index
    a = 0
    for n in list:
        if n <= threshold:
            break
        else:
            a += 1
    return a



def cal_modularity(G):
    m = nx.community.modularity(G, nx.community.label_propagation_communities(G))

    return m


def explore_curv(G):
    # G = nx.read_gml(path, destringizer=int, label='id')
    datasetname = os.path.split(G.graph['path'])[1][0:-4]
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    edge_rc_list = list(orc.G.edges.data("ricciCurvature"))
    edge_rc_all = [rc[2] for rc in edge_rc_list]
    # 查看edge_rc_all有多少个负数
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
    print("现在处理的网络是{}，有{}个节点，有{}条连边".format(datasetname, len(G.nodes), len(G.edges)))
    # 计算平均数
    print("负曲率的平均值为{}".format(sum(edge_neg) / len(edge_neg)))
    # 负曲率里面的最大值和最小值
    print("负曲率里面的最大值为{}".format(max(edge_neg)))
    print("负曲率里面的最小值为{}".format(min(edge_neg)))
    print("正曲率的平均值为{}".format(sum(edge_pos) / len(edge_pos)))
    print("正曲率里面的最大值为{}".format(max(edge_pos)))
    print("正曲率里面的最小值为{}".format(min(edge_pos)))
    print("负曲率有{}个,正曲率有{}个,零曲率的个数为{}".format(x, y, len(edge_zero)))

def cal_MI(ranking_list):
    from collections import Counter

    N_L = len(ranking_list)
    if N_L < 2:
        return 0  # Undefined or trivial case with less than 2 nodes

    # Count the frequency of each ranking value
    ranking_counts = Counter(ranking_list)

    # Calculate the numerator
    numerator = sum(N_alpha * (N_alpha - 1) for N_alpha in ranking_counts.values())

    # Calculate the denominator
    denominator = N_L * (N_L - 1)

    # Calculate the Monotonicity Index
    MI = (1 - numerator / denominator) ** 2
    return MI

# Example usage:
ranking_list = [1, 2, 3, 4, 5]
mi_value = cal_MI(ranking_list)
print("Monotonicity Index (MI):", mi_value)


def edgeIndex(G): #这里是双向连边
    source_nodes = []
    target_nodes = []
    for e in list(G.edges()):

        n1 = int(e[0])
        n2 = int(e[1])
        if e[0] == e[1]:
            source_nodes.append(n1)
            target_nodes.append(n2)
            continue
        source_nodes.append(n1)
        source_nodes.append(n2)
        target_nodes.append(n2)
        target_nodes.append(n1)
    source_nodes = torch.Tensor(source_nodes).reshape((1, -1))
    target_nodes = torch.Tensor(target_nodes).reshape((1, -1))
    edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long)

    return edge_index

def save_graph_gml(G):
    GraphFile = G.graph['path']
    File = os.path.join(GraphFile)
    nx.write_gml(G, File, stringizer= str)



def edgeindex2match(edge_index):
    match = {} #key为节点编号，value为邻居节点编号
    for i in range(len(edge_index[0])):
        if edge_index[0][i] in match.keys():
            match[edge_index[0][i]].append(edge_index[1][i])
        else:
            match[edge_index[0][i]] = [edge_index[1][i]]
        # if edge_index[1][i] in match.keys():
        #     match[edge_index[1][i]].append(edge_index[0][i])
        # else:
        #     match[edge_index[1][i]] = [edge_index[0][i]]
    return match

def convert_edge_index_to_graph(edge_index):
    #将 edge_index 转换为 NetworkX 图
    G = nx.Graph()
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    return G

def drop_edge(edge_index, methods = 'neg', alpha = 0.5, verbose = 'INFO'):
    G = convert_edge_index_to_graph(edge_index)
    orc = OllivierRicci(G, alpha=alpha, verbose=verbose)
    orc.compute_ricci_curvature()
    #orc.compute_ricci_flow(iterations=10)
    edge_rc_list = list(orc.G.edges.data("ricciCurvature"))
    #edge_rc_list = list(orc.G.edges.data("weight"))
    edge_rc_all = [rc[2] for rc in edge_rc_list]
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
    if methods == 'neg': #删除固定负曲率的边
        edge_rc_list_sorted = sorted(edge_rc_list, key=lambda x: x[2])
        # num_edges_to_remove = int(len(edge_rc_list_sorted) * args.drop_percent)
        num_edges_to_remove = int(len(edge_neg) * args.drop_percent)
        # num_edges_to_remove = int(len(edge_rc_list_sorted) * 0.40)
        edges_to_remove = edge_rc_list_sorted[:num_edges_to_remove]
        for edge in edges_to_remove:
            G.remove_edge(edge[0], edge[1])
    elif methods == 'pos':
        for i in range(len(edge_rc_list)):
            if edge_rc_list[i][2] > 0:
                G.remove_edge(edge_rc_list[i][0], edge_rc_list[i][1])
    elif methods == 'drop_percent':
        edge_rc_list_sorted = sorted(edge_rc_list, key=lambda x: x[2])
        num_edges_to_remove = int(len(edge_rc_list_sorted) * 0.10)
        edges_to_remove = edge_rc_list_sorted[:num_edges_to_remove]
        for edge in edges_to_remove:
            G.remove_edge(edge[0], edge[1])

    # G2 = tools.add_self_loop(G)
    edge_index = edgeIndex(G)
    print('负曲率边数{}'.format(len(edge_neg)))
    print('正曲率边数{}'.format(len(edge_pos)))
    print('零曲率边数{}'.format(len(edge_zero)))
    print('删除的边数{}'.format(num_edges_to_remove))

    return edge_index

# def feature_nodes(G):  # generate node features
#     NODES_LIST = list(G.nodes)
#
#     # the number of its own degree centrality 返回一个数， 度中心性
#     degree_dict = nx.degree_centrality(G)
#     degree_list = np.array([degree_dict[i] for i in NODES_LIST])[:, None]
#     degree_list_norm = degree_list / np.max(degree_list)
#     degree_list_norm = degree_list_norm.reshape(degree_list_norm.shape[0])
#     save_node_feature('degree_centrality_norm', degree_list_norm.tolist())
#
#
#     # the number of its two-hop neighbors，返回一个数，二阶邻居的个数 局部特征
#     second_neighbor = counts_high_order_nodes(self.G, depth=2)
#     second_neighbor_list = np.array([second_neighbor[i] for i in NODES_LIST])[:, None]
#     second_neighbor_list_norm = second_neighbor_list / np.max(second_neighbor_list)
#     # second_neighbor_list_norm = second_neighbor_list_norm.reshape(second_neighbor_list_norm.shape[0])
#     self.save_node_feature('second_neighbor_list_norm', second_neighbor_list_norm.tolist())
#
#     # average degree of its one-hop neighbors返回一个数，局部特征
#     neighbor_average_degree = nx.average_neighbor_degree(self.G)
#     neighbor_average_degree_list = np.array([neighbor_average_degree[i] for i in NODES_LIST])[:, None]
#     neighbor_average_degree_list_norm = neighbor_average_degree_list / np.max(neighbor_average_degree_list)
#     # neighbor_average_degree_list_norm = neighbor_average_degree_list_norm.reshape(neighbor_average_degree_list_norm.shape[0])
#     self.save_node_feature('neighbor_average_degree_list_norm', neighbor_average_degree_list_norm.tolist())
#
#     # the number of its local clustering coefficient
#     local_clustering_dict = nx.clustering(self.G)
#     local_clustering_list = np.array([local_clustering_dict[i] for i in NODES_LIST])[:, None]
#     # local_clustering_list = local_clustering_list.reshape(local_clustering_list.shape[0])
#     self.save_node_feature('local_clustering_list', local_clustering_list.tolist())
#
#
# def save_node_feature(self, feature_name, feature_list):
#     for i in self.G:
#         self.G.nodes[i][feature_name] = feature_list[i]

def counts_high_order_nodes(G, depth=2):
    NODES_LIST = list(G.nodes)
    output = {}
    output = output.fromkeys(NODES_LIST)
    for node in NODES_LIST:
        layers = dict(nx.bfs_successors(G, source=node, depth_limit=depth))
        high_order_nodes = sum([len(i) for i in layers.values()])
        output[node] = high_order_nodes
    return output

def generate_feature_matrix(G):
    NODES_LIST = list(G.nodes)

    # Calculate degree centrality and normalize
    degree_dict = nx.degree_centrality(G)
    degree_list = np.array([degree_dict[i] for i in NODES_LIST])
    degree_list_norm = degree_list / np.max(degree_list)

    # Calculate the number of two-hop neighbors and normalize
    second_neighbor = counts_high_order_nodes(G, depth=2)
    second_neighbor_list = np.array([second_neighbor[i] for i in NODES_LIST])
    second_neighbor_list_norm = second_neighbor_list / np.max(second_neighbor_list)

    # Calculate average degree of one-hop neighbors and normalize
    neighbor_average_degree = nx.average_neighbor_degree(G)
    neighbor_average_degree_list = np.array([neighbor_average_degree[i] for i in NODES_LIST])
    neighbor_average_degree_list_norm = neighbor_average_degree_list / np.max(neighbor_average_degree_list)

    # Calculate local clustering coefficient
    local_clustering_dict = nx.clustering(G)
    local_clustering_list = np.array([local_clustering_dict[i] for i in NODES_LIST])

    #Calculate PR value
    pr = nx.pagerank(G)
    pr_list = np.array([pr[i] for i in NODES_LIST])
    # pr_norm = pr_list / np.max(pr_list)


    # Combine the four features into a feature matrix
    feature_matrix = np.stack((degree_list_norm, second_neighbor_list_norm, pr_list, neighbor_average_degree_list_norm), axis=-1)

    return torch.from_numpy(feature_matrix).float()

def cal_curve(G, drop_ratio, drop_times):
    #原始图索引
    edge_index = edgeIndex(G)
    #计算原始图曲率
    orc = OllivierRicci(G, alpha=0.5, verbose='INFO')
    orc.compute_ricci_curvature()
    edge_rc_list = list(orc.G.edges.data("ricciCurvature"))
    edge_rc_all = [rc[2] for rc in edge_rc_list]
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
    drop_edge_index = []
    #初始化一个空的列表存放每次删完后的边索引
    for i in range(drop_times):
        edge_rc_list_sorted = sorted(edge_rc_list, key=lambda x: x[2])
        # num_edges_to_remove = int(len(edge_rc_list_sorted) * args.drop_percent)
        num_edges_to_remove = int(len(edge_neg) * drop_ratio)
        # num_edges_to_remove = int(len(edge_rc_list_sorted) * 0.40)
        edges_to_remove = edge_rc_list_sorted[:num_edges_to_remove]
        for edge in edges_to_remove:
            G.remove_edge(edge[0], edge[1])
    edge_index = edgeIndex(G)
    drop_edge_index.append(edge_index)
    # if methods == 'neg': #删除固定负曲率的边


    return edge_index

def cal_curve(G, drop_percent, drop_times):
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
    #original_edge_rc_list = edge_rc_list.copy()

    # 初始化一个空的列表存放每次删完后的边索引
    for i in range(drop_times):
        if not edge_neg:  # 检查是否还有负曲率边可以删除
            break

        edge_rc_list_sorted = sorted(edge_rc_list, key=lambda x: x[2])#根据边曲率排序
        num_edges_to_remove = int(len(edge_neg) * drop_percent)#移除负曲率边中多少比例的边
        edges_to_remove = edge_rc_list_sorted[:num_edges_to_remove]#取出这些边
        #print("edges_to_remove:", edges_to_remove)
        for edge in edges_to_remove:
            G.remove_edge(edge[0], edge[1])#移除这些边
            #更新original_edge_rc_list
            edge_rc_list.remove(edge)
        edge_index = edgeIndex(G)#获得新的边索引
        drop_edge_index.append(edge_index)#存储新的边索引
        #把drop_edge_index转化为list
        edge_neg = [rc for rc in edge_rc_list if rc[2] < 0]#更新负曲率边

    drop_edge_index = list(drop_edge_index)

    return drop_edge_index



# def trans_edgeindex(edge_index):#将edge_index的第二维全都改成end

# def add_self_loop(G):
#     for node in G.nodes():
#         G.add_edge(node, node)
#     return G

# num_nodes = 500
# tau1 = 3
# tau2 = 1.5
# mu = 0.1
# # G = nx.LFR_benchmark_graph(num_nodes, tau1, tau2, mu, average_degree=5, min_community=50, max_community=80)
# # G = nx.barabasi_albert_graph(num_nodes, 5)
#
# G = nx.connected_caveman_graph(5,50)
# print(cal_modularity(G))
# path = './data/BA_{}.gml'.format(100_3)
# G = nx.read_gml(path, destringizer = int, label='id')
# x = generate_feature_matrix(G)
# print(x)
# feature = torch.eye(100, dtype=torch.float)
# print(feature)
# datasetname = 'com-dblp'  # cora com-dblp karate dolphins football  email-Eu-core infect-dublin
# dataset = './data/real-world/{}/{}'.format(datasetname,datasetname)  #LFR_100_0  barabasi_albert_200_16  GN1000 caveman_250
# # dataset = './data/synthetic/caveman_250'  #LFR_100_0  barabasi_albert_200_16  GN1000 caveman_250
# path = os.path.join('{}.gml'.format(dataset))
# G = nx.read_gml(path, destringizer = int, label='id')
# G.graph['path'] = path
# # logger.info('{}网络的模块度为{}'.format(datasetname,tools.cal_modularity(G)))
# explore_curv(G)