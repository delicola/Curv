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
from collections import Counter

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

def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

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

# datasetname = 'com-dblp'  # cora com-dblp karate dolphins football  email-Eu-core infect-dublin
# dataset = './data/real-world/{}/{}'.format(datasetname,datasetname)  #LFR_100_0  barabasi_albert_200_16  GN1000 caveman_250
# # dataset = './data/synthetic/caveman_250'  #LFR_100_0  barabasi_albert_200_16  GN1000 caveman_250
# path = os.path.join('{}.gml'.format(dataset))
# G = nx.read_gml(path, destringizer = int, label='id')
# G.graph['path'] = path
# # logger.info('{}网络的模块度为{}'.format(datasetname,tools.cal_modularity(G)))
# explore_curv(G)