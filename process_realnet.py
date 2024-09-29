import os

import networkx as nx
from scipy.io import mmread

import tools
from tools import *


def process_real_net(name, type):# 将mtx和txt转为gml
    if 'mtx' == type:
        path = os.path.join('./data/real-world/{}/{}.mtx'.format(name, name))
        graph = mmread(path)
        G = nx.Graph(graph)
    elif 'txt' == type:
        path = os.path.join('./data/real-world/{}/{}.txt'.format(name, name))
        fh = open(path, "rb")
        G = nx.read_edgelist(fh, comments='%', nodetype=int)
    G.graph['path'] = os.path.join('./data/real-world/{}/{}.gml'.format(name, name))
    print("name:{},nodes:{},edges:{}".format(name, G.number_of_nodes(), G.number_of_edges()))
    G = G.to_undirected()
    G.remove_nodes_from(list(nx.isolates(G)))
    G.remove_edges_from(nx.selfloop_edges(G))
    nx.convert_node_labels_to_integers(G,first_label=0)
    save_graph_gml(G)


def print_network_stats(G):
    # Number of nodes
    num_nodes = G.number_of_nodes()
    print(f"Number of nodes: {num_nodes}")

    # Number of edges
    num_edges = G.number_of_edges()
    print(f"Number of edges: {num_edges}")

    # Average degree
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    print(f"Average degree: {avg_degree}")

    # Clustering coefficient
    cluster_coeff = nx.average_clustering(G)
    print(f"Clustering coefficient: {cluster_coeff}")

    # Density
    density = nx.density(G) * 1000
    print(f"Density (x10^-3): {density}")

    #Diameter
    #输出网络最大连通子图
    G2 = max(nx.connected_components(G), key=len)
    G3 = G.subgraph(G2)
    diameter = nx.diameter(G3)
    print(f"Diameter: {diameter}")

    # Modularity
    # Assuming G is undirected and communities are detected using the Louvain method
    # communities = nx.community.best_partition(G)
    # modularity = nx.community.modularity(communities, G)
    m = nx.community.modularity(G, nx.community.label_propagation_communities(G))
    print(f"partition: {len(nx.community.label_propagation_communities(G))}")
    print(f"Modularity: {m}")

    #
    # # Number of real communities
    # num_communities = len(set(communities.values()))
    # print(f"Number of real communities: {num_communities}")


def MI(G):
    # list = rank_by_PR_curv(G, alpha=0.5, beta=0.5)
    # print(f"MI of PR_curv: {tools.cal_MI(list)}")
    for name in G.graph.keys():
        if 'rank' in name:
            m = tools.cal_MI(G.graph[name])
            print(f"MI of {name}: {m}")
        else:
            continue

names = ['karate', 'dolphins', 'football', 'email-Eu-core', 'cora', 'PGP']  #  dolphins  cora com-dblp email-Eu-core infect-dublin  polbooks karate football  PGP
# process_real_net(name, 'mtx')
for name in names:
    G = nx.read_gml(os.path.join('./data/real-world/{}/{}.gml'.format(name, name)))
    print(name)
    # MI(G)
    print_network_stats(G)

# print_network_stats(G)
# print(G)
