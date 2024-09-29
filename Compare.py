import os
import time
from Rank_methods import *
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from config import parser

methods = ['degree', 'curv', 'curv2','PR']


def generate_rank_list(G, args, methods = methods):
    datasetname = os.path.split(G.graph['path'])[1][0:-4]
    if 'degree' in methods:
        try:
            G.graph['rank_degree']
        except KeyError:
            time_start = time.time()
            G.graph['rank_degree'] = rank_by_degree(G)
            time_end = time.time()  # 记录结束时间
            time_sum = time_end - time_start
            logger.info('{} 数据集,DC方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))
        finally:
            if args.recalculate is True:
                time_start = time.time()
                G.graph['rank_degree'] = rank_by_degree(G)
                time_end = time.time()  # 记录结束时间
                time_sum = time_end - time_start
                logger.info('{} 数据集,重新计算的DC方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))

    if 'curv' in methods:
        try:
            G.graph['rank_curv']
        except KeyError:
            time_start = time.time()
            G.graph['rank_curv'] , G.graph['rank_curv2']= rank_by_curv(G)
            time_end = time.time()
            time_sum = time_end - time_start
            logger.info('{} 数据集,curv方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))
        finally:
            if args.recalculate is True:
                time_start = time.time()
                G.graph['rank_curv'] , G.graph['rank_curv2'] = rank_by_curv(G)
                time_end = time.time()
                time_sum = time_end - time_start
                logger.info('{} 数据集,重新计算的curv方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))


def network_dismantling_plot(G, args):
    datasetname = os.path.split(G.graph['path'])[1][0:-4]
    colors = ['green', 'black', 'red', 'blue', 'cyan', 'magenta', 'yellow', 'purple', 'orange', 'brown']
    i = 0
    num_ND_array = np.zeros((nx.number_of_nodes(G)-2, 1))
    for method in methods:
        rank = G.graph['rank_{}'.format(method)]
        x, y = tools.network_dismantling(G, rank, 1, method=method)
        x = np.array(x)
        # np.save(os.path.join(datapath, 'x.npy'), x)
        y = np.array(y)
        # np.save(os.path.join(datapath, 'y.npy'), y)
        if args.plotremove is True:
            plt.plot(x, y, marker='v', color=colors[i], label='{}'.format(method), linewidth=2, markersize=5)
            logger.info("现在开始画图，处理的是{}数据集".format(datasetname))
        else:
            logger.info("你选择了不画图，处理的是{}数据集".format(datasetname))
        stop_t = args.threshold_dismantling
        stop = tools.return_index(y, stop_t)
        print('{}方法AUC值：{}'.format(method, tools.cal_area_R(y)))
        i += 1
        logger.info("模型的暂停index为{},比例是{:.8f},阈值为{}".format(stop, stop/len(G.nodes()), stop_t))
        num_ND_array = np.concatenate((num_ND_array, y.reshape(-1, 1)), axis=1)
    # np.save(os.path.join(os.path.split(G.graph['path'])[0],'{}_num_ND.npy'.format(datasetname, datasetname)), num_ND_array)
    # np_to_csv = pd.DataFrame(data=num_ND_array)
    # np_to_csv.to_csv(os.path.join(os.path.split(G.graph['path'])[0],'{}_num_ND.csv'.format(datasetname, datasetname)),
    #                  index=False)

    plt.title('{}'.format(datasetname))
    plt.legend()
    plt.show()

#path = './data/BA_{}.gml'.format(1003)
# path = './data/BA_{}.gml'.format(100)
path = './data/LFR_{}.gml'.format(200)

G = nx.read_gml(path, destringizer = int, label='id')

args = parser.parse_args()
# G = nx.barabasi_albert_graph(100,2)
# path = './data/synthetic/BA_{}/BA_{}.gml'.format(100,100)
# G.graph['path'] = path
generate_rank_list(G, args)
network_dismantling_plot(G, args)
#输出节点度值
print(G.degree())

