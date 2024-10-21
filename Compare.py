import os
import time
from Rank_methods import rank_by_curv, rank_by_degree, rank_by_PR, rank_by_CI
import tools
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from config import parser
import random
import torch




def generate_rank_list(G, args, methods):
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

    if 'PR' in methods:
        try:
            G.graph['rank_PR']
        except KeyError:
            time_start = time.time()
            G.graph['rank_PR'] = rank_by_PR(G)
            time_end = time.time()
            time_sum = time_end - time_start
            logger.info('{} 数据集,PR方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))
        finally:
            if args.recalculate is True:
                time_start = time.time()
                G.graph['rank_PR'] = rank_by_PR(G)
                time_end = time.time()
                time_sum = time_end - time_start
                logger.info('{} 数据集,重新计算的PR方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))

    if 'CI' in methods:
        try:
            G.graph['rank_CI']
        except KeyError:
            time_start = time.time()
            G.graph['rank_CI'] = rank_by_CI(G)
            time_end = time.time()
            time_sum = time_end - time_start
            logger.info('{} 数据集,CI方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))
        finally:
            if args.recalculate is True:
                time_start = time.time()
                G.graph['rank_CI'] = rank_by_CI(G)
                time_end = time.time()
                time_sum = time_end - time_start
                logger.info('{} 数据集,重新计算的CI方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))


def network_dismantling_plot(G, args):
    datasetname = os.path.split(G.graph['path'])[1][0:-4]
    colors = ['green', 'black', 'red', 'blue', 'cyan', 'magenta', 'yellow', 'purple', 'orange', 'brown']
    i = 0
    num_ND_array = np.zeros((nx.number_of_nodes(G)-2, 1))
    for method in methods:
        #获取当前时间

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
        logger.info('{}方法AUC值：{}'.format(method, tools.cal_area_R(y)))
        i += 1
        logger.info("模型的暂停index为{},比例是{:.8f},阈值为{}".format(stop, stop/len(G.nodes()), stop_t))
        num_ND_array = np.concatenate((num_ND_array, y.reshape(-1, 1)), axis=1)

    # np.save(os.path.join(os.path.split(G.graph['path'])[0],'{}_num_ND.npy'.format(datasetname, datasetname)), num_ND_array)
    # np_to_csv = pd.DataFrame(data=num_ND_array)
    # np_to_csv.to_csv(os.path.join(os.path.split(G.graph['path'])[0],'{}_num_ND.csv'.format(datasetname, datasetname)),
    #                  index=False)
    if args.savegraph is True:
        now = time.localtime()
        t = time.strftime("%m%d%H%M%S", now)
        # 判断是否有pic文件夹，没有则创建一个文件夹
        if not os.path.exists('./data/{}/{}/pic'.format(args.nettype, args.name)):
            os.makedirs('./data/{}/{}/pic'.format(args.nettype, args.name))
            plt.savefig('./data/{}/{}/pic/{}_{}_LCCsize.png'.format(args.nettype, args.name, args.name, t))
        else:
            plt.savefig('./data/{}/{}/pic/{}_{}_LCCsize.png'.format(args.nettype, args.name, args.name, t))
        logger.info("保存图片 {}".format(t))
    plt.title('{}'.format(datasetname))
    plt.legend()
    if args.show is True:
        plt.show()
    plt.close()

#path = './data/BA_{}.gml'.format(1003)
# path = './data/BA_{}.gml'.format(100)
# path = './data/LFR_{}.gml'.format(200)
# name = 'polbooks'  #football polbooks
# path = './data/real-world/{}/{}.gml'.format(name,name)
    if 'PR' in methods:
        try:
            G.graph['rank_PR']
        except KeyError:
            time_start = time.time()
            G.graph['rank_PR'] = rank_by_PR(G)
            time_end = time.time()
            time_sum = time_end - time_start
            logger.info('{} 数据集,PR方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))
        finally:
            if args.recalculate is True:
                time_start = time.time()
                G.graph['rank_PR'] = rank_by_PR(G)
                time_end = time.time()
                time_sum = time_end - time_start
                logger.info('{} 数据集,重新计算的PR方法生成节点排序列表的时间是：{}s'.format(datasetname, time_sum))
if __name__ == '__main__':
    methods = ['degree', 'curv', 'curv2', 'PR', 'CI']
    # methods = ['degree', 'curv', 'curv2', 'PR']
    #methods = ['degree', 'PR']
    args = parser.parse_args()
    seed_value = args.seed  # 设定随机数种子

    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

    torch.backends.cudnn.deterministic = True

    path = './data/{}/{}/{}.gml'.format(args.nettype,args.name,args.name)

    logger.add('./data/{}/{}/{}.log'.format(args.nettype,args.name,args.name))

    logger.info('现在处理的是 {} 数据集\n 随机种子是 {} 删边比例是 {} 训练了 {} 次'.format(args.name, args.seed, args.drop_percent, args.epoch))

    #G = nx.read_gml(path, destringizer = int, label='id')

    # edges = [
    #     (0, 1), (0, 5), (1, 5), (2, 3), (2, 6), (2, 7), (3, 7), (3, 8),
    #     (4, 5), (4, 9), (5, 6), (5, 9), (5, 10), (6, 7), (6, 12),
    #     (6, 10), (7, 8), (7, 12), (10, 11), (10, 16), (10, 14), (10, 15),
    #     (11, 12), (11, 15), (11, 16), (13, 14), (15, 16)
    # ]

    edges_old = [
        (1, 2), (1, 3), (2, 1), (2, 3), (2, 5),
        (3, 1), (3, 2), (3, 4), (4, 3), (4, 5),
        (4, 8), (4, 9), (5, 2), (5, 4), (5, 6),
        (6, 5), (6, 7), (7, 6), (7, 8), (8, 4),
        (8, 7), (9, 4), (9, 10), (9, 12), (9, 13),
        (10, 9), (10, 11), (11, 10), (12, 9), (13, 9),
        (13, 14), (14, 13), (14, 15), (14, 16), (15, 14),
        (16, 14)
    ]

    # 将边直接转换为从0开始编号
    edges = [(u - 1, v - 1) for u, v in edges_old]


    G = nx.Graph()
    G.add_edges_from(edges)
    G.graph['path'] = './my.gml'


    args = parser.parse_args()
    print(args)
    # G = nx.barabasi_albert_graph(100,2)
    # path = './data/synthetic/BA_{}/BA_{}.gml'.format(100,100)
    # G.graph['path'] = path
    generate_rank_list(G, args, methods=methods)
    network_dismantling_plot(G, args)
    #输出节点度值
    print(G.degree())

