import argparse
from tools import add_flags_from_config

config_args = {
    'config':{
        'momentum': (0.999, 'momentum in optimizer'),
        'recalculate': (False, '是否重新计算节点排序列表'),
        'beta': ('adaptive','传染率,adaptive表示自适应传染率'),
        'infected_ratio': (0.1, '初始感染比例'),
        'plotremove':(True, '是否画拆解图'),
        'threshold_dismantling':(0.05, '网络拆解之后剩余节点比例'),
        'drop_precent':(0.1, '删除负曲率的比例'),
        }}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)