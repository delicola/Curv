import argparse
import tools

nettype = 'real-world'
# nettype = 'synthetic'
name = 'football'

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

config_args = {
    'config':{
        'momentum': (0.999, 'momentum in optimizer'),
        'recalculate': (False, '是否重新计算节点排序列表'),
        'beta': ('adaptive','传染率,adaptive表示自适应传染率'),
        'infected_ratio': (0.1, '初始感染比例'),
        'plotremove':(True, '是否画拆解图'),
        'threshold_dismantling':(0.05, '网络拆解之后剩余节点比例'),
        'drop_percent':(0.2, '删除负曲率边的比例'),
        'epoch':(100, '训练的epoch'),
        'nettype':('real-world', '网络类型'),
        'name':(name, '网络名称'),
        'dataset_path':('./data/{}/{}/{}.gml'.format(nettype,name,name), '数据集位置'),
        'seed':(2022, '随机种子'),
        'savegraph':(True, '是否保存图'),
        'show':(False, '是否展示图'),
        }}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)