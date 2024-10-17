import networkx as nx


def export_graph_to_txt_CI(graph, filename):
    """
    Export a networkx graph to a text file in adjacency list format.
    Each line in the file will start with a node ID, followed by the IDs of its neighbors, separated by spaces.
    """
    with open(filename, 'w') as f:
        for node in graph.nodes():
            neighbors = ' '.join(map(str, graph.neighbors(node)))
            f.write(f"{node} {neighbors}\n")

def collective_influence_l(G, l=1, **kwargs):

    import tempfile
    from subprocess import check_output, STDOUT, run
    from os import remove, close
    import numpy as np

    folder = f"CI/"
    # cd_cmd = f"cd {folder} && "

    G.remove_edges_from(nx.selfloop_edges(G))

    nodes = list(G.nodes)
    # 创建从原始节点编号到从1开始的连续编号的映射
    node_id_mapping = {node: i for i, node in enumerate(nodes, start=1)}
    # 创建反向映射
    reverse_node_id_mapping = {v: k for k, v in node_id_mapping.items()}

    network_fd, network_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()

    # network_path = 'network.txt'
    # network_fd = 'network.txt'
    # output_path = 'output.txt'
    # output_fd = 'output.txt'

    tmp_file_handles = [network_fd, output_fd]
    tmp_file_paths = [network_path, output_path]

    folder = f"CI/"
    cd_cmd = f"cd {folder} && "

    cmds = [
        "make clean",
        "make",
        f"./CI {network_path} {l} {output_path}"
        # f'./CI {network_path} {l} {stop_condition / network.num_vertices()} {output_path}'
    ]
    nodes = []
    try:
        # with open(network_fd, 'w') as f:  #network_fd 将网络临时转换成CI可以处理的txt文件
        #     for node in G.nodes():
        #         neighbors = ' '.join(map(str, G.neighbors(node)))
        #         f.write(f"{node} {neighbors}\n")
        with open(network_fd, 'w') as f:  #network_fd 将网络临时转换成CI可以处理的txt文件
            for node in G.nodes():
                # neighbors = ' '.join(map(str, G.neighbors(node)))
                neighbors = ' '.join(str(node_id_mapping[neighbor]) for neighbor in G.neighbors(node))
                node_new = node_id_mapping[node]
                # f.write(f"{node} {neighbors}\n")
                f.write(f"{node_new} {neighbors}\n")

        for cmd in cmds:
            try:
                print(f"Running cmd: {cmd}")
                print(
                    # check_output(
                    run(
                        cd_cmd + cmd,
                        # cmd,
                        shell=True,
                        text=True,
                        stderr=STDOUT,
                    )
                )
            except Exception as e:
                raise RuntimeError(f"ERROR! When running cmd: {cmd}. {e}")

        with open(output_fd, "r+") as tmp:
            for line in tmp.readlines():
                # if not line.startswith('#') and not line.startswith(''):
                _, node, _, _ = line.strip().split("\t")
                node = reverse_node_id_mapping[int(node)]

                nodes.append(int(node))
            # print(nodes)
    except Exception as e:
        raise e

    finally:
        for fd, path in zip(tmp_file_handles, tmp_file_paths):
            try:
                close(fd)

            except:
                pass

            try:
                remove(path)

            except:
                pass
    return nodes

# G = nx.karate_club_graph()
# collective_influence_l(G)