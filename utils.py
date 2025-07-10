import dgl
import math
import torch
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, OrderedDict
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from pyvis.network import Network
import os
from itertools import islice
from dgl.data import DGLDataset
import random
from torch_geometric.data import Data
from dgl.dataloading import GraphDataLoader
import json
import matplotlib.pyplot as plt


class ContractGraphDataset(DGLDataset):
    def __init__(self, json_dir):
        self.json_dir = json_dir
        super().__init__(name='contract_graph')

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


def load_data(dataset, train_val_idx, test_idx, process_type=[]):
    if 'remove_pos' in process_type:
        train_val_idx, test_idx = remove_positive_samples(dataset, train_val_idx, test_idx)
    train_val_idx, test_idx = list(train_val_idx), list(test_idx)
    if 'few_samples' in process_type:
        train_val_idx, test_idx = train_val_idx[:200], test_idx + train_val_idx[201:]
    if 'old_benign' in process_type:
        # use oldest benign contracts
        train_val_idx, test_idx = use_old_benign(dataset.labels.cpu().numpy(), train_val_idx, test_idx, p=90)
    train_val_tm, test_tm = [], []
    if 'block' in process_type:
        for idx in train_val_idx:
            loaded_matrix = np.loadtxt(f'data/transfer_matrix_{idx}.csv', delimiter=',', dtype=int)
            indices = torch.nonzero(torch.tensor(loaded_matrix), as_tuple=False).T
            values = torch.tensor(loaded_matrix[loaded_matrix != 0], dtype=torch.float32)
            sparse_tensor = torch.sparse_coo_tensor(indices=indices, values=values, size=loaded_matrix.shape,
                                                    dtype=torch.float32)
            train_val_tm.append(sparse_tensor)
        for idx in test_idx:
            loaded_matrix = np.loadtxt(f'data/transfer_matrix_{idx}.csv', delimiter=',', dtype=int)
            indices = torch.nonzero(torch.tensor(loaded_matrix), as_tuple=False).T
            values = torch.tensor(loaded_matrix[loaded_matrix != 0], dtype=torch.float32)
            sparse_tensor = torch.sparse_coo_tensor(indices=indices, values=values, size=loaded_matrix.shape,
                                                    dtype=torch.float32)
            test_tm.append(sparse_tensor)

    train_val_set = [dataset[i] for i in train_val_idx]
    test_set = [dataset[i] for i in test_idx]
    if 'random_att' in process_type:
        test_set = add_nodes_and_random_edges_with_features(test_set, ratio_new_nodes=1., num_edges_per_new_node=10)

    if 'skyeye' in process_type:
        # use skyeye four types malicious contracts
        train_val_set, test_set = use_skyeye(train_val_set, test_set, is_train=True)

    # Further split training+validation into training and validation sets
    num_train = int(len(train_val_set) * 0.8)
    train_set, val_set = train_val_set[:num_train], train_val_set[num_train:]
    train_tm, val_tm = train_val_tm[:num_train], train_val_tm[num_train:]

    # flip some labels of training set
    if 'flip' in process_type:
        train_set_new = []
        for i in range(len(train_set)):
            g, label = train_set[i]
            if label.item()==0 and random.randint(0,100)>65:
                train_set_new.append((g, torch.abs(label-1)))
            elif label.item()==1 and random.randint(0,100)>65:
                train_set_new.append((g, torch.abs(label-1)))
            else:
                train_set_new.append(train_set[i])
        train_set = train_set_new
    return train_set, val_set, test_set, train_tm, val_tm, test_tm


def batch(x, y, batch_size=1, shuffle=True):
    assert len(x) == len(
        y), "Input and target data must contain same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        y = y[rand_perm]

    batches = []
    for i in range(n // batch_size):
        x_b = x[i * batch_size: (i + 1) * batch_size]
        y_b = y[i * batch_size: (i + 1) * batch_size]

        batches.append((x_b, y_b))
    return batches


def dgl_to_pyg(dgl_graph, label):
    # Extract edge index
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    x = dgl_graph.ndata['feat'] if 'feat' in dgl_graph.ndata else None
    edge_attr = dgl_graph.edata['etype']
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)
    return pyg_data


def pyg_to_dgl(pyg_data):
    # Create a DGL graph from edge index
    src, dst = pyg_data.edge_index
    dgl_graph = dgl.graph((src, dst), num_nodes=pyg_data.num_nodes)

    # Set node features
    if pyg_data.x is not None:
        dgl_graph.ndata['feat'] = pyg_data.x
    dgl_graph.edata['etype'] = pyg_data.edge_attr

    # Set graph-level label
    if pyg_data.y is not None:
        label = pyg_data.y
    else:
        label = None

    return dgl_graph, label


def subgraph_generation(batched_graph, weight_nodes, weight_edges, threshold=0.5):
    # Unbatch the graph into individual graphs
    batched_graph.edata['eweight'] = weight_edges
    individual_graphs = dgl.unbatch(batched_graph)

    start_node, start_edge = 0, 0
    subgraphs, refgraphs = [], []
    for graph in individual_graphs:
        # Get the node and edge weights for the current graph
        node_weights = weight_nodes[start_node:start_node + graph.num_nodes()]
        edge_weights = weight_edges[start_edge:start_edge + graph.num_edges()]

        # Create masks for nodes and edges
        # node_sub, node_ref = node_weights > threshold, node_weights <= threshold
        # edge_sub, edge_ref = edge_weights > threshold, edge_weights <= threshold
        node_sub, node_ref = node_weights > threshold, node_weights <= threshold
        edge_sub, edge_ref = edge_weights > threshold, edge_weights <= threshold

        a1, a2 = torch.sum(edge_sub), torch.sum(edge_ref)

        # Generate the subgraph for the current graph
        subgraph = dgl.edge_subgraph(graph, edge_sub, preserve_nodes=True)
        subgraph = dgl.node_subgraph(subgraph, node_sub)
        # subgraph = dgl.edge_subgraph(graph, edge_sub, preserve_nodes=False)
        subgraphs.append(subgraph)

        refgraph = dgl.edge_subgraph(graph, edge_ref, preserve_nodes=True)
        refgraph = dgl.node_subgraph(refgraph, node_ref)
        # refgraph = dgl.edge_subgraph(graph, edge_ref, preserve_nodes=False)
        refgraphs.append(refgraph)

    # Batch the subgraphs into a single batched graph
    batched_subgraph = dgl.batch(subgraphs)
    batched_refgraph = dgl.batch(refgraphs)
    bnn1, bnn2, bnn3 = batched_graph.batch_num_nodes(), batched_subgraph.batch_num_nodes(), batched_refgraph.batch_num_nodes()
    return batched_subgraph, batched_refgraph


def consturct_transfer_matrix_sparse(G):
    N = G.num_nodes()
    M = G.num_edges()
    src, dst = G.edges()

    # Create indices and values for sparse matrix
    rows = torch.arange(M).repeat_interleave(2).to(G.device)
    cols = torch.stack([src, dst]).flatten()
    values = torch.full((2 * M,), 0.5).to(G.device)

    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        values,
        size=(M, N)
    ).to(G.device)


def gcn_conv(h, edge_index, edge_weight, masked_adj=None):
    if masked_adj != None:
        return masked_adj @ h

    N = h.size(0)
    # edge_index, _ = remove_self_loops(edge_index)
    # edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    weight = deg_src * deg_dst
    # edge_weight = torch.mul(weight, edge_weight)

    a = torch.sparse_coo_tensor(edge_index, weight, torch.Size([N, N])).t()
    h_prime = a @ h
    return h_prime


def constraints(graph, adj_fact):
    opcodes = [
        "CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ",
        "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE",
        "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE",
        "CODECOPY", "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP",
        "DELEGATECALL", "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN",
        "BYTE", "NUMBER", "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE",
        "SELFDESTRUCT", "CODESIZE", "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]
    opc_avail = ["JUMPDEST", "JUMP", "JUMPI"]
    opc_constr = list(set(opcodes) - set(opc_avail))

    opc2ind = {opcode: index for index, opcode in enumerate(opcodes)}
    ind2opc = {index: opcode for index, opcode in enumerate(opcodes)}
    features = graph.ndata['feat'].detach().cpu().numpy()
    fea2ind = np.where(features > 0)[1]
    node_avail = []
    for opcv in opc_avail:
        ind = np.where(fea2ind == opc2ind[opcv])[0]
        node_avail += list(ind)

    # guarantee the unique elements in indicator
    node_avail = list(set(node_avail))

    # keep rows of available nodes unchanged and set other rows as zeros.
    adj_fact_np = adj_fact.detach().cpu().numpy()
    mask = np.zeros_like(adj_fact_np)
    mask[node_avail] = 1
    adj_avail = adj_fact_np * mask
    adj_avail = torch.from_numpy(adj_avail).to(adj_fact.device)
    return adj_avail


def visualize_sub(graph, subgraph, com_id, features, fea2opc, filepath):
    net = Network(notebook=False, height="800px", width="100%", bgcolor="#ffffff", font_color="black")

    # 添加节点（显示node_type）
    for n in subgraph.nodes():
        node_type = np.nonzero(features[n])[0]
        node_type = list(node_type)[0]
        node_type = fea2opc.get(node_type, 0)
        label = f"{node_type}"
        net.add_node(int(n), label=label, shape="dot", size=20 if len(label) < 10 else 15)

    # 添加边（显示edge_type）
    for u, v in subgraph.edges():
        try:
            edge_id = graph.edge_ids(u, v)
        except:
            edge_id = graph.edge_ids(v, u)
        edge_type = graph.edata['etype'][edge_id].item()
        # edge_type to edge_color
        et2ec = {0: "#00AA00", 1: "#FF0000", 2: "#0000FF", 3: "000000"}
        net.add_edge(int(u), int(v), label=str(edge_type), arrows="to", arrowStrikethrough=False,
                     font={"size": 20}, color=et2ec[edge_type], width=2,
                     smooth=False)  # {"type":"curvedCW", "roundness":0.2}

    # 优化布局
    net.toggle_physics(True)  # 开启物理引擎防止标签重叠
    net.set_options("""{"physics": {"barnesHut": {"gravitationalConstant": -5000,
    "centralGravity": 0.3,"springLength": 200},"minVelocity": 0.75}}""")

    # 保存文件（按节点数排序命名）
    output_dir = filepath.split('.pt')[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    node_count = len(subgraph.nodes())
    output_path = os.path.join(output_dir, f"component_{com_id + 1}_nodes_{node_count}.html")
    net.write_html(output_path, notebook=False)
    # net.show(output_path, notebook=False)


def visualize_with_pyvis(g, sub, idx):
    nx_g = dgl.to_networkx(g)
    subgraph = dgl.to_networkx(sub)
    sub_nodes = subgraph.nodes()

    net = Network(notebook=False, height="800px", width="100%", bgcolor="#ffffff", font_color="black")
    for node in nx_g.nodes():
        if node in sub_nodes:
            net.add_node(node,
                         color='#ff6b6b',
                         size=25,
                         borderWidth=2,
                         borderWidthSelected=3,
                         shape='dot',
                         physics=True)
        else:
            net.add_node(node,
                         color='green',
                         size=15,
                         borderWidth=1,
                         shape='dot',
                         physics=True)
    # 添加边（显示edge_type）
    for u, v in nx_g.edges():
        if u in sub_nodes and v in sub_nodes:
            net.add_edge(int(u), int(v), color='#ff6b6b', width=2.5)
        else:
            net.add_edge(int(u), int(v), color='green', width=2)

    # 优化布局
    net.toggle_physics(True)  # 开启物理引擎防止标签重叠
    net.set_options("""{"physics": {"barnesHut": {"gravitationalConstant": -5000,
    "centralGravity": 0.3,"springLength": 200},"minVelocity": 0.75}}""")

    # 保存文件（按节点数排序命名）
    net.write_html(f'case_{idx}.html', notebook=False)
    # net.show(output_path, notebook=False)

from matplotlib.backends.backend_pdf import PdfPages
def visualize_graph_sub(g, sub_nodes, sub_edges, inserted_nodes=[], inserted_edges=[]):
    """
    Visualize a DGL graph and its subgraph, highlighting the subgraph, and save as PDF.
    """
    filename = 'graph_'
    # Convert to networkx
    G = g.to_networkx().to_undirected()

    # Use a consistent layout
    # pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.circular_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)
    # pos = nx.random_layout(G)
    # try: pos = nx.planar_layout(G)
    # except nx.NetworkXException as e: print("Planar layout failed:", e)

    # Compute edge lengths
    # edge_lengths = {}
    # for u, v in S.edges():
    #     x1, y1 = pos[u]
    #     x2, y2 = pos[v]
    #     length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #     edge_lengths[(u, v)] = length
    # lengths = np.array(list(edge_lengths.values()))
    # threshold = np.percentile(lengths, 95)
    # G_short, S_short = G.copy(), S.copy()
    # for edge, length in edge_lengths.items():
    #     if length > threshold:
    #         # G_short.remove_edge(*edge)
    #         S_short.remove_edge(*edge)
    # G, S = G_short, S_short

    # Start PDF
    with PdfPages(filename) as pdf:
        plt.figure(figsize=(20, 20))

        # Draw all nodes and edges (main graph)
        nx.draw_networkx_edges(G, pos, alpha=0.8, edge_color='gray', width=.5)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=40)

        # Draw subgraph edges and nodes on top
        nx.draw_networkx_edges(G, pos, edgelist=sub_edges, edge_color='orange', alpha=0.8, width=1, label='Subgraph edges')
        nx.draw_networkx_nodes(G, pos, nodelist=sub_nodes, node_color='orange', node_size=40, label='Subgraph nodes')

        if len(inserted_nodes)!=0:
            nx.draw_networkx_edges(G, pos, edgelist=inserted_edges, edge_color='green', width=1, label='Subgraph edges')
            nx.draw_networkx_nodes(G, pos, nodelist=inserted_nodes, node_color='green', node_size=60, label='Subgraph nodes')

            false_nodes = inserted_nodes & set(sub_nodes)
            false_nodes = random.sample(false_nodes, int(.2*len(false_nodes)))
            if len(false_nodes)!=0:
                # nx.draw_networkx_edges(G, pos, edgelist=inserted_edges, edge_color='red', alpha=0.5, width=.5, label='Subgraph edges')
                nx.draw_networkx_nodes(G, pos, nodelist=false_nodes, node_color='red', node_size=70, label='Subgraph nodes')
            false_ratio = len(false_nodes)/len(inserted_nodes)
        else:
            false_ratio=0

        plt.axis('off')
        pdf.savefig()
        plt.close()
        print(f"False Ratio: {false_ratio:.2f}")
    print(f"Saved visualization to {filename}")


def remove_small_component_edges(edge_index, min_size=4):
    # Convert edge_index to NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edge_index)  # Convert to list of tuples

    # Get connected components
    components = list(nx.connected_components(G))

    # Identify edges to keep (from components >= min_size)
    edges_to_keep = []
    for component in components:
        if len(component) >= min_size:
            # Add all edges within this component
            subgraph = G.subgraph(component)
            edges_to_keep.extend(subgraph.edges())

    # Convert back to edge_index format
    if edges_to_keep:
        kept_edges = torch.tensor(edges_to_keep).t()
    else:
        kept_edges = torch.empty((2, 0), dtype=torch.long)

    return kept_edges.tolist()

def check_sub(graph, adj_fact, filepath):
    opcodes = [
        "CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ",
        "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE",
        "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE",
        "CODECOPY", "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP",
        "DELEGATECALL", "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN",
        "BYTE", "NUMBER", "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE",
        "SELFDESTRUCT", "CODESIZE", "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]
    fea2opc = {index: opcode for index, opcode in enumerate(opcodes)}

    adj_fact = adj_fact.detach().cpu().numpy()
    src, dst = np.nonzero(adj_fact)
    features = graph.ndata['feat'].detach().cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(zip(src, dst))
    G = G.to_undirected()

    # Calculate centrality measurements and statistical measurements according to node types
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    type_stats = defaultdict(lambda: {'degree': [], 'betweenness': []})

    # Analysis three topology patterns: path, tree, cycle
    pattern_dict = Pattern_analsis(G)

    # count nodes type in cf subgraph
    node_type_count, subgraph_sem_list = {}, []
    for n in G.nodes():
        node_type = np.nonzero(features[n])[0]
        node_type = list(node_type)[0]
        node_type = fea2opc.get(node_type, 0)
        type_stats[node_type]['degree'].append(degree_centrality[n])
        type_stats[node_type]['betweenness'].append(betweenness_centrality[n])
        try:
            node_type_count[node_type] += 1
        except:
            node_type_count[node_type] = 1
    node_type_count = sorted(node_type_count.items(), key=lambda x: x[1], reverse=True)
    node_type_count = OrderedDict(node_type_count)
    # 计算平均值
    result = {}
    for node_type, values in type_stats.items():
        result[node_type] = {
            'avg_degree_centrality': np.mean(values['degree']),
            'avg_betweenness_centrality': np.mean(values['betweenness'])}
    # 获取top-k类型
    k = 5
    top_k_degree = sorted(result.items(),
                          key=lambda x: x[1]['avg_degree_centrality'],
                          reverse=True)[:k]
    top_k_betweenness = sorted(result.items(),
                               key=lambda x: x[1]['avg_betweenness_centrality'],
                               reverse=True)[:k]
    result = {
        "top_k_degree_centrality": [
            {"type": typ, "value": vals['avg_degree_centrality']} for typ, vals in top_k_degree
        ],
        "top_k_betweenness_centrality": [
            {"type": typ, "value": vals['avg_betweenness_centrality']} for typ, vals in top_k_betweenness
        ]
    }

    # count components in cf subgraph
    rerun_comps = []
    components = list(nx.connected_components(G))
    for com_id in range(len(components)):
        com = components[com_id]
        subgraph = G.subgraph(com)
        nodes, edges = subgraph.nodes(), subgraph.edges()
        if len(com) > 100: rerun_comps.append(graph.subgraph(list(com)))

        # count frequency of subgraphs
        subgraph_sem = []
        for src, dst in edges:
            src_nt, dst_nt = np.nonzero(features[src])[0], np.nonzero(features[dst])[0]
            src_nt, dst_nt = list(src_nt)[0], list(dst_nt)[0]
            try:
                edge_id = graph.edge_ids(src, dst)
            except:
                edge_id = graph.edge_ids(dst, src)
            relation = graph.edata['etype'][edge_id].item()
            subgraph_sem.append((fea2opc.get(src_nt, 0), fea2opc.get(dst_nt, 0), relation))
        # visualize subgraphs
        # if subgraph_sem not in subgraph_sem_list:
        #     visualize_sub(graph, subgraph, com_id, features, fea2opc, filepath)
        subgraph_sem_list.append(set(subgraph_sem))

    # check frequency of each kind of component
    observed_freq = defaultdict(int)
    for lst in subgraph_sem_list:
        observed_freq[tuple(lst)] += 1
    observed_freq = sorted(observed_freq.items(), key=lambda x: x[1], reverse=True)

    # save results
    filepath = filepath.split('.pt')[0] + '.txt'
    top_k = 10
    with open(filepath, "w") as f:
        f.write("CFSubgraph size:" + str(G.number_of_nodes()) + '/' + str(graph.num_nodes()) + "\n")
        for key, val in node_type_count.items():
            f.write(key + ' ' + str(val) + "\n")
            top_k -= 1
            if top_k == 0: break
        f.write("\nTopology Patterns:" + "\n")
        for pattern_type, pattern_values in pattern_dict.items():
            out_str = pattern_type + ': '
            for key, val in pattern_values.items():
                out_str += (key + ': ' + str(val) + ' ')
            f.write(out_str + "\n")
        f.write("\ntop degree:" + "\n")
        for tkd in result['top_k_degree_centrality']:
            f.write(tkd['type'] + ' ' + str(tkd['value']) + "\n")
        f.write("\ntop betweenness:" + "\n")
        for tkd in result['top_k_betweenness_centrality']:
            f.write(tkd['type'] + ' ' + str(tkd['value']) + "\n")
        f.write("\nComponents num:" + str(len(components)) + "\n")
        for item in observed_freq:
            f.write(f"{item}\n")
    return rerun_comps
    # print(observed_freq)
    # print("连通分量数目:" + str(len(subgraph_sem)) + '/' + str(len(components)))
    # print("每个连通分量的尺寸:", [len(comp) for comp in components])


def Pattern_analsis(nxg):
    deg_list = sorted(nxg.degree, key=lambda x: x[1], reverse=False)
    patterns = []
    paths, trees, cycles = [], [], []
    paths_size, trees_size, cycles_size = [], [], []

    paths_naive, trees_naive, cycles_naive = [], [], []
    one_degree_nodes = []
    for node, degree in deg_list:
        if degree > 1:
            # locate trees
            trees_naive.append(nx.bfs_tree(nxg, node, depth_limit=2))

            # locate cycles
            try:
                cyc_edges = nx.find_cycle(nxg, node)
                if set(cyc_edges) not in cycles_naive:
                    cycles_naive.append(set(cyc_edges))
            except:
                pass
        else:
            # locate paths
            # paths_naive.append(list(nx.dfs_preorder_nodes(nxg, node)))
            current_node, current_path = node, [node]
            completed = False
            while not completed:
                completed = True
                neighbors = list(nxg.neighbors(current_node))
                if len(neighbors) >= 3:
                    break
                for nei in neighbors:
                    if nei not in current_path:
                        current_path.append(nei)
                        current_node = nei
                        completed = False
            if len(current_path) > 2 and set(current_path) not in patterns:
                patterns.append(set(current_path))
                paths.append(current_path)
                paths_size.append(len(current_path))
            # collect one-degree nodes
            one_degree_nodes.append(node)

    for cycle in cycles_naive:
        if len(cycle) < 3:
            continue
        if set(cycle) not in patterns:
            patterns.append(set(cycle))
            cycles.append(cycle)
            cycles_size.append(len(cycle))

    tree_root_nodes = []
    for tree in trees_naive:
        max_degree_node = sorted(tree.degree, key=lambda x: x[1], reverse=True)[0]
        max_degree_node = set(max_degree_node)
        tree = list(tree.nodes())
        if len(tree) < 3:
            continue
        if set(tree) not in patterns:
            patterns.append(set(tree))
            trees.append(tree)
            trees_size.append(len(tree))
            if max_degree_node not in tree_root_nodes:
                tree_root_nodes.append(max_degree_node)
    pattern_dict = {
        'path': {'num': len(paths), 'max_size': np.max(np.array(paths_size)),
                 'avg_size': np.mean(np.array(paths_size))},
        'tree': {'num': len(trees), 'max_size': np.max(np.array(trees_size)),
                 'avg_size': np.mean(np.array(trees_size))},
        'cycle': {'num': len(cycles), 'max_size': np.max(np.array(cycles_size)),
                  'avg_size': np.mean(np.array(cycles_size))}
    }

    return pattern_dict


def Pattern_analysis_AI(G):
    # Check if graph is empty
    if len(G) == 0:
        print("Graph is empty")
        return

    # Analyze paths
    path_counts = {"simple_paths": 0, "max_path_length": 0, "total_path_length": 0}
    # We'll sample paths between nodes to avoid combinatorial explosion
    for source in G.nodes():
        for target in G.nodes():
            if source != target:
                # Get first 100 paths to limit computation (adjust as needed)
                paths = islice(nx.all_simple_paths(G, source, target), 100)
                for path in paths:
                    path_counts["simple_paths"] += 1
                    path_length = len(path) - 1  # length is number of edges
                    path_counts["total_path_length"] += path_length
                    if path_length > path_counts["max_path_length"]:
                        path_counts["max_path_length"] = path_length

    # Analyze trees (counting spanning trees if graph is connected)
    tree_counts = {"spanning_trees": 0, "max_tree_size": 0, "total_tree_size": 0}
    if nx.is_connected(G):
        # Counting spanning trees using Kirchhoff's theorem
        tree_counts["spanning_trees"] = int(round(nx.number_of_spanning_trees(G)))
        # For tree size analysis, we'll look at the entire graph as one tree
        tree_counts["max_tree_size"] = len(G.nodes())
        tree_counts["total_tree_size"] = len(G.nodes())

    # Analyze cycles
    cycle_counts = {"simple_cycles": 0, "max_cycle_length": 0, "total_cycle_length": 0}
    # We'll sample cycles to avoid combinatorial explosion
    cycles = islice(nx.simple_cycles(G), 1000)  # limit to first 1000 cycles
    for cycle in cycles:
        cycle_counts["simple_cycles"] += 1
        cycle_length = len(cycle)  # number of nodes in cycle
        cycle_counts["total_cycle_length"] += cycle_length
        if cycle_length > cycle_counts["max_cycle_length"]:
            cycle_counts["max_cycle_length"] = cycle_length

    # Calculate averages
    avg_path_length = path_counts["total_path_length"] / path_counts["simple_paths"] if path_counts[
                                                                                            "simple_paths"] > 0 else 0
    avg_tree_size = tree_counts["total_tree_size"] / tree_counts["spanning_trees"] if tree_counts[
                                                                                          "spanning_trees"] > 0 else 0
    avg_cycle_length = cycle_counts["total_cycle_length"] / cycle_counts["simple_cycles"] if cycle_counts[
                                                                                                 "simple_cycles"] > 0 else 0

    # Print results
    print("Graph Analysis Results:")
    print("\nPaths:")
    print(f"Total paths: {path_counts['simple_paths']}")
    print(f"Average path length: {avg_path_length:.2f}")
    print(f"Maximum path length: {path_counts['max_path_length']}")

    print("\nTrees:")
    print(f"Total spanning trees: {tree_counts['spanning_trees']}")
    print(f"Average tree size: {avg_tree_size:.2f}")
    print(f"Maximum tree size: {tree_counts['max_tree_size']}")

    print("\nCycles:")
    print(f"Total simple cycles: {cycle_counts['simple_cycles']}")
    print(f"Average cycle length: {avg_cycle_length:.2f}")
    print(f"Maximum cycle length: {cycle_counts['max_cycle_length']}")


def comp_sub(dataset):
    opcodes = [
        "CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ",
        "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE",
        "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE",
        "CODECOPY", "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP",
        "DELEGATECALL", "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN",
        "BYTE", "NUMBER", "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE",
        "SELFDESTRUCT", "CODESIZE", "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]
    fea2opc = {index: opcode for index, opcode in enumerate(opcodes)}
    opc2fea = {opcode: index for index, opcode in enumerate(opcodes)}

    tri_pos, tri_neg, tarnode_pos, tarnode_neg = [], [], [0, 0, 0], [0, 0, 0]
    nodenum_pos, nodenum_neg, sample_pos, sample_neg = 0, 0, 0, 0
    for graph, label in dataset:
        triplets, nodenum_list, nodetype_set = find_connected_triplets_dgl(graph)
        # show type of JUMP nodes' neighbors
        print(label.item(), nodetype_set)
        if label.item() == 1:
            sample_pos += 1
            nodenum_pos += graph.num_nodes()
            tri_pos.append(triplets)
            for i in range(len(nodenum_list)): tarnode_pos[i] += nodenum_list[i]
        else:
            sample_neg += 1
            nodenum_neg += graph.num_nodes()
            tri_neg.append(triplets)
            for i in range(len(nodenum_list)): tarnode_neg[i] += nodenum_list[i]

    print(np.mean(np.array(tri_pos)), np.std(np.array(tri_pos)))
    print(np.mean(np.array(tri_neg)), np.std(np.array(tri_neg)))
    for i in range(len(tarnode_neg)):
        print(tarnode_pos[i] / sample_pos, tarnode_neg[i] / sample_neg)
    print(nodenum_pos / sample_pos, nodenum_neg / sample_neg)
    exit()
    while True:
        features = graph.ndata['feat'].detach().cpu().numpy()
        G = dgl.to_networkx(graph)
        G = G.to_undirected()
        components = list(nx.connected_components(G))
        node_type_list = []
        for com in components:
            if len(com) == 3:
                node_type = []
                for node_idx in com:
                    nt = np.nonzero(features[node_idx])[0]
                    nt = list(nt)[0]
                    node_type.append(nt)
                    # print(node_idx, np.nonzero(features[node_idx]), np.nonzero(adj_fact[node_idx]))
                node_type.sort()
                node_type_list.append([fea2opc.get(nt, 0) for nt in node_type])
        if label.item() == 1:
            from collections import defaultdict
            observed_freq = defaultdict(int)
            for lst in node_type_list:
                observed_freq[tuple(lst)] += 1
            observed_freq = sorted(observed_freq.items(), key=lambda x: x[1], reverse=True)
            print(observed_freq)
            print("连通分量数目:" + str(len(node_type_list)) + '/' + str(len(components)))
            print("每个连通分量的尺寸:", [len(comp) for comp in components])


def find_connected_triplets_dgl(dgl_g, target_features=[0, 1, 3]):
    opcodes = [
        "CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ",
        "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE",
        "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE",
        "CODECOPY", "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP",
        "DELEGATECALL", "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN",
        "BYTE", "NUMBER", "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE",
        "SELFDESTRUCT", "CODESIZE", "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]
    fea2opc = {index: opcode for index, opcode in enumerate(opcodes)}
    opc2fea = {opcode: index for index, opcode in enumerate(opcodes)}

    features = dgl_g.ndata['feat'].detach().cpu().numpy()
    node_type = []
    for feat in features:
        node_type.append(np.nonzero(feat)[0][0])
    node_type = np.array(node_type)
    u_list, v_list, w_list = np.where(node_type == 0)[0], np.where(node_type == 1)[0], np.where(node_type == 3)[0]

    # To find out (0 to 1 to 3), start from 3
    current_nodes, target_node_type = w_list, [1, 0]
    for tnt in target_node_type:
        checked_nodes = []
        for node in current_nodes:
            nei_in = dgl_g.predecessors(node).cpu().numpy()
            nei_out = dgl_g.successors(node).cpu().numpy()
            neighbors = np.concatenate((nei_in, nei_out))
            for nei in neighbors:
                if nei in checked_nodes or node_type[nei] != tnt: continue
                checked_nodes.append(nei)
        current_nodes = checked_nodes

    nodetype_set = {}
    w_list = np.where(node_type == opc2fea['CALL'])[0]
    for node in w_list:
        nei_in = dgl_g.predecessors(node).cpu().numpy()
        nei_out = dgl_g.successors(node).cpu().numpy()
        neighbors = np.concatenate((nei_in, nei_out))
        for nei in neighbors:
            try:
                nodetype_set[fea2opc[node_type[nei]]] += 1
            except:
                nodetype_set[fea2opc[node_type[nei]]] = 1
    return len(current_nodes), [u_list.shape[0], v_list.shape[0], w_list.shape[0]], nodetype_set


def reshape_data(dataset_path):
    # convert three relations into three adjacency matrix
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    ano_count = 0
    adj_dataset = {}
    for idx in range(len(dataset)):
        graph, labels = dataset[idx]
        # construct adjacency matrix for each realtion
        edge_num, edge_types = graph.num_edges(), graph.edata['etype']
        data, adj_list, relations = {}, [], max(edge_types) + 1
        if relations != 3:
            ano_count += 1
            print(idx, relations)
        continue
        src, dst = graph.edges()
        for i in range(relations):
            indicator = torch.where(edge_types == i)[0]
            g = dgl.graph((src[indicator], dst[indicator]), num_nodes=graph.num_nodes())
            adj = g.adjacency_matrix().to_dense()
            adj_list.append(adj.flatten().numpy())
            data[i] = adj.flatten().numpy().tolist()
        df = pd.DataFrame(data)
        df.to_csv('RGCN/shuffled_dataset/relation_adj_' + str(idx) + '.csv', index=False)
    print(ano_count)
    return


def prepare_data(batched_graph, labels, relations=3):
    adj_list = []
    adj, features = batched_graph.adjacency_matrix().to(batched_graph.device).float(), batched_graph.ndata['feat']
    num_nodes_per_graph, batch_matrix, start_idx = batched_graph.batch_num_nodes(), \
        torch.zeros((labels.shape[0], batched_graph.num_nodes())), 0
    for i, num_nodes in enumerate(num_nodes_per_graph):
        batch_matrix[i, start_idx:start_idx + num_nodes] = 1.
        start_idx += num_nodes
    batch_matrix = batch_matrix / batch_matrix.sum(dim=1, keepdim=True)
    batch_matrix = batch_matrix.to(batched_graph.device)
    # batch_matrix = []
    # # Load the CSV file
    # df = pd.read_csv('RGCN/shuffled_dataset/relation_adj_' + str(idx) + '.csv')
    # for i in range(relations):
    #     adj_flat = torch.from_numpy(df[str(i)].values)
    #     num_elements = adj_flat.size()
    #     n = int(torch.sqrt(torch.tensor(num_elements)))
    #     adj = adj_flat.reshape((n, n)).to_sparse().float()
    #     adj_list.append(adj.to(batched_graph.device))
    return adj, adj_list, features, batch_matrix


def adj2dgl(masked_adj, graph, mask_thresh=0.25):
    adj = torch.zeros((graph.num_nodes(), graph.num_nodes()), dtype=torch.float32).to(masked_adj.device)
    src, dst = graph.edges()
    adj[src, dst] = 1.0

    masked_adj = torch.round(masked_adj)
    masked_src, masked_dst = torch.nonzero(masked_adj, as_tuple=True)
    masked_graph = dgl.graph((masked_src, masked_dst), num_nodes=graph.num_nodes())
    diff = (adj - masked_adj).detach().numpy()
    residual_src, residual_dst = torch.nonzero(torch.round(adj - masked_adj), as_tuple=True)
    residual_graph = dgl.graph((residual_src, residual_dst), num_nodes=graph.num_nodes())

    if graph.ndata:
        for key in graph.ndata:
            masked_graph.ndata[key] = graph.ndata[key]
            residual_graph.ndata[key] = graph.ndata[key]

    if graph.edata:
        for key in graph.edata:
            masked_graph.edata[key] = torch.ones(masked_graph.num_edges(), dtype=graph.edata[key].dtype)
            residual_graph.edata[key] = torch.ones(residual_graph.num_edges(), dtype=graph.edata[key].dtype)

    masked_nodes = set(masked_src.tolist()) | set(masked_dst.tolist())
    residual_nodes = set(residual_src.tolist()) | set(residual_dst.tolist())
    masked_nodes, residual_nodes = torch.tensor(list(masked_nodes)), torch.tensor(list(residual_nodes))
    if len(masked_nodes.tolist()) != 0:
        masked_graph, residual_graph = dgl.node_subgraph(graph, masked_nodes), dgl.node_subgraph(graph, residual_nodes)

    # check connected components in masked adj
    # nx_graph = masked_graph.to_networkx().to_undirected()
    # connected_components = list(nx.connected_components(nx_graph))
    # print(f"The graph is not connected. It has {len(connected_components)} components.")
    # print("Sizes of each component:", [len(comp) for comp in connected_components])

    return masked_graph, residual_graph

# trian and test functions for PMLP_RGCN (PMLP_GCN that using 3 adj matrix )
# def train(model, dataset, train_set, val_set):
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     criterion = nn.CrossEntropyLoss()
#
#     # Train the model
#     best_val_loss = float('inf')
#     for epoch in range(EPOCH):
#         model.train()
#         num_tests, num_correct, epoch_loss = 0, 0, 0
#         for idx in tqdm(list(train_set)):
#             batched_graph, labels = dataset[idx]
#             batched_graph, labels = batched_graph.to(device), labels.to(device)
#             adj, adj_list, features, batch_matrix = prepare_data(idx, batched_graph, labels)
#             optimizer.zero_grad()
#             pred = model(features, adj_list, use_mp=True)
#             if len(batch_matrix)!=0: pred = batch_matrix @ pred
#             else: pred = torch.mean(pred, dim=0)
#             loss = criterion(pred, labels)
#             epoch_loss += loss.item()
#             if len(batch_matrix)!=0:
#                 num_correct += (pred.argmax(1) == labels).sum().item()
#                 num_tests += len(labels)
#             else:
#                 num_correct += (pred.argmax() == labels).sum().item()
#                 num_tests+=1
#             loss.backward()
#             optimizer.step()
#         train_accuracy = num_correct / num_tests
#         train_loss = epoch_loss / len(train_dataloader)
#
#         # Validate the model
#         if epoch % 10 == 0:
#             model.eval()
#             val_loss = 0
#             with torch.no_grad():
#                 for idx in val_set:
#                     batched_graph, labels = dataset[idx]
#                     batched_graph, labels = batched_graph.to(device), labels.to(device)
#                     adj, adj_list, features, batch_matrix = prepare_data(idx, batched_graph, labels)
#                     pred = model(features, adj_list, use_mp=True)
#                     pred = batch_matrix @ pred
#                     if len(batch_matrix) != 0:
#                         num_correct += (pred.argmax(1) == labels).sum().item()
#                         num_tests += len(labels)
#                     else:
#                         num_correct += (pred.argmax() == labels).sum().item()
#                         num_tests += 1
#                     val_loss += criterion(pred, labels).item()
#             val_accuracy = num_correct / num_tests
#             val_loss /= len(val_dataloader)
#
#             # Save the best model for this fold
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model_state = model.state_dict()
#                 torch.save(best_model_state, 'trained_models/best_model_rgcn_'+str(epoch)+'.pt')
#             print(f"Epoch {epoch + 1}: Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
#         print(
#             f"Epoch {epoch + 1}: Train loss: {train_loss}, Train accuracy: {train_accuracy}, num_correct: {num_correct}, num_tests: {num_tests}")
#
#     return
#
#
# def test(model, dataset, test_set):
#     # Test the model
#     model = model.to(device)
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for idx in test_set:
#             batched_graph, labels = dataset[idx]
#             batched_graph, labels = batched_graph.to(device), labels.to(device)
#             adj, adj_list, features, batch_matrix = prepare_data(idx, batched_graph, labels)
#             pred = model(features, adj, use_mp=True)
#             if len(batch_matrix)!=0:
#                 pred = batch_matrix @ pred
#                 all_preds.extend(pred.argmax(1).tolist())
#                 all_labels.extend(labels.tolist())
#             else:
#                 pred = torch.mean(pred, dim=0)
#                 all_preds.append(pred.argmax().item())
#                 all_labels.extend(labels.item())
#
#     # Calculate metrics for this fold
#     # accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
#     recall = recall_score(all_labels, all_preds, average='binary')
#     f1 = f1_score(all_labels, all_preds, average='binary')
#     return precision, recall, f1