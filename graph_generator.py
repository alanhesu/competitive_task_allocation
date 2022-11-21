import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

from gameloop import GameLoop
from agent import euclidean
from allocator import Allocator
import params

def generate_random_graph(num_nodes, edge_mult):
    G = nx.complete_graph(num_nodes)

    # generate random node positions
    nodepos = {}
    for node in G.nodes:
        pos = 100*np.random.rand(2)
        nodepos[node] = pos

    nx.set_node_attributes(G, nodepos, 'pos')

    # set edge weights
    weights = {}
    for edge in G.edges:
        p1 = G.nodes[edge[0]]['pos']
        p2 = G.nodes[edge[1]]['pos']
        weight = euclidean(p1, p2)
        weights[edge] = weight

    nx.set_edge_attributes(G, weights, 'weight')

    # randomly increase edge weights to simulate obstacles
    for edge in G.edges:
        mult = np.random.uniform(1, edge_mult)
        G.edges[edge]['weight'] = G.edges[edge]['weight']*mult
        G.edges[edge]['mult'] = mult

    return G

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_graph(G, fname):
    G_dict = nx.node_link_data(G)
    json_object = json.dumps(G_dict, indent=4, cls=NumpyEncoder)
    with open(fname, 'w') as f:
        f.write(json_object)

def read_graph(fname):
    with open(fname, 'r') as f:
        dat = json.load(f)

    for node in dat['nodes']:
        node['pos'] = np.array(node['pos'])

    G = nx.node_link_graph(dat)

    return G

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--edgemult', type=float, required=True)
    parser.add_argument('--file', type=str, required=True)

    args = parser.parse_args()

    G = generate_random_graph(args.nodes, args.edgemult)
    write_graph(G, args.file)
