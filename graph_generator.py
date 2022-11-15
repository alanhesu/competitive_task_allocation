import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

nx.write_gpickle(generate_random_graph(params.NUM_NODES, params.EDGE_MULT), 'randgraph.gpickle')