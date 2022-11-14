import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gameloop import GameLoop
from agent import euclidean
from allocator import Allocator
import params

G = nx.complete_graph(params.NUM_NODES)

# generate random node positions
nodepos = {}
for node in G.nodes:
    pos = 100*np.random.rand(params.NUM_AGENTS)
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
    mult = np.random.uniform(1, params.EDGE_MULT)
    G.edges[edge]['weight'] = G.edges[edge]['weight']*mult
    G.edges[edge]['mult'] = mult

labels = nx.get_node_attributes(G, 'pos')
# plt.figure()
# nx.draw(G, with_labels=True, labels=labels)
# plt.show()

# gameloop = GameLoop(graph=G, num_agents=params.NUM_AGENTS)
# gameloop.loop()

allocator = Allocator(graph=G, popsize=params.POPSIZE, num_agents=params.NUM_AGENTS)
allocator.allocate()