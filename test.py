import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from gameloop import GameLoop
from agent import euclidean
from allocator import Allocator
from graph_generator import generate_random_graph
import params

if __name__ == '__main__':
    G = nx.read_gpickle('randgraph.gpickle')
    # G = generate_random_graph(params.NUM_NODES, params.EDGE_MULT)
    # plt.figure()
    # nx.draw(G, with_labels=True, labels=labels)
    # plt.show()

    # gameloop = GameLoop(graph=G, num_agents=params.NUM_AGENTS)
    # gameloop.loop()

    allocator = Allocator(graph=G, popsize=params.POPSIZE, num_agents=params.NUM_AGENTS)
    allocator.allocate()