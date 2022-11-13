import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gameloop import GameLoop

G = nx.complete_graph(5)

nodepos = {}
for node in G.nodes:
    pos = 100*np.random.rand(2)
    nodepos[node] = pos

nx.set_node_attributes(G, nodepos, 'pos')

labels = nx.get_node_attributes(G, 'pos')
# plt.figure()
# nx.draw(G, with_labels=True, labels=labels)
# plt.show()

gameloop = GameLoop(graph=G, num_agents=2)
gameloop.loop()