import networkx as nx
import numpy as np
import copy
from agent import Agent, euclidean
from gameloop import GameLoop
import matplotlib.pyplot as plt
import params

class Allocator:
    def __init__(self, graph=None, popsize=params.POPSIZE, num_agents=params.NUM_AGENTS, metric=params.METRIC):
        self.graph = graph
        self.popsize = popsize
        self.num_agents = num_agents
        self.metric = metric
        # initialize the gameloop
        self.gameloops = []
        for i in range(0, self.popsize):
            gameloop = GameLoop(graph=self.graph, num_agents=self.num_agents, id=i)
            self.gameloops.append(gameloop)

        # initialie the GA class

    def allocate(self):
        # randomly initialize nodeweights
        nodeweights_pop = self.init_nodeweights()
        scores = np.zeros(self.popsize)

        # in a loop until convergence:
        for i in range(0, params.MAX_ITER):
            print('allocator iteration {}'.format(i))
            # run the game loop n number of times to get n matrices of nodeweights
            for i, gameloop in enumerate(self.gameloops):
                gameloop.set_nodeweights(nodeweights_pop[gameloop.id])
                gameloop.reset()
                gameloop.loop()

                # calculate score based on metric
                if (self.metric == 'total'):
                    scores[i] = gameloop.total_cost()
                elif (self.metric == 'minmax'):
                    scores[i] = gameloop.minmax()
                print(gameloop.total_cost())
                print(gameloop.minmax())

            # run GA to get new nodeweights
            print(list(nodeweights_pop.values())[0].shape)
            print(scores.shape)
            # inputs: dictionary of 2d np array of weights, 1d np array of scores

    def init_nodeweights(self):
        nodeweights_pop = {}
        for gameloop in self.gameloops:
            nodeweights_pop[gameloop.id] = np.random.rand(self.num_agents, len(list(self.graph.nodes)))
        return nodeweights_pop