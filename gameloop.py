import networkx as nx
import numpy as np
import copy
from agent import Agent, euclidean
import matplotlib.pyplot as plt

class GameLoop():
    def __init__(self, graph=None, num_agents=2, start=None):
        self.graph = graph
        self.num_agents = num_agents
        if (start is None):
            self.start = list(self.graph.nodes)[0] #TODO just choosing the first node for now
        else:
            self.start = start

        self.init_agents()
        self.reset()

        # self.fig = plt.figure()
        # plt.ion()
        # for point in self.graph.nodes(data='pos'):
        #     plt.scatter(point[1][0], point[1][1])

        # plt.draw()
        # plt.grid(True)
        # plt.show(block=False)
        # plt.pause(0.001)

    def loop(self):
        while not all(self.done_tasks.values()):
            # print('gameloop')
            for agent in self.agents:
                agent.step()
                #TODO: update done tasks list and propogate for all agents
                for key in self.done_tasks:
                    self.done_tasks[key] = self.done_tasks[key] or agent.get_done_tasks()[key]
                    agent.update_done_tasks(self.done_tasks)

    def reset(self):
        self.done_tasks = {x: False for x in self.graph.nodes}
        del self.done_tasks[self.start]
        for agent in self.agents:
            agent.reset()

    def init_agents(self):
        self.agents = []
        for i in range(0, self.num_agents):
            nodeweights = {x: np.random.rand() for x in self.graph.nodes}
            del nodeweights[self.start]
            newagent = Agent(graph=self.graph, start=self.start, id=i, nodeweights=nodeweights)
            self.agents.append(newagent)