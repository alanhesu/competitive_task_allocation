import networkx as nx
import numpy as np
import copy
from agent import Agent, euclidean

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

    def loop(self):
        while any(self.done_tasks.values()):
            print('gameloop')
            for agent in self.agents:
                agent.step()
                #TODO: update done tasks list and propogate for all agents

    def reset(self):
        self.done_tasks = {x: False for x in self.graph.nodes}
        for agent in self.agents:
            agent.reset()

    def init_agents(self):
        self.agents = []
        for i in range(0, self.num_agents):
            nodeweights = {np.random.rand() for x in self.graph.nodes}
            newagent = Agent(graph=self.graph, start=self.start, nodeweights=nodeweights)
            self.agents.append(newagent)