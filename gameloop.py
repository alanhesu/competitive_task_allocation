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

    def loop(self):
        while not all(self.done_tasks.values()):
            # print('gameloop')
            for agent in self.agents:
                agent.step()
                #TODO: update done tasks list and propogate for all agents
                for key in self.done_tasks:
                    self.done_tasks[key] = self.done_tasks[key] or agent.get_done_tasks()[key]
                    agent.update_done_tasks(self.done_tasks)
        self.plot_graph()
        print(self.total_cost())
        print(self.minmax())

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

    def plot_graph(self):
        plt.figure()
        # plot nodes
        nodes = nx.get_node_attributes(self.graph, 'pos')
        # nx.draw(self.graph, with_labels=True, labels=nodes)
        x = [x[0] for x in nodes.values()]
        y = [x[1] for x in nodes.values()]
        plt.scatter(x, y, s=100)
        # plot start node
        plt.scatter(nodes[self.start][0], nodes[self.start][1], s=100, color='r')

        # plot agent histories
        for agent in self.agents:
            x = [x[0] for x in agent.travel_hist]
            y = [x[1] for x in agent.travel_hist]
            plt.plot(x, y, label='agent {}'.format(agent.id))
        plt.legend()
        # plt.show()
        plt.savefig('graph.png')

    def total_cost(self):
        cost = 0
        for agent in self.agents:
            for i in range(1, len(agent.travel_hist)):
                cost += euclidean(agent.travel_hist[i], agent.travel_hist[i-1])
        return cost

    def minmax(self):
        max_cost = -np.inf
        for agent in self.agents:
            cost = 0
            for i in range(1, len(agent.travel_hist)):
                cost += euclidean(agent.travel_hist[i], agent.travel_hist[i-1])
            if (cost > max_cost):
                max_cost = cost
        return max_cost