import networkx as nx
import numpy as np
import copy
from agent import Agent, euclidean
import matplotlib.pyplot as plt
import params

class GameLoop():
    def __init__(self, graph=None, num_agents=params.NUM_AGENTS, start=None, id=1,
                global_dones=params.GLOBAL_DONES, see_dones=params.SEE_DONES, see_intent=params.SEE_INTENT, plot=False):
        self.graph = graph
        self.num_agents = num_agents
        self.id = id
        self.plot = plot

        self.global_dones = global_dones
        self.see_dones = see_dones
        self.see_intent = see_intent

        if (start is None):
            self.start = list(self.graph.nodes)[0] #TODO just choosing the first node for now
        else:
            self.start = start

        self.init_agents()
        self.reset()

    def loop(self):
        while not all(self.agent_dones.values()):
            for agent in self.agents:
                if (agent.done):
                    continue
                agent.step()

                if (self.global_dones):
                    for key in self.done_tasks:
                        self.done_tasks[key] = self.done_tasks[key] or agent.get_done_tasks()[key]
                        agent.update_done_tasks(self.done_tasks)

                if (self.see_intent):
                    self.intents[agent] = agent.goal

                self.agent_dones[agent.id] = agent.done

        if (self.plot):
            self.plot_graph()

    def reset(self):
        self.done_tasks = {x: False for x in self.graph.nodes if x != self.start}
        self.intents = {x: None for x in self.agents}
        # del self.done_tasks[self.start]
        self.agent_dones = {}
        for agent in self.agents:
            agent.reset()
            self.agent_dones[agent.id] = agent.done
            agent.intents = self.intents

    def init_agents(self):
        self.agents = []
        for i in range(0, self.num_agents):
            nodeweights = {x: np.random.rand() for x in self.graph.nodes}
            # del nodeweights[self.start]
            newagent = Agent(graph=self.graph, start=self.start, id=i, nodeweights=nodeweights, see_dones=self.see_dones, see_intent = self.see_intent)
            self.agents.append(newagent)

    def set_nodeweights(self, nodeweights_arr):
        # nodeweights_arr is a 2d np array
        for r, agent in enumerate(self.agents):
            for c, node in enumerate(agent.nodeweights_base.keys()):
                agent.nodeweights_base[node] = nodeweights_arr[r,c]

    def update_global_done_tasks(self):
        for agent in self.agents:
            for key in self.done_tasks:
                self.done_tasks[key] = self.done_tasks[key] or agent.get_done_tasks()[key]

    def plot_graph(self, fname=None):
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
            plt.scatter(x, y, label='agent {}'.format(agent.id), s=1)
        plt.legend()
        # plt.show()
        score = params.PHI*self.minmax() + (1 - params.PHI)*self.total_cost()
        plt.title('score = {}'.format(score))
        if (fname is None):
            plt.savefig('graph_{}.png'.format(self.id))
        else:
            plt.savefig(fname)
        plt.close()

    def total_cost(self):
        cost = 0
        for agent in self.agents:
            cost += len(agent.travel_hist)
        self.update_global_done_tasks()
        cost += params.INCOMPLETE_PENALTY*len([x for x in self.done_tasks.values() if not x])
        # if (not all(self.done_tasks.values())):
        #     cost += params.INCOMPLETE_PENALTY
        return cost

    def minmax(self):
        max_cost = -np.inf
        for agent in self.agents:
            cost = len(agent.travel_hist)
            if (cost > max_cost):
                max_cost = cost
        self.update_global_done_tasks()
        max_cost += params.INCOMPLETE_PENALTY*len([x for x in self.done_tasks.values() if not x])
        # if (not all(self.done_tasks.values())):
        #     max_cost += params.INCOMPLETE_PENALTY
        return max_cost