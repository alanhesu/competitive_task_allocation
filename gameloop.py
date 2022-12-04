import networkx as nx
import numpy as np
import copy
from agent import Agent, euclidean
import matplotlib.pyplot as plt
import params

class GameLoop():
    def __init__(self,
                graph=None,
                num_agents=params.NUM_AGENTS,
                start=None,
                id=1,
                comm_dones=params.COMM_DONES,
                see_dones=params.SEE_DONES,
                see_intent=params.SEE_INTENT,
                comm_range=params.COMM_RANGE,
                phi=params.PHI,
                incomplete_penalty=params.INCOMPLETE_PENALTY,
                plot=False,
                agent_kwargs=None):
        self.graph = graph
        self.num_agents = num_agents
        self.id = id
        self.plot = plot
        self.agent_kwargs = agent_kwargs

        self.comm_dones = comm_dones
        self.see_dones = see_dones
        self.see_intent = see_intent
        self.comm_range = comm_range
        self.phi = phi
        self.incomplete_penalty = incomplete_penalty

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

                for other_agent in self.agents:
                    # loop through other agents and communicate if they're in range
                    if (other_agent.id == agent.id):
                        continue # don't communicate with yourself
                    if (self.comm_dones and euclidean(other_agent.position, agent.position) <= self.comm_range):
                        agent.update_done_tasks(other_agent.done_tasks)
                    if (self.see_intent):
                        if (euclidean(other_agent.position, agent.position) <= self.comm_range):
                            agent.intents[other_agent] = other_agent.goal
                        else:
                            agent.intents[other_agent] = None # forget the other agent's intent if you're out of range

                # if (self.comm_dones):
                #     for key in self.done_tasks:
                #         self.done_tasks[key] = self.done_tasks[key] or agent.get_done_tasks()[key]
                #         agent.update_done_tasks(self.done_tasks)

                # if (self.see_intent):
                #     self.intents[agent] = agent.goal

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
            agent.intents = copy.copy(self.intents)

    def init_agents(self):
        self.agents = []
        for i in range(0, self.num_agents):
            nodeweights = {x: np.random.rand() for x in self.graph.nodes}
            # del nodeweights[self.start]
            newagent = Agent(graph=self.graph, start=self.start, id=i, nodeweights=nodeweights, see_dones=self.see_dones, see_intent=self.see_intent, **self.agent_kwargs)
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
        score = self.phi*self.minmax() + (1 - self.phi)*self.total_cost()
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
        cost += self.incomplete_penalty*len([x for x in self.done_tasks.values() if not x])
        return cost

    def minmax(self):
        max_cost = -np.inf
        for agent in self.agents:
            cost = len(agent.travel_hist)
            if (cost > max_cost):
                max_cost = cost
        self.update_global_done_tasks()
        max_cost += self.incomplete_penalty*len([x for x in self.done_tasks.values() if not x])
        return max_cost