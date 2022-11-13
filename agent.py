import networkx as nx
import numpy as np
from enum import Enum
import copy

class States(Enum):
    IDLE = 1
    MOVING = 2
    DOING_TASK = 3
    HOMING = 4

class Agent:
    def __init__(self, graph=None, start=None, id=1, obs=[], speed=1, eps=.2, nodeweights=None):
        self.graph = graph # a networkx graph
        self.start = start # starting node in the graph
        self.id = id
        self.obs = obs
        self.speed = speed
        self.eps = eps
        self.nodeweights_base = nodeweights # a dict of node:weight pairs

        self.nodes = dict(self.graph.nodes(data=True))
        self.reset()

    def step(self):
        # print(self.position)
        if (self.state == States.IDLE):
            # decide where to go
            num = np.random.rand()
            if (num < self.eps):
                self.goal = np.random.choice(list(self.nodeweights.keys()))
            else:
                goal_ind = np.argmax(list(self.nodeweights.values()))
                self.goal = list(self.nodeweights.keys())[goal_ind]
            self.state = States.MOVING

            print('{} moving towards {}'.format(self.id, self.goal))

        if (self.state == States.MOVING):
            # move towards goal at a speed defined by self.speed

            vector = (self.graph.nodes[self.goal]['pos'] - self.position)/np.linalg.norm(self.graph.nodes[self.goal]['pos'] - self.position)
            if (np.isnan(vector).any()):
                print('nan')
            dist = euclidean(self.position, self.graph.nodes[self.goal]['pos'])
            if (np.linalg.norm(self.speed*vector) >= dist):
                print('{} at {}'.format(self.id, self.goal))
                # we travel far enough to reach the goal
                self.position = self.graph.nodes[self.goal]['pos']
                if (self.done_tasks[self.goal]):
                    # the task is already done, so we need to find another one
                    del self.nodeweights[self.goal]
                    self.state = States.IDLE
                else:
                    # start doing the task
                    self.done_tasks[self.goal] = True
                    del self.nodeweights[self.goal]
                    self.state = States.DOING_TASK
            else:
                self.position = self.position + self.speed*vector

        elif (self.state == States.DOING_TASK):
            # TODO: wait for some time while doing the task
            self.state = States.IDLE

    def reset(self):
        self.nodeweights = copy.deepcopy(self.nodeweights_base)
        self.done_tasks = {x: False for x in self.nodeweights_base.keys()}
        self.state = States.IDLE
        self.goal = None
        self.position = self.graph.nodes[self.start]['pos']

    def update_done_tasks(self, task_info):
        # set agent's done task list to union of self.done_tasks and task_info
        for key in self.done_tasks:
            self.done_tasks[key] = self.done_tasks[key] or task_info[key]

    def get_done_tasks(self):
        # return agent's done task list
        return self.done_tasks

def euclidean(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))