import networkx as nx
import numpy as np
from enum import Enum
import copy
import params

class States(Enum):
    IDLE = 1
    MOVING = 2
    DOING_TASK = 3
    HOMING = 4

class Agent:
    def __init__(self,
                graph=None,
                start=None,
                id=1,
                obs=[],
                speed=1,
                eps=params.EPSILON,
                nodeweights=None,
                gamma=params.GAMMA,
                alpha=params.WEIGHT_ALPHA,
                beta=params.WEIGHT_BETA,
                see_dones=params.SEE_DONES,
                see_intent=params.SEE_INTENT):
        self.graph = graph # a networkx graph
        self.start = start # starting node in the graph
        self.id = id
        self.obs = obs
        self.speed = speed
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.see_dones = see_dones
        self.see_intent = see_intent
        self.nodeweights_base = nodeweights # a dict of node:weight pairs

        # calculate the max distance as the diagonal of the graph bounding box so we can scale
        # the distance for the score calculation
        self.max_dist = self.calc_max_dist()
        self.reset()

    def step(self):
        # print(self.position)
        #TODO should we be allowed to change direction, or only do these checks in the idle state?
        # if (self.goal is not None):
        #     print(self.id, self.goal, self.state, self.position, self.graph.nodes[self.goal]['pos'], self.done_tasks)
        if (all(self.done_tasks.values())):
            self.goal = self.start
            self.state = States.MOVING
            # print('{} moving towards {} {}'.format(self.id, self.goal, self.graph.nodes[self.goal]['pos']))

        if (self.see_dones):
            delete = [key for key in self.nodeweights if key in self.done_tasks and self.done_tasks[key]]
            # remove from nodeweights if the task has already been done
            for key in delete:
                del self.nodeweights[key]
                if (self.goal == key):
                    self.state = States.IDLE

        if (self.see_intent):
            for agent in self.intents:
                other_goal = self.intents[agent]
                if (agent.id != self.id
                    and other_goal is not None
                    and other_goal == self.goal
                    and other_goal != self.start
                    and other_goal in self.nodeweights):
                    # if we're farther from the goal by euclidean distance, go somewhere else by setting this as off limits temporarily
                    other_dist = euclidean(agent.position, self.graph.nodes[other_goal]['pos'])
                    self_dist = euclidean(self.position, self.graph.nodes[self.goal]['pos'])
                    if (self_dist > other_dist):
                        # print('{} moving away from {}'.format(self.id, self.goal))
                        self.state = States.IDLE

                        # actually, just mark it as done for this agent
                        del self.nodeweights[other_goal]
                        break

        if (self.state == States.IDLE):
            # decide where to go
            num = np.random.rand()
            weights = self.calc_nodeweights()
            if (num < self.eps):
                self.goal = np.random.choice(list(weights.keys()))
            else:
                goal_ind = np.argmax(list(weights.values()))
                self.goal = list(weights.keys())[goal_ind]
            self.state = States.MOVING

            # print('{} moving towards {} {}'.format(self.id, self.goal, self.graph.nodes[self.goal]['pos']))

        elif (self.state == States.MOVING):
            # move towards goal at a speed defined by self.speed
            if (np.allclose(self.position, self.graph.nodes[self.goal]['pos'])):
                # we're already here
                dX = np.zeros(2)
            else:
                dX = self.move_pretend()
            dist = euclidean(self.position, self.graph.nodes[self.goal]['pos'])
            if (np.linalg.norm(dX) >= dist or np.linalg.norm(dX) == 0): # we travel far enough to reach the goal
                # print('{} at {}'.format(self.id, self.goal))
                self.prev_node = self.goal
                self.position = self.graph.nodes[self.goal]['pos']
                if (self.goal == self.start):
                    # at home
                    self.state = States.IDLE
                elif (self.done_tasks[self.goal]):
                    # the task is already done, so we need to find another one
                    del self.nodeweights[self.goal]
                    self.state = States.IDLE
                else:
                    # start doing the task
                    self.done_tasks[self.goal] = True
                    del self.nodeweights[self.goal]
                    self.state = States.DOING_TASK
            else:
                self.position = self.position + dX

        elif (self.state == States.DOING_TASK):
            # TODO: wait for some time while doing the task
            self.state = States.IDLE

        self.travel_hist.append(self.position)
        self.time += 1

        # check the done condition
        if (np.allclose(self.position, self.graph.nodes[self.start]['pos'])
            and self.goal == self.start
            and self.time > 1):
            self.done = True

    def reset(self):
        self.nodeweights = copy.deepcopy(self.nodeweights_base)
        self.done_tasks = {x: False for x in self.nodeweights_base.keys() if x != self.start}
        self.state = States.IDLE
        self.goal = None
        self.position = self.graph.nodes[self.start]['pos']
        self.prev_node = self.start
        self.time = 0
        self.travel_hist = [] # a list of where it's been
        self.done = False

    def update_done_tasks(self, task_info):
        # set agent's done task list to union of self.done_tasks and task_info
        for key in self.done_tasks:
            self.done_tasks[key] = self.done_tasks[key] or task_info[key]

    def get_done_tasks(self):
        # return agent's done task list
        return self.done_tasks

    def move_pretend(self):
        # move along the edge at some speed
        # pretend there's obstacles by decreasing the movement distance by a factor based on
        # edgeweight/distance(pos, goal)
        # return [dx, dy]
        vector = (self.graph.nodes[self.goal]['pos'] - self.position)/np.linalg.norm(self.graph.nodes[self.goal]['pos'] - self.position)
        edge = tuple(sorted([self.goal, self.prev_node]))
        if (edge[0] == 0 and edge[1] == 0):
            edge = (0, 1) # hack a fix to an edge case
        edgemult = self.graph.edges[edge]['mult']

        dX = self.speed*vector/edgemult
        return dX

    def calc_nodeweights(self):
        weights = copy.deepcopy(self.nodeweights)
        for key in weights:
            weight = weights[key]
            dist = euclidean(self.position, self.graph.nodes[key]['pos'])
            # scale the distance score
            scaled_dist = (self.max_dist - dist)/self.max_dist

            # don't decay the start node weight
            if (key == self.start):
                weight = self.alpha*scaled_dist + self.beta*weight
            else:
                weight = self.alpha*scaled_dist + self.beta*self.gamma**self.time*weight
            weights[key] = weight

        return weights

    def calc_max_dist(self):
        nodes = nx.get_node_attributes(self.graph, 'pos')
        x = [x[0] for x in nodes.values()]
        y = [x[1] for x in nodes.values()]
        minx = np.min(x)
        maxx = np.max(x)
        miny = np.min(y)
        maxy = np.max(y)
        return euclidean(np.array([minx, miny]), np.array([maxx, maxy]))

def euclidean(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))