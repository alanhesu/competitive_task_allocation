import networkx as nx
import numpy as np
import copy
from random import choices, random, gauss, randrange, randint
from agent import Agent, euclidean
from gameloop import GameLoop
import matplotlib.pyplot as plt
import params

class Allocator:
    def __init__(self, graph=None, popsize=params.POPSIZE, num_agents=params.NUM_AGENTS, num_elite=params.NUM_ELITE, metric=params.METRIC):
        self.graph = graph
        self.popsize = popsize
        self.num_agents = num_agents
        self.num_elite = num_elite
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
        for iter in range(0, params.MAX_ITER):
            print('allocator iteration {}'.format(iter))
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
            #print(list(nodeweights_pop.values())[0]) # 1 game 2 agents 5 weight nodes each agent
            print(scores.shape)
            
            new_population  = self.selection_pair(nodeweights_pop,scores) # elites survive
            while len(new_population) <= params.POPSIZE:
                pass
            '''
            sort scores and the highest two scores (of games) are kept, others discarded 
            until rest of discarded games (len scores - 2) are filled, crossover the two games until only 1 empty game left
            for last game, mutation 
            '''
            # inputs: dictionary of 2d np array of weights, 1d np array of scores

    def init_nodeweights(self):
        nodeweights_pop = {}
        for gameloop in self.gameloops:
            nodeweights_pop[gameloop.id] = np.random.rand(self.num_agents, len(list(self.graph.nodes)))
        return nodeweights_pop

    # select agents that survive to next generation based on fitness, number based on num_elite parameter
    # explore: roulette, fittest half, random
    # Returns: list of surviving agents
    def selection_pair(self, pop_weights, scores):
        ranked_scores = [sorted(scores).index(x) for x in scores]
        elite_agents = []
        for _ in range(self.num_elite):
            elite_idx = ranked_scores.index(max(ranked_scores))
            ranked_scores[elite_idx] = -1 
            elite_agents.append(pop_weights[elite_idx])
    
        return elite_agents

    def single_point_crossover(self, node_a, node_b):
        length = len(node_a)
        if length < 2:
            return node_a, node_b

        p = randint(1, length - 1)
        return node_a[0:p] + node_b[p:], node_b[0:p] + node_a[p:]


    def mutation(self, population, probability = 0.5):
        for _ in range(1): #set mutation rate
            index = randrange(len(population))
            population[index] = population[index] if random() > probability else abs(population[index] - 1)
        return population


    