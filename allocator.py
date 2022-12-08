import networkx as nx
import numpy as np
import copy
from random import choice, choices, random, sample, randrange, randint
import sys
import time

from agent import Agent, euclidean
from gameloop import GameLoop
import matplotlib.pyplot as plt
import params

class Allocator:
    def __init__(self,
                graph=None,
                popsize=params.POPSIZE,
                num_agents=params.NUM_AGENTS,
                num_parent=params.NUM_PARENT,
                num_elite=params.NUM_ELITE,
                phi=params.PHI,
                max_iter=params.MAX_ITER,
                operator_threshold=params.OPERATOR_THRESHOLD,
                adaptive_var_threshold=params.ADAPTIVE_VAR_THRESHOLD,
                operator_step_size = params.OPERATOR_STEP_SIZE,
                start_weight=params.START_WEIGHT,
                mutation_rate=params.MUTATION_RATE,
                crossover_function=params.CROSSOVER_FUNCTION,
                mutation_function=params.MUTATION_FUNCTION,
                max_quiescence=params.MAX_QUIESCENCE,
                gameloop_kwargs=None,
                agent_kwargs=None):
        self.graph = graph
        self.popsize = popsize
        self.num_agents = num_agents
        self.num_parent = num_parent
        self.num_elite = num_elite
        self.phi = phi
        self.max_iter = max_iter
        self.operator_threshold = operator_threshold
        self.adaptive_var_threshold = adaptive_var_threshold
        self.operator_step_size = operator_step_size
        self.start_weight = start_weight
        self.mutation_rate = mutation_rate
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.max_quiescence = max_quiescence
        self.gameloop_kwargs = gameloop_kwargs
        self.agent_kwargs = agent_kwargs

        # initialize the gameloop
        self.gameloops = []
        for i in range(0, self.popsize):
            gameloop = GameLoop(graph=self.graph, num_agents=self.num_agents, id=i, agent_kwargs=agent_kwargs, **gameloop_kwargs)
            self.gameloops.append(gameloop)

        # GA Convergence
        self.successive_iter = 0

        # initialize data logging
        self.scores_hist = []

    def allocate(self):
        # randomly initialize nodeweights
        nodeweights_pop = self.init_nodeweights()
        scores = np.zeros(self.popsize)
        recent_scores = []
        starttime = time.time()

        # in a loop until convergence:
        for iter in range(0, self.max_iter):
            print('allocator iteration {}'.format(iter))
            # run the game loop n number of times to get n matrices of nodeweights
            for i, gameloop in enumerate(self.gameloops):
                gameloop.set_nodeweights(nodeweights_pop[gameloop.id])
                gameloop.reset()
                gameloop.loop()

                # calculate score based on metric
                scores[i] = self.phi*gameloop.minmax() + (1 - self.phi)*gameloop.total_cost()

                sys.stdout.flush()

            self.scores_hist.append(copy.deepcopy(scores))

            elites, avg_elite_score = self.selection_pair(nodeweights_pop,scores) # elites survive
            recent_scores.append(avg_elite_score)
            recent_scores, convergence = self.adaptive_convergence(recent_scores)
            if convergence:
                break

            new_population = copy.deepcopy(elites)[0:self.num_elite]
            while len(new_population) < self.popsize:
                operator = random()
                if operator > self.operator_threshold:  #   crossover
                    parent_a, parent_b = sample(elites, k=2)

                    childA, childB = self.crossover(parent_a, parent_b)

                    new_population.append(childA)
                    if len(new_population) < self.popsize:
                        new_population.append(childB)

                else:                                   #   mutation
                    parent = choice(elites)
                    child = self.mutation(parent)
                    new_population.append(child)

            for i, key in enumerate(nodeweights_pop):
                nodeweights_pop[key] = new_population[i]

        elapsed = time.time() - starttime
        self.plot_data()

        # get some metrics
        ind = np.argmin(self.scores_hist[-1])
        best_total = self.gameloops[ind].total_cost()
        best_minmax = self.gameloops[ind].minmax()

        return np.min(self.scores_hist[-1]), best_total, best_minmax, elapsed

    def init_nodeweights(self):
        nodeweights_pop = {}
        interval = int(len(list(self.graph.nodes))/self.num_agents)
        for gameloop in self.gameloops:
            nodeweights_pop[gameloop.id] = np.random.rand(self.num_agents, len(list(self.graph.nodes)))
            if (self.start_weight != 1):
                nodeweights_pop[gameloop.id][:,gameloop.start] = self.start_weight
        return nodeweights_pop

    def get_mean_score_hist(self):
        scores_hist = np.stack(self.scores_hist, axis=0)
        return np.mean(scores_hist, axis=1)

    def get_min_score_hist(self):
        scores_hist = np.stack(self.scores_hist, axis=0)
        return np.min(scores_hist, axis=1)

    def plot_data(self, fname=None):
        plt.figure()
        plt.plot(self.get_mean_score_hist(), label='average')
        plt.plot(self.get_min_score_hist(), label='min')
        plt.legend(loc='best')
        if (fname is None):
            plt.savefig('score_hist.png')
        else:
            plt.savefig(fname)
        plt.close()

    # select agents that survive to next generation based on fitness, number based on num_elite parameter
    # Returns: list of surviving agents, average elite score
    def selection_pair(self, pop_weights, scores):
        ranked_scores = [sorted(scores).index(x) for x in scores]
        elite_agents = []
        min_scores = []
        for i in range(self.num_parent):
            elite_idx = ranked_scores.index(min(ranked_scores))
            if i < self.num_elite:
                min_scores.append(scores[elite_idx])
            ranked_scores[elite_idx] = np.inf
            elite_agents.append(pop_weights[elite_idx])

        avg_elite_score = sum(min_scores[0:self.num_elite])/self.num_elite

        return elite_agents, avg_elite_score

    # TODO clean this up
    def adaptive_convergence(self, recent_scores):
        convergence = False
        if len(recent_scores) < self.max_quiescence + 1:
            return recent_scores, convergence

        recent_scores.pop(0)
        curr_variance = (np.max(recent_scores) - np.min(recent_scores))/np.min(recent_scores)

        if curr_variance < self.adaptive_var_threshold:
            self.operator_threshold += self.operator_step_size
            if self.operator_threshold >= 1.0:
                self.operator_threshold = 1.0

        if curr_variance == 0:
            self.successive_iter += 1
        else:
            self.successive_iter = 0
            self.operator_threshold = params.OPERATOR_THRESHOLD

        if self.successive_iter == self.max_quiescence:
            convergence = True
            self.successive_iter = 0
        # print("\n\nMUTATION RATE", self.operator_threshold)
        # print(recent_scores)
        # print(self.successive_iter)
        return recent_scores, convergence


## Crossover Functions
    def crossover(self, parentA, parentB):
        crossover_selection = ['SINGLE','TWO','UNIFORM']
        choice = ''
        # if self.crossover_function == 'SINGLE':
        #     return self.crossover_single(parentA, parentB)
        # elif self.crossover_function == "TWO":
        #     return self.crossover_two_point(parentA, parentB)
        # elif self.crossover_function == "UNIFORM":
        #     return self.crossover_uniform(parentA, parentB)
        if self.crossover_function == 'SINGLE':
            choice = choices(crossover_selection, cum_weights=(50, 25, 25), k=1)[0]
        elif self.crossover_function == "TWO":
            choice = choices(crossover_selection, cum_weights=(25, 50, 25), k=1)[0]
        elif self.crossover_function == "UNIFORM":
            choice = choices(crossover_selection, cum_weights=(25, 25, 50), k=1)[0]
        else: #MIXED
            rand_select = np.random.rand()
            if rand_select < (1/3):
                return self.crossover_single(parentA, parentB)
            elif rand_select > (2/3):
                return self.crossover_two_point(parentA, parentB)
            else:
                return self.crossover_uniform(parentA, parentB)

        if choice == 'SINGLE':
            return self.crossover_single(parentA, parentB)
        elif choice == 'TWO':
            return self.crossover_two_point(parentA, parentB)
        else:
            return self.crossover_uniform(parentA, parentB)

    def crossover_single(self, parentA, parentB):
        len = parentA.shape
        childA = np.empty(len)
        childB = np.empty(len)
        split_ind = np.random.randint(1, len[1]-1)

        childA[:,0:split_ind] = parentA[:,0:split_ind]
        childA[:,split_ind:] = parentB[:,split_ind:]

        childB[:,0:split_ind] = parentB[:,0:split_ind]
        childB[:,split_ind:] = parentA[:, split_ind:]

        return childA, childB

    def crossover_two_point(self, parentA, parentB):
        len = parentA.shape
        childA = np.empty(len)
        childB = np.empty(len)
        left_pt = np.random.randint(1, len[1]-1)
        right_pt = np.random.randint(left_pt, len[1])

        childA = np.hstack((parentA[:,0:left_pt], parentB[:,left_pt:right_pt], parentA[:,right_pt:]))
        childB = np.hstack((parentB[:,0:left_pt], parentA[:,left_pt:right_pt], parentB[:,right_pt:]))

        return childA, childB

    def crossover_uniform(self, parentA, parentB):
        len = parentA.shape
        bit_array = np.random.choice([0, 1], len)
        childA = np.where(bit_array, parentA, parentB)
        childB = np.where(1-bit_array, parentA, parentB)

        return childA, childB

## Mutation Functions
    def mutation(self, parent):
        mutation_selection = ['RESET', 'SWAP', 'INVERSION']
        choice = ''
        # if self.mutation_function == 'RESET':
        #     return self.mutation_random_reset(parent)
        # elif self.mutation_function == "SWAP":
        #     return self.mutation_swap(parent)
        # elif self.mutation_function == "INVERSION":
        #     return self.mutation_inversion(parent)
        if self.mutation_function == 'RESET':
            choice = choices(mutation_selection, cum_weights=(50, 25, 25), k=1)[0]
        elif self.mutation_function == "SWAP":
            choice = choices(mutation_selection, cum_weights=(25, 50, 25), k=1)[0]
        elif self.mutation_function == "INVERSION":
            choice = choices(mutation_selection, cum_weights=(25, 25, 50), k=1)[0]
        else: #MIXED
            rand_select = np.random.rand()
            if rand_select < (1/3):
                return self.mutation_random_reset(parent)
            elif rand_select > (2/3):
                return self.mutation_swap(parent)
            else:
                return self.mutation_inversion(parent)

        if choice == 'REST':
            return self.mutation_random_reset(parent)
        elif choice == 'SWAP':
            return self.mutation_swap(parent)
        else:
            return self.mutation_inversion(parent)

    def mutation_random_reset(self, parent):
        def mutate(gene):
            temp = random()
            if temp < self.mutation_rate: # TODO this value will need to be scaled dynamically
                return random()
            else:
                return gene

        mute = np.vectorize(mutate)
        child = mute(parent)
        return child

    def mutation_swap(self, parent):
        inds = np.arange(0, parent.shape[1])
        for r in range(0, parent.shape[0]):
            ind_pair = np.random.choice(inds, size=2, replace=False)
            temp = parent[r,ind_pair[0]]
            parent[r,ind_pair[0]] = parent[r,ind_pair[1]]
            parent[r,ind_pair[1]] = temp

        return parent

    def mutation_inversion(self, parent):
        left_idx = randint(0, parent.shape[1]-2)
        right_idx = randint(left_idx+1, parent.shape[1]-1)
        for agent in range(parent.shape[0]):
            slice = parent[agent, left_idx:right_idx]
            parent[agent, left_idx:right_idx] = np.fliplr([slice])
        return parent