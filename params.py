# problem size
NUM_NODES = 5
NUM_AGENTS = 4
NUM_ELITE = 2
POPSIZE = 4

# obstacles
EDGE_MULT = 2 # randomly multiply edge weight by up to this amount

# agent params
EPSILON = 0 # epsilon-greedy
GAMMA = .99 # weight decay
WEIGHT_ALPHA = 1 # distance score weight
WEIGHT_BETA = 1 # node score weight

# GA hyperparams
MAX_ITER = 10
METRIC = 'total'