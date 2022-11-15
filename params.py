# problem size
NUM_NODES = 5
NUM_AGENTS = 4
NUM_ELITE = 2
POPSIZE = 4

# obstacles
EDGE_MULT = 2 # randomly multiply edge weight by up to this amount

# agent params
EPSILON = 0 # epsilon-greedy
GAMMA = 1 # weight decay
WEIGHT_ALPHA = 0 # distance score weight
WEIGHT_BETA = 1 # node score weight
START_WEIGHT = .1 # initial nodeweight for returning to start. Set to 1 to make it random

# GA hyperparams
MAX_ITER = 1
METRIC = 'total' # total or minmax
INCOMPLETE_PENALTY = -1e10