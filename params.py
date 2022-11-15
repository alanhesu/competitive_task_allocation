# problem size
NUM_NODES = 5
NUM_AGENTS = 2
POPSIZE = 2

# obstacles
EDGE_MULT = 2 # randomly multiply edge weight by up to this amount

# agent params
EPSILON = 0 # epsilon-greedy
GAMMA = .99 # weight decay
WEIGHT_ALPHA = 1 # distance score weight
WEIGHT_BETA = 1 # node score weight

# GA hyperparams
MAX_ITER = 1
METRIC = 'total' # total or minmax