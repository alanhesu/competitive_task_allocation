# problem size
NUM_NODES = 10
NUM_AGENTS = 2
POPSIZE = 2

# observability params
GLOBAL_DONES = False # set to true: all agents update their done_tasks list from global information

# obstacles
EDGE_MULT = 2 # randomly multiply edge weight by up to this amount

# agent params
EPSILON = .2 # epsilon-greedy
GAMMA = 1 # weight decay
WEIGHT_ALPHA = 0 # distance score weight
WEIGHT_BETA = 1 # node score weight
START_WEIGHT = .1 # initial nodeweight for returning to start. Set to 1 to make it random

# GA hyperparams
MAX_ITER = 10
METRIC = 'total' # total or minmax
INCOMPLETE_PENALTY = -1e10