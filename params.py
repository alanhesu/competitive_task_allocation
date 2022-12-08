# problem size
NUM_NODES = 10
NUM_AGENTS = 3
NUM_PARENT = 10
NUM_ELITE = 4
POPSIZE = 20

# observability params
COMM_DONES = True # set to true: all agents update their done_tasks list from global information
SEE_DONES = True # set to true: an agent will not select a goal if it is already done, must have GLOBAL_DONES=True
SEE_INTENT = True # set to true: an agent will not select a goal if another is already moving towards it
COMM_RANGE = 1000 # set to a really large number to be global
#TODO: communication radius
#TODO: agents communicate with each other their done tasks/goals instead of globally

# obstacles
EDGE_MULT = 2 # randomly multiply edge weight by up to this amount

# agent params
EPSILON = 0 # epsilon-greedy
GAMMA = .999 # weight decay
WEIGHT_ALPHA = .2 # distance score weight
WEIGHT_BETA = 1 # node score weight
START_WEIGHT = .1 # initial nodeweight for returning to start. Set to 1 to make it random

# GA hyperparams
MAX_ITER = 40
OPERATOR_THRESHOLD = .6 # crossover for less than threshold, mutation for >=
ADAPTIVE_VAR_THRESHOLD = 0.1 # rate of improvement threshold for adaptive randomness
OPERATOR_STEP_SIZE = 0.05 # step size for increasing mutation rate over crossover
CROSSOVER_FUNCTION = 'SINGLE' # SINGLE, TWO, UNIFORM, MIXED
MUTATION_FUNCTION = 'SWAP'  # RESET, SWAP, INVERSION, MIXED
MUTATION_RATE = .3 # note this and operator threshold DO NOT NEED TO ADD TO 1
PHI = .9 # score = phi*minmax + (1-phi)*totalcost
INCOMPLETE_PENALTY = 3e3
