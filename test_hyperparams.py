import argparse
import time
import pandas as pd
import itertools
import os

from test import run_tests
import params

agent_kwargs = {
    'eps': params.EPSILON,
    'gamma': params.GAMMA,
    'alpha': params.WEIGHT_ALPHA,
    'beta': params.WEIGHT_BETA,
}

gameloop_kwargs = {
    'comm_dones': params.COMM_DONES,
    'see_dones': params.SEE_DONES,
    'see_intent': params.SEE_INTENT,
    'comm_range': params.COMM_RANGE,
    'phi': params.PHI,
    'incomplete_penalty': params.INCOMPLETE_PENALTY,
}

allocator_kwargs = {
    'popsize': params.POPSIZE,
    'num_parent': params.NUM_PARENT,
    'num_elite': params.NUM_ELITE,
    'phi': params.PHI,
    'max_iter': params.MAX_ITER,
    'operator_threshold': params.OPERATOR_THRESHOLD,
    'adaptive_var_threshold': params.ADAPTIVE_VAR_THRESHOLD,
    'operator_step_size': params.OPERATOR_STEP_SIZE,
    'start_weight': params.START_WEIGHT,
    'mutation_rate': params.MUTATION_RATE,
    'crossover_function': params.CROSSOVER_FUNCTION,
    'mutation_function': params.MUTATION_FUNCTION,
    'max_quiescence': params.MAX_QUIESCENCE,
}

def test_hyperparams(testname, params):
    # get all hyperparameter combinations
    keys, values = zip(*params.items())
    params_perm = [dict(zip(keys, v)) for v in itertools.product(*values)]
    param_df = pd.DataFrame(params_perm)

    # add columns for metrics
    headers = ['score (5, 2)', 'score (10, 2)', 'score (20, 2)', 'score (5, 5)', 'score (10, 5)', 'score (20, 5)',
                'totalcost (5, 2)', 'totalcost (10, 2)', 'totalcost (20, 2)', 'totalcost (5, 5)', 'totalcost (10, 5)', 'totalcost (20, 5)',
                'minmax (5, 2)', 'minmax (10, 2)', 'minmax (20, 2)', 'minmax (5, 5)', 'minmax (10, 5)', 'minmax (20, 5)',
                'elapsed (5, 2)', 'elapsed (10, 2)', 'elapsed (20, 2)', 'elapsed (5, 5)', 'elapsed (10, 5)', 'elapsed (20, 5)']
    param_df = pd.concat([pd.DataFrame(columns=['name']), param_df, pd.DataFrame(columns=headers)], axis=1)
    param_df.to_csv('{}.csv'.format(testname), index=False)

    for i in range(0, len(param_df)):
        print('Test {}/{}'.format(i, len(param_df)))
        param_df.at[i,'name'] = '{}_{:04d}'.format(testname, i)
        row = param_df.iloc[i]

        # dont run the test if num_elite>num_parent or num_parent>popsize
        if (row['num_elite'] > row['num_parent'] or row['num_parent'] > row['popsize']):
            print('Not running num_elite={}, num_parent={}, popsize={}'.format(row['num_elite'], row['num_parent'], row['popsize']))
            continue

        # populate kwargs
        for key in row.keys():
            if (key in agent_kwargs):
                agent_kwargs[key] = row[key]
            if (key in gameloop_kwargs):
                gameloop_kwargs[key] = row[key]
            if (key in allocator_kwargs):
                allocator_kwargs[key] = row[key]

        # get metrics
        metrics = run_tests([2, 5], '{}_{:04d}'.format(testname, i), all=True, agent_kwargs=agent_kwargs, gameloop_kwargs=gameloop_kwargs, allocator_kwargs=allocator_kwargs)
        for metric in metrics:
            for config_key in metrics[metric]:
                col = '{} {}'.format(metric, config_key)
                if (col in row):
                    param_df.at[i,col] = metrics[metric][config_key]

        param_df.to_csv('{}.csv'.format(testname), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testname', type=str, required=True)

    args = parser.parse_args()

    if (os.path.exists('{}.csv'.format(args.testname))):
        print('{}.csv already exists. Please provide a different testname'.format(args.testname))
        exit()

    params = {
        'eps': params.EPSILON, # dont change
        'gamma': params.GAMMA, # dont change
        'alpha': params.WEIGHT_ALPHA, # dont change
        'beta': params.WEIGHT_BETA, # dont change
        'comm_dones': False,
        'see_dones': False,
        'see_intent': False,
        'comm_range': [10000],
        'phi': params.PHI,  # dont change
        'incomplete_penalty': params.INCOMPLETE_PENALTY, # dont change
        'popsize': [20],#, 40, 80],
        'num_parent': [10],#, 20],
        'num_elite': [2],#, 5, 10],
        'max_iter': 20,
        'operator_threshold': [0.1,0.3,0.6,0.9], #, .6, 1],
        'adaptive_var_threshold': params.ADAPTIVE_VAR_THRESHOLD,
        'operator_step_size': [0.02,0.03,0.04,0.05], #[0, .03, .1],
        'start_weight': params.START_WEIGHT, # dont change
        'mutation_rate': params.MUTATION_RATE, #[.1, .3, .6],
        'crossover_function': 'MIXED',  # dont change
        'mutation_function': 'MIXED',  # dont change
        'max_quiescence': [3,5,8],
    }

    # so we dont have to wrap everthing in a list manually
    for key in params:
        if (not isinstance(params[key], list)):
            params[key] = [params[key]]

    test_hyperparams(args.testname, params)
