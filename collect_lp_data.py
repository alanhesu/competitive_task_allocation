import glob
import pandas as pd
import os
import numpy as np

from vrp_lp import solve_VRP

def collect_lp_data(agents):
    metrics_dict = {
        'score': {},
        'totalcost': {},
        'minmax': {},
        'elapsed': {},
    }

    graphs = glob.glob('graphs/*.json')

    for num_agents in agents:
        for fname in graphs:
            print('solving {}agents_{}'.format(num_agents, fname))
            try:
                score, totalcost, minmax, elapsed = solve_VRP(fname, num_agents)
            except:
                score = None
                totalcost = None
                minmax = None
                elapsed = None

            basename = os.path.basename(fname)
            ind = basename.index('nodes')
            num_nodes = int(basename[0:ind])
            config_key = (num_nodes, num_agents)
            if (config_key not in metrics_dict['score']):
                metrics_dict['score'][config_key] = [score]
                metrics_dict['totalcost'][config_key] = [totalcost]
                metrics_dict['minmax'][config_key] = [minmax]
                metrics_dict['elapsed'][config_key] = [elapsed]
            else:
                metrics_dict['score'][config_key].append(score)
                metrics_dict['totalcost'][config_key].append(totalcost)
                metrics_dict['minmax'][config_key].append(minmax)
                metrics_dict['elapsed'][config_key].append(elapsed)

    for key in metrics_dict:
        for config_key in metrics_dict[key]:
            metrics_dict[key][config_key] = np.mean(metrics_dict[key][config_key])

    new_metrics_dict = {}
    for key in metrics_dict:
        for config in metrics_dict[key]:
            new_metrics_dict['{} {}'.format(key, config)] = metrics_dict[key][config]

    df = pd.DataFrame([new_metrics_dict])
    df.to_csv('lp_soln.csv')

if __name__ == '__main__':
    collect_lp_data([2, 5])