import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os
import datetime

from gameloop import GameLoop
from agent import euclidean
from allocator import Allocator
from graph_generator import read_graph
import params

def test_run(fname, num_agents, testname):
    G = read_graph(fname)
    allocator = Allocator(graph=G, num_agents=num_agents)
    score = allocator.allocate()

    # plot the first gameloop in the allocator
    plot_name = os.path.join(testname, '{}agents_{}.jpg'.format(num_agents, os.path.basename(fname)[0:-5]))
    allocator.gameloops[0].plot_graph(plot_name)
    plot_score_name = os.path.join(testname, 'scorehist_{}agents_{}.jpg'.format(num_agents, os.path.basename(fname)[0:-5]))
    allocator.plot_data(plot_score_name)

    return score

def test_files(files, agents, testname):
    scores_dict = {}
    for fname in files:
            for num_agents in agents:
                score = test_run(fname, num_agents, testname)
                print('file: {}, agents: {}, score: {}'.format(fname, num_agents, score))

                # get the number of nodes from the filename so we can keep track of scores
                basename = os.path.basename(fname)
                ind = basename.index('nodes')
                num_nodes = int(basename[0:ind])
                keystring = '{} nodes {} agents'.format(num_nodes, num_agents)
                if (keystring not in scores_dict):
                    scores_dict[keystring] = [score]
                else:
                    scores_dict[keystring].append(score)

    for keystring in scores_dict:
        scores_dict[keystring] = np.mean(scores_dict[keystring])

    return scores_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, nargs='+', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='run test on all graphs in graphs directory')
    group.add_argument('--file', type=str, help='txt file containing graphs to test')
    parser.add_argument('--testname', type=str, required=True)

    args = parser.parse_args()

    stamp = datetime.datetime.now()
    testname = args.testname + '_' + stamp.strftime('%m-%d_%H_%M_%S')
    os.mkdir(testname)

    if (args.file):
        with open(args.file, 'r') as f:
            lines = f.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i].strip()
        scores_dict = test_files(lines, args.agents, testname)
        print('average scores:', scores_dict)

    elif (args.all):
        files = glob.glob('graphs/*.json')
        files.sort()

        scores_dict = test_files(files, args.agents, testname)
        print('average scores:', scores_dict)