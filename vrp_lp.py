from pulp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import time
import networkx as nx

from graph_generator import read_graph

def solve_VRP(fname, num_agents, start_node=0, phi=.9):
    G = read_graph(fname)
    sites = list(G.nodes)
    position = []
    distance = []
    for node in sites:
        position.append({'node': node, 'pos': G.nodes[node]['pos']})
        distance_dict = {x: 0.0 for x in sites}
        for edge in G.edges(node):
            distance_dict[edge[1]] = G.edges[edge]['weight']
        distance_dict['node'] = node
        distance.append(distance_dict)

    position = pd.DataFrame(position).set_index('node')
    distance = pd.DataFrame(distance).set_index('node')

    positions = dict((node, position.loc[node, 'pos']) for node in sites)
    distances = dict(((s1, s2), distance.loc[s1, s2]) for s1 in positions for s2 in positions if s1 != s2)

    #create the problme
    prob=LpProblem("vrp", LpMinimize)

    #indicator variable if site i is connected to site j in the tour
    x = LpVariable.dicts('x',distances, 0,1,LpBinary)
    #dummy vars to eliminate subtours
    u = LpVariable.dicts('u', sites, 0, len(sites)-1, LpInteger)

    #the objective
    totalcost = lpSum([x[(i,j)]*distances[(i,j)] for (i,j) in distances])
    prob+=totalcost

    #constraints
    for k in sites:
        cap = 1 if k != start_node else num_agents
        #inbound connection
        prob+= lpSum([ x[(i,k)] for i in sites if (i,k) in x]) ==cap
        #outbound connection
        prob+=lpSum([ x[(k,i)] for i in sites if (k,i) in x]) ==cap

    #subtour elimination
    N=len(sites)/num_agents
    for i in sites:
        for j in sites:
            if i != j and (i != start_node and j!= start_node) and (i,j) in x:
                prob += u[i] - u[j] <= (N)*(1-x[(i,j)]) - 1

    starttime = time.time()
    prob.solve()
    elapsed = time.time() - starttime
    print('elapsed', elapsed)
    #prob.solve(GLPK_CMD(options=['--simplex']))
    print(LpStatus[prob.status])
    print('total distance', value(prob.objective))
    tour_lengths = get_tour_lengths(distance, x, start_node)
    print(tour_lengths)

def get_tour_lengths(distance, x, start_node):
    tours = []
    tour_lengths = []
    for node in distance.index:
        if (node == start_node):
            continue
        if (x[(start_node, node)].value() == 1):
            newtour = []
            newtour_length = 0
            beg_node = start_node
            end_node = node
            while end_node != start_node:
                newtour.append(end_node)
                newtour_length += distance.loc[beg_node, end_node]

                # find the next node
                beg_node = end_node
                end_node = find_next_node(distance.index, x, beg_node)
                if (end_node is None):
                    break

            newtour.append(end_node)
            newtour_length += distance.loc[beg_node, end_node]

            tours.append(newtour)
            tour_lengths.append(newtour_length)

    return tour_lengths

def find_next_node(nodes, x, selfnode):
    for node in nodes:
        if (node == selfnode):
            continue
        if (x[(selfnode, node)].value() == 1):
            return node
    return None

if __name__ == '__main__':
    solve_VRP('graphs/10nodes_3.json', 2)