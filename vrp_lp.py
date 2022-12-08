from pulp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse

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
    distances_k = dict(((s1, s2, q), distance.loc[s1, s2]) for s1 in positions for s2 in positions for q in range(num_agents) if s1 != s2)
    distances = dict(((s1, s2), distance.loc[s1, s2]) for s1 in positions for s2 in positions if s1 != s2)

    #create the problme
    prob=LpProblem("vrp", LpMinimize)

    #indicator variable if site i is connected to site j in the tour
    x = LpVariable.dicts('x',distances_k, 0,1,LpBinary)
    #dummy vars to eliminate subtours
    u = LpVariable.dicts('u', sites, 0, len(sites)-1, LpInteger)

    # minmax variable
    Q = LpVariable('Q')

    #the objective
    totalcost = lpSum([x[(i,j,k)]*distances[(i,j)] for (i,j) in distances for k in range(num_agents)])
    # prob+=totalcost
    prob += phi*Q + (1-phi)*totalcost
    # prob += Q

    #constraints
    for k in range(num_agents):
        prob += lpSum([x[(0,j,k)] for j in sites if j != start_node]) == 1
        prob += lpSum([x[(i,0,k)] for i in sites if i != start_node]) == 1

    for j in sites:
        if j == start_node:
            continue
        prob += lpSum([x[(i,j,k)] for i in sites for k in range(num_agents) if (i,j,k) in x and i != j]) == 1
        prob += lpSum([x[(j,i,k)] for i in sites for k in range(num_agents) if (j,i,k) in x and i != j]) == 1

    for j in sites:
        if j == start_node:
            continue
        for k in range(num_agents):
            prob += lpSum([x[(i,j,k)] for i in sites if i != j]) == lpSum([x[(j,i,k)] for i in sites if i != j])

    #subtour elimination
    N=len(sites)/num_agents
    for i in sites:
        for j in sites:
            if i != j and (i != start_node and j!= start_node) and (i,j,0) in x:
                prob += u[i] - u[j] <= (N)*(1-lpSum([x[(i,j,k)] for k in range(num_agents)])) - 1

    for k in range(num_agents):
        prob += lpSum([x[(i,j,k)]*distances[(i,j)] for (i,j) in distances]) <= Q

    starttime = time.time()
    prob.solve()
    elapsed = time.time() - starttime
    print('elapsed', elapsed)
    #prob.solve(GLPK_CMD(options=['--simplex']))
    print(LpStatus[prob.status])
    tours, tour_lengths = get_tour_lengths(distance, x, start_node, num_agents)
    print(tour_lengths)
    score = value(prob.objective)
    print('score: {}'.format(score))
    graphname = os.path.basename(os.path.splitext(fname)[0])
    graphname = '{}agents_{}_soln'.format(num_agents, graphname)
    plot_tours(tours, positions, graphname, value(prob.objective))

    print(Q.value(), totalcost.value())

    return score, totalcost.value(), Q.value(), elapsed

def get_tour_lengths(distance, x, start_node, num_agents):
    tours = []
    tour_lengths = []
    for k in range(0, num_agents):
        for node in distance.index:
            if (node == start_node):
                continue
            if (x[(start_node, node, k)].value() == 1):
                newtour = []
                newtour_length = 0
                beg_node = start_node
                end_node = node
                while end_node != start_node:
                    newtour.append((beg_node, end_node))
                    newtour_length += distance.loc[beg_node, end_node]

                    # find the next node
                    beg_node = end_node
                    end_node = find_next_node(distance.index, x, beg_node, num_agents)
                    if (end_node is None):
                        break

                newtour.append((beg_node, end_node))
                newtour_length += distance.loc[beg_node, end_node]

                tours.append(newtour)
                tour_lengths.append(newtour_length)

    return tours, tour_lengths

def find_next_node(nodes, x, selfnode, num_agents):
    for k in range(0, num_agents):
        for node in nodes:
            if (node == selfnode):
                continue
            if (x[(selfnode, node, k)].value() == 1):
                return node
    return None

def plot_tours(tours, positions, plot_name, score):
    #draw the tours
    colors = [np.random.rand(3) for i in range(len(tours))]
    for t,c in zip(tours,colors):
        for a,b in t:
            p1,p2 = positions[a], positions[b]
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]], color=c)

    #draw the map again
    for s in positions:
        p = positions[s]
        plt.plot(p[0],p[1],'o')
        plt.text(p[0]+.01,p[1],s,horizontalalignment='left',verticalalignment='center')

    # plt.gca().axis('off')
    # plt.show()
    plt.title('score: {}'.format(score))
    plt.savefig(plot_name + '.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('--num_agents', type=int, required=True)
    args = parser.parse_args()

    solve_VRP(args.fname, args.num_agents)