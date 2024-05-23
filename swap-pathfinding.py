#from IPython.core.display import display, HTML

import sys
from pulp import *
import gurobipy as gp
from gurobipy import GRB
from numpy import array, eye, hstack, ones, vstack, zeros
import numpy as np
import numpy.linalg as linalg
from pylab import dot, random
import time
import networkx as nx
import networkx.relabel as relabel
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = [10, 5]
import contextlib
import sys
import random
from itertools import combinations 
import gc
import heapq
import copy
import math
import random
import statistics
import json
import plotly
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def save_dict_as_json(dictionary, directory):
    filename, file_extension = os.path.splitext(directory)
    if file_extension == "":
        directory = directory + ".txt"
    with open(directory, "w") as f:
        json.dump(dictionary, f)
        

def create_ibmq_poughkeepsie_hardware_graph(scale = 1.):
    G=nx.Graph()
    G.add_nodes_from(list(range(20)))
    pos = {}
    for i in range(4):
        for j in range(5):
            pos[i * 5 + j] = array([j * scale, -i * scale])
    
    for i in range(4):
        G.add_edge(i,i+1)
    for i in range(5, 9):
        G.add_edge(i,i+1)
    for i in range(10, 14):
        G.add_edge(i,i+1)
    for i in range(15, 19):
        G.add_edge(i,i+1)
    G.add_edge(0,5)
    G.add_edge(4,9)
    G.add_edge(5,10)
    G.add_edge(7,12)
    G.add_edge(9,14)
    G.add_edge(10,15)
    G.add_edge(14,19)
    
    return G, pos
    
def create_ibmq_paris_hardware_graph(scale = 1.):
    G=nx.Graph()
    G.add_nodes_from(list(range(27)))
    pos = {}
    # define positions
    for node_index in range(27):
        if node_index == 0:
            pos[node_index] = array([0 * scale, -1 * scale])
        elif 1 <= node_index and node_index <= 3:
            pos[node_index] = array([1 * scale, (-node_index) * scale])
        elif 4 <= node_index and node_index <= 5:
            pos[node_index] = array([2 * scale, (-(node_index-4)*2 - 1) * scale])
        elif 6 <= node_index and node_index <= 7:
            pos[node_index] = array([3 * scale, (-(node_index-6)) * scale])
        elif 8 <= node_index and node_index <= 9:
            pos[node_index] = array([3 * scale, (-(node_index-6) - 1) * scale])
        elif 10 <= node_index and node_index <= 11:
            pos[node_index] = array([4 * scale, (-(node_index-10)*2 - 1) * scale])
        elif 12 <= node_index and node_index <= 14:
            pos[node_index] = array([5 * scale, (-(node_index-11)) * scale])
        elif 15 <= node_index and node_index <= 16:
            pos[node_index] = array([6 * scale, (-(node_index-15)*2 - 1) * scale])
        elif 17 <= node_index and node_index <= 18:
            pos[node_index] = array([7 * scale, (-(node_index-17)) * scale])
        elif 19 <= node_index and node_index <= 20:
            pos[node_index] = array([7 * scale, (-(node_index-17) - 1) * scale])
        elif 21 <= node_index and node_index <= 22:
            pos[node_index] = array([8 * scale, (-(node_index-21)*2 - 1) * scale])
        elif 23 <= node_index and node_index <= 25:
            pos[node_index] = array([9 * scale, (-(node_index-22)) * scale])
        elif node_index == 26:
            pos[node_index] = array([10 * scale, -3 * scale])
        
        G.add_edge(0,1)
        G.add_edge(1,2)
        G.add_edge(2,3)
        G.add_edge(1,4)
        G.add_edge(3,5)
        G.add_edge(4,7)
        G.add_edge(5,8)
        G.add_edge(6,7)
        G.add_edge(8,9)
        G.add_edge(7,10)
        G.add_edge(8,11)
        G.add_edge(10,12)
        G.add_edge(11,14)
        G.add_edge(12,13)
        G.add_edge(13,14)
        G.add_edge(12,15)
        G.add_edge(14,16)
        G.add_edge(15,18)
        G.add_edge(16,19)
        G.add_edge(17,18)
        G.add_edge(19,20)
        G.add_edge(18,21)
        G.add_edge(19,22)
        G.add_edge(21,23)
        G.add_edge(22,25)
        G.add_edge(23,24)
        G.add_edge(24,25)
        G.add_edge(25,26)
        
    return G, pos
    
def create_ibmq_melbourne_hardware_graph(scale = 1.):
    G=nx.Graph()
    G.add_nodes_from(list(range(15)))
    pos = {}
    # define positions
    for node_index in range(15):
        if node_index <= 6:
            pos[node_index] = array([node_index * scale, 0 * scale])
            if node_index != 6:
                G.add_edge(node_index, node_index+1)
            G.add_edge(node_index, 14 - node_index)
        else:
            pos[node_index] = array([(14-node_index) * scale, -1 * scale])
            if node_index != 14:
                G.add_edge(node_index, node_index+1)

    return G, pos

def create_ibmq_rochester_hardware_graph(scale = 1.):
    G=nx.Graph()
    G.add_nodes_from(list(range(53)))
    pos = {}
    # define positions
    for node_index in range(53):
        if node_index <= 4:
            pos[node_index] = array([(node_index+2) * scale, 0 * scale])
            if node_index != 4:
                G.add_edge(node_index,node_index+1)
        elif node_index == 5:
            pos[node_index] = array([2 * scale, -1 * scale])
            G.add_edge(0,5)
            G.add_edge(5,9)
        elif node_index == 6:
            pos[node_index] = array([6 * scale, -1 * scale])
            G.add_edge(4,6)
            G.add_edge(6,13)
        elif node_index >= 7 and node_index <= 15:
            pos[node_index] = array([(node_index - 7) * scale, -2 * scale])
            if node_index != 15:
                G.add_edge(node_index,node_index+1)
        elif node_index == 16:
            pos[node_index] = array([0 * scale, -3 * scale])
            G.add_edge(7,16)
            G.add_edge(16,19)
        elif node_index == 17:
            pos[node_index] = array([4 * scale, -3 * scale])
            G.add_edge(11,17)
            G.add_edge(17,23)
        elif node_index == 18:
            pos[node_index] = array([8 * scale, -3 * scale])
            G.add_edge(15,18)
            G.add_edge(18,27)
        elif node_index >= 19 and node_index <= 27:
            pos[node_index] = array([(node_index - 19) * scale, -4 * scale])
            if node_index != 27:
                G.add_edge(node_index,node_index+1)
        elif node_index == 28:
            pos[node_index] = array([2 * scale, -5 * scale])
            G.add_edge(21,28)
            G.add_edge(28,32)
        elif node_index == 29:
            pos[node_index] = array([6 * scale, -5 * scale])
            G.add_edge(25,29)
            G.add_edge(29,36)
        elif node_index >= 30 and node_index <= 38:
            pos[node_index] = array([(node_index - 30) * scale, -6 * scale])
            if node_index != 38:
                G.add_edge(node_index,node_index+1)
        elif node_index == 39:
            pos[node_index] = array([0 * scale, -7 * scale])
            G.add_edge(30,39)
            G.add_edge(39,42)
        elif node_index == 40:
            pos[node_index] = array([4 * scale, -7 * scale])
            G.add_edge(34,40)
            G.add_edge(40,46)
        elif node_index == 41:
            pos[node_index] = array([8 * scale, -7 * scale])
            G.add_edge(38,41)
            G.add_edge(41,50)
        elif node_index >= 42 and node_index <= 50:
            pos[node_index] = array([(node_index - 42) * scale, -8 * scale])
            if node_index != 50:
                G.add_edge(node_index,node_index+1)
        elif node_index == 51:
            pos[node_index] = array([2 * scale, -9 * scale])
            G.add_edge(44,51)
        elif node_index == 52:
            pos[node_index] = array([6 * scale, -9 * scale])
            G.add_edge(48,52)
    
    return G, pos
    
def create_rigetti_acorn_hardware_graph(scale = 1.):
    G=nx.Graph()
    node_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    G.add_nodes_from(node_list)
    scale *= 1.2
    pos = {}
    # define positions
    pos[0] = array([0.5 * scale, 0 * scale])
    pos[1] = array([1.5 * scale, 0 * scale])
    pos[2] = array([2.5 * scale, 0 * scale])
    pos[3] = array([3.5 * scale, 0 * scale])
    pos[4] = array([4.5 * scale, 0 * scale])
    
    pos[5] = array([0 * scale, -0.5 * scale])
    pos[6] = array([1 * scale, -0.5 * scale])
    pos[7] = array([2 * scale, -0.5 * scale])
    pos[8] = array([3 * scale, -0.5 * scale])
    pos[9] = array([4 * scale, -0.5 * scale])
    
    pos[10] = array([0.5 * scale, -1 * scale])
    pos[11] = array([1.5 * scale, -1 * scale])
    pos[12] = array([2.5 * scale, -1 * scale])
    pos[13] = array([3.5 * scale, -1 * scale])
    pos[14] = array([4.5 * scale, -1 * scale])
    
    pos[15] = array([0 * scale, -1.5 * scale])
    pos[16] = array([1 * scale, -1.5 * scale])
    pos[17] = array([2 * scale, -1.5 * scale])
    pos[18] = array([3 * scale, -1.5 * scale])
    pos[19] = array([4 * scale, -1.5 * scale])
    
    G.add_edge(5, 0)
    G.add_edge(0, 6)
    G.add_edge(6, 1)
    G.add_edge(1, 7)
    G.add_edge(7, 2)
    G.add_edge(2, 8)
    G.add_edge(8, 3)
    G.add_edge(3, 9)
    G.add_edge(9, 4)
    
    G.add_edge(5, 10)
    G.add_edge(6, 11)
    G.add_edge(7, 12)
    G.add_edge(8, 13)
    G.add_edge(9, 14)
    
    G.add_edge(15, 10)
    G.add_edge(10, 16)
    G.add_edge(16, 11)
    G.add_edge(11, 17)
    G.add_edge(17, 12)
    G.add_edge(12, 18)
    G.add_edge(18, 13)
    G.add_edge(13, 19)
    G.add_edge(19, 14)

    return G, pos

def time_expand_swap(graph, time, start_nodes = None, end_nodes = None, weights = None, start_name="source", end_name="target"):
    # weights is a dict where key is an edge as a tuple, value is the weight as an integer. 
    # Waiting qubits have zero cost
    
    NG = nx.DiGraph()
    pos = {}
    
    nodes = graph.nodes()
    edges = graph.edges()
    
    node_lookup_from_orig = {}
    orig_nodes = {}
    
    it = 0
    for node in nodes:
        node_lookup_from_orig[node] = it
        orig_nodes[it] = node
        it += 1
    
    for i in range(time):
        for j in range(len(nodes)):
            NG.add_node(i*len(nodes) + j)
            NG.nodes[i*len(nodes) + j]["time"] = i
            NG.nodes[i*len(nodes) + j]["location_id"] = j
            NG.nodes[i*len(nodes) + j]["out_nodes"] = []
            NG.nodes[i*len(nodes) + j]["in_nodes"] = []
            NG.nodes[i*len(nodes) + j]["pos"] = array([i, j])
            pos[i*len(nodes) + j] = array([i, j])
    
    for i in range(time):
        if not i == time - 1:
            for j in range(len(nodes)):
                neighbours = graph.neighbors(orig_nodes[j])
                for k in neighbours:
                    other = node_lookup_from_orig[k] + (i+1)*len(nodes)
                    NG.add_edge(i*len(nodes) + j, other)
                    
                    temp_edge = (j, NG.nodes[other]["location_id"])
                    if weights != None:
                        if temp_edge in weights.keys():
                            NG.edges[(i*len(nodes) + j, other)]['weight'] = weights[temp_edge]
                        else:
                            temp_edge = (NG.nodes[other]["location_id"], j)
                            if temp_edge in weights.keys():
                                NG.edges[(i*len(nodes) + j, other)]['weight'] = weights[temp_edge]
                    else:
                        NG.edges[(i*len(nodes) + j, other)]['weight'] = 1
                        
                    NG.nodes[i*len(nodes) + j]["out_nodes"].append(other)
                    NG.nodes[other]["in_nodes"].append(i*len(nodes) + j)
                NG.add_edge(i*len(nodes) + j, (i+1)*len(nodes) + j)
                NG.edges[(i*len(nodes) + j, (i+1)*len(nodes) + j)]['weight'] = 0
                NG.nodes[i*len(nodes) + j]["out_nodes"].append((i+1)*len(nodes) + j)
                NG.nodes[(i+1)*len(nodes) + j]["in_nodes"].append(i*len(nodes) + j)
                
    if start_nodes is not None:
        NG.add_node(start_name)
        NG.nodes[start_name]["out_nodes"] = []
        NG.nodes[start_name]["in_nodes"] = []
        NG.nodes[start_name]["location_id"] = -1
        NG.nodes[start_name]["pos"] = array([-1.5, (len(nodes)-1) / 2.])
        pos[start_name] = array([-1.5, (len(nodes)-1) / 2.])
        for node in start_nodes:
            NG.add_edge(start_name, node_lookup_from_orig[node])
            NG.edges[(start_name, node_lookup_from_orig[node])]['weight'] = 0
            NG.nodes[start_name]["out_nodes"].append(node_lookup_from_orig[node])
            NG.nodes[node_lookup_from_orig[node]]["in_nodes"].append(start_name)
            
    if end_nodes is not None:
        NG.add_node(end_name)
        NG.nodes[end_name]["out_nodes"] = []
        NG.nodes[end_name]["in_nodes"] = []
        NG.nodes[end_name]["location_id"] = -2
        NG.nodes[end_name]["pos"] = array([time + 0.5, (len(nodes)-1) / 2.])
        pos[end_name] = array([time + 0.5, (len(nodes)-1) / 2.])
        for node in end_nodes:
            NG.add_edge((time-1)*len(nodes) + node_lookup_from_orig[node], end_name)
            NG.edges[((time-1)*len(nodes) + node_lookup_from_orig[node], end_name)]['weight'] = 0
            NG.nodes[(time-1)*len(nodes) + node_lookup_from_orig[node]]["out_nodes"].append(end_name)
            NG.nodes[end_name]["in_nodes"].append((time-1)*len(nodes) + node_lookup_from_orig[node])

    #nx.draw(NG, with_labels=True, font_weight='bold', pos=pos)
    #plt.show()
    return NG, pos
    
def approximate_time_depth_lower_bound(graph, start_nodes_by_group = None, end_nodes_by_group = None):

    start = time.time()
    
    #new_graph = copy.deepcopy(graph)
    #    
    #if start_nodes_by_group is not None:
    #    for index, start_nodes in enumerate(start_nodes_by_group):
    #        source_node_name = "source_" + str(index)
    #        new_graph.add_node(source_node_name)
    #        for node in start_nodes:
    #            new_graph.add_edge(source_node_name, node)
    #            
    #if end_nodes_by_group is not None:
    #    for index, end_nodes in enumerate(end_nodes_by_group):
    #        target_node_name = "target_" + str(index)
    #        new_graph.add_node(target_node_name)
    #        for node in end_nodes:
    #            new_graph.add_edge(node, target_node_name)
    
    longest_shortest_length = 0
    
    for group_index in range(len(start_nodes_by_group)):
        shortest_paths = []
        for start_index, start_node in enumerate(start_nodes_by_group[group_index]):
            shortest_paths.append([])
            source = start_nodes_by_group[group_index][start_index]
            for end_index, end_node in enumerate(end_nodes_by_group[group_index]):
                target = end_nodes_by_group[group_index][end_index]
                shortest_paths[start_index].append(nx.shortest_path_length(graph, source=source, target=target))
            
        for start_index, start_node in enumerate(start_nodes_by_group[group_index]):
            shortest_length_to_targets = math.inf
            for end_index, end_node in enumerate(end_nodes_by_group[group_index]):
                shortest_length_to_targets = min(shortest_length_to_targets, shortest_paths[start_index][end_index])
            longest_shortest_length = max(longest_shortest_length, shortest_length_to_targets)
        
        for end_index, end_node in enumerate(end_nodes_by_group[group_index]):
            shortest_length_to_targets = math.inf
            for start_index, start_node in enumerate(start_nodes_by_group[group_index]):
                shortest_length_to_targets = min(shortest_length_to_targets, shortest_paths[start_index][end_index])
            longest_shortest_length = max(longest_shortest_length, shortest_length_to_targets)

    
    end_init = time.time()
    runtime = end_init - start
    runtime *= 1000
    msg = "approximate_time_depth_lower_bound execution time: {time} ms"
    print("Longest shortest path:", longest_shortest_length + 1)
    print(msg.format(time=runtime))
    
    return longest_shortest_length + 1
        
def time_expand_swap_multi(graph, time, start_nodes_by_group = None, end_nodes_by_group = None, weights = None):
    # 
    # weights is a dict where key is an edge as a tuple, value is the weight as an integer. 
    # Waiting qubits have zero cost
    
    NG = nx.DiGraph()
    pos = {}
    
    nodes = graph.nodes()
    edges = graph.edges()
    
    node_lookup_from_orig = {}
    orig_nodes = {}
    
    it = 0
    for node in nodes:
        node_lookup_from_orig[node] = it
        orig_nodes[it] = node
        it += 1
    
    for i in range(time):
        for j in range(len(nodes)):
            NG.add_node(i*len(nodes) + j)
            NG.nodes[i*len(nodes) + j]["time"] = i
            NG.nodes[i*len(nodes) + j]["location_id"] = j
            NG.nodes[i*len(nodes) + j]["out_nodes"] = []
            NG.nodes[i*len(nodes) + j]["in_nodes"] = []
            NG.nodes[i*len(nodes) + j]["pos"] = array([i, j])
            pos[i*len(nodes) + j] = array([i, j])
    
    for i in range(time):
        if not i == time - 1:
            for j in range(len(nodes)):
                neighbours = graph.neighbors(orig_nodes[j])
                for k in neighbours:
                    other = node_lookup_from_orig[k] + (i+1)*len(nodes)
                    NG.add_edge(i*len(nodes) + j, other)
                    
                    temp_edge = (j, NG.nodes[other]["location_id"])
                    if weights != None:
                        if temp_edge in weights.keys():
                            NG.edges[(i*len(nodes) + j, other)]['weight'] = weights[temp_edge]
                        else:
                            temp_edge = (NG.nodes[other]["location_id"], j)
                            if temp_edge in weights.keys():
                                NG.edges[(i*len(nodes) + j, other)]['weight'] = weights[temp_edge]
                    else:
                        NG.edges[(i*len(nodes) + j, other)]['weight'] = 1
                        
                    NG.nodes[i*len(nodes) + j]["out_nodes"].append(other)
                    NG.nodes[other]["in_nodes"].append(i*len(nodes) + j)
                NG.add_edge(i*len(nodes) + j, (i+1)*len(nodes) + j)
                NG.edges[(i*len(nodes) + j, (i+1)*len(nodes) + j)]['weight'] = 0
                NG.nodes[i*len(nodes) + j]["out_nodes"].append((i+1)*len(nodes) + j)
                NG.nodes[(i+1)*len(nodes) + j]["in_nodes"].append(i*len(nodes) + j)
    
    outside_id = -1
    if start_nodes_by_group is not None:
        for index, start_nodes in enumerate(start_nodes_by_group):
            source_node_name = "source_" + str(index)
            NG.add_node(source_node_name)
            NG.nodes[source_node_name]["out_nodes"] = []
            NG.nodes[source_node_name]["in_nodes"] = []
            NG.nodes[source_node_name]["location_id"] = outside_id
            outside_id -= 1
            NG.nodes[source_node_name]["pos"] = array([-1.5, start_nodes[0]])
            pos[source_node_name] = array([-1.5, start_nodes[0]])
            for node in start_nodes:
                NG.add_edge(source_node_name, node_lookup_from_orig[node])
                NG.edges[(source_node_name, node_lookup_from_orig[node])]['weight'] = 0
                NG.nodes[source_node_name]["out_nodes"].append(node_lookup_from_orig[node])
                NG.nodes[node_lookup_from_orig[node]]["in_nodes"].append(source_node_name)
            
    if end_nodes_by_group is not None:
        for index, end_nodes in enumerate(end_nodes_by_group):
            target_node_name = "target_" + str(index)
            NG.add_node(target_node_name)
            NG.nodes[target_node_name]["out_nodes"] = []
            NG.nodes[target_node_name]["in_nodes"] = []
            NG.nodes[target_node_name]["location_id"] = outside_id
            outside_id -= 1
            NG.nodes[target_node_name]["pos"] = array([time + 0.5, end_nodes[0]])
            pos[target_node_name] = array([time + 0.5, end_nodes[0]])
            for node in end_nodes:
                NG.add_edge((time-1)*len(nodes) + node_lookup_from_orig[node], target_node_name)
                NG.edges[((time-1)*len(nodes) + node_lookup_from_orig[node], target_node_name)]['weight'] = 0
                NG.nodes[(time-1)*len(nodes) + node_lookup_from_orig[node]]["out_nodes"].append(target_node_name)
                NG.nodes[target_node_name]["in_nodes"].append((time-1)*len(nodes) + node_lookup_from_orig[node])

    #nx.draw(NG, with_labels=True, font_weight='bold', pos=pos)
    #plt.show()
    return NG, pos

def generate_linear_program_pulp(time_expanded_graph, start_name="source", end_name="target"):
    # return (c, G, h)
    # rows (constraints) and columns (variables / edge #)
    start = time.time()
    print("Converting to linear program and solving using pulp...")
    
    time_expanded_graph_edges = [e for e in time_expanded_graph.edges]
    number_of_variables = len(time_expanded_graph_edges)
    
    variable_to_edge = {}
    for i in range(len(time_expanded_graph.edges())):
        variable_to_edge[i] = time_expanded_graph_edges[i]
    edge_to_variable = {}
    for i in range(len(time_expanded_graph.edges())):
        edge_to_variable[time_expanded_graph_edges[i]] = i
    
    
    lp_problem = LpProblem("swap_based_pathfinding", LpMinimize)
    
    weights = [(float)(time_expanded_graph.edges[x]["weight"]) for x in time_expanded_graph_edges]
    
    # Includes Max and Min capacity constraints (LpContinuous)
    flows = [LpVariable(str(edge_to_variable[x]), 0, 1, LpBinary) for x in time_expanded_graph_edges]
    
    # The objective function is added first
    lp_problem += lpSum([flows[i]*weights[i] for i in range(len(time_expanded_graph_edges))]), "sum_of_path_segment_costs"
    
    
    # constraints
    
    # Source Constraint
    number_of_out_nodes = len(time_expanded_graph.nodes[start_name]["out_nodes"])
    lp_problem += lpSum([flows[edge_to_variable[(start_name, x)]] for x in time_expanded_graph.nodes[start_name]["out_nodes"]]) >= number_of_out_nodes, start_name
    

    # flow conservation constraints
    # for each node, get variable corresponding to in and out edges (+ in edge, - out edge)
    for node in time_expanded_graph.nodes:
        if node == start_name or node == end_name:
            continue
        in_edges = []
        out_edges = []
        
        for in_node in time_expanded_graph.nodes[node]['in_nodes']:
            in_edges.append((in_node, node))
        for out_node in time_expanded_graph.nodes[node]['out_nodes']:
            out_edges.append((node, out_node))
        
        lp_problem += lpSum([flows[edge_to_variable[edge]] for edge in in_edges] + [-flows[edge_to_variable[edge]] for edge in out_edges])<=0, ("flow_conservation_upper_" + str(node))
        lp_problem += lpSum([-flows[edge_to_variable[edge]] for edge in in_edges] + [flows[edge_to_variable[edge]] for edge in out_edges])<=0, ("flow_conservation_lower_" + str(node))
    
    
    # swap constraints
    # <= 1
    for node in time_expanded_graph.nodes:
        # swap constraints don't affect source and target nodes
        if node == start_name or node == end_name:
            continue
            
        for out_node in time_expanded_graph.nodes[node]['out_nodes']:
            this_id = time_expanded_graph.nodes[node]['location_id']
            target_id = time_expanded_graph.nodes[out_node]['location_id']
            
            if this_id == target_id:
                continue
            
            involved_edges = []
            involved_edges.append((node, out_node))
            
            for in_node in time_expanded_graph.nodes[out_node]['in_nodes']:
                if in_node == node:
                    continue
                in_node_id = time_expanded_graph.nodes[in_node]['location_id']
                if in_node_id == target_id:
                    # this is the node we want to force into this_id location
                    for out_in_node in time_expanded_graph.nodes[in_node]['out_nodes']:
                        out_in_node_id = time_expanded_graph.nodes[out_in_node]['location_id']
                        if out_in_node_id == this_id:
                            # out_in_node is the initial node except one time step forward
                            continue
                        involved_edges.append((in_node, out_in_node))
                        

            lp_problem += lpSum([flows[edge_to_variable[edge]] for edge in involved_edges])<=1, ("swap_constraint_" + str((node, out_node)))
    
    lp_problem.writeLP("SwapBasedPathfinding.lp")
    lp_problem.solve()
    
    print("Status:", LpStatus[lp_problem.status])
    print("Total Cost = ", value(lp_problem.objective))
    end = time.time()
    runtime = end - start
    runtime *= 1000
    msg = "execution time: {time} ms"
    print(msg.format(time=runtime))
    print()
    
    #for v in lp_problem.variables():
    #    print(v.name, "=", v.varValue)

    flow_results = None
    if str(LpStatus[lp_problem.status]) == "Optimal":
        flow_results_dict = {}
        for v in lp_problem.variables():
            if v.name != "__dummy":
                flow_results_dict[int(v.name)] = v.varValue

        flow_results = []
        found_dummy = False
        for i in range(len(flow_results_dict.keys())):
            if i not in flow_results_dict:
                flow_results.append(0)
                found_dummy = True
            else:
                flow_results.append(flow_results_dict[i])
        if found_dummy == True:
            time_expanded_graph_edges = [e for e in time_expanded_graph.edges]
            solution_edges = []
            for i in range(len(flow_results)):
                if flow_results[i] > 0.9:
                    solution_edges.append(time_expanded_graph_edges[i])
            print("")
            print("DUMMY FOUND - solution_edges:")
            print(str(solution_edges))
            print("")
            assert(False)

            
    return flow_results, LpStatus[lp_problem.status]

def generate_linear_program_pulp_multi(time_expanded_graph, start_names=["source"], end_names=["target"], allow_out_flow_less_than_in = False, trim_impossible_variables = False):
    # return (c, G, h)
    # rows (constraints) and columns (variables / edge #)
    start = time.time()
    print("Converting to linear program and solving using pulp...")
    
    time_expanded_graph_edges = [e for e in time_expanded_graph.edges]
    number_of_variables = len(start_names) * len(time_expanded_graph_edges)

    variable_to_edge = {}
    var_iter = 0
    for i in range(len(start_names)):
        for j in range(len(time_expanded_graph_edges)):
            variable_to_edge[var_iter] = time_expanded_graph_edges[j]
            var_iter += 1
        
    edge_to_variables = {}
    variables = []
    var_iter = 0
    for i in range(len(start_names)):
        for j in range(len(time_expanded_graph_edges)):
            if time_expanded_graph_edges[j] not in edge_to_variables:
                edge_to_variables[time_expanded_graph_edges[j]] = []
            edge_to_variables[time_expanded_graph_edges[j]].append(var_iter)
            variables.append(var_iter)
            var_iter += 1

    if trim_impossible_variables == True:
        # loop through each start_names and flag each variable
        variables_start_flags = [False] * len(variables)
        for start_index, start_name in enumerate(start_names):
            node_iter = 0
            searched_nodes = [(node_iter, start_name)]
            node_iter += 1
            assessed_variables = set()
            
            while len(searched_nodes) != 0:
                node_name = heapq.heappop(searched_nodes)[1]
                
                out_nodes = time_expanded_graph.nodes[node_name]["out_nodes"]
                for out_node in out_nodes:
                    variable = edge_to_variables[(node_name, out_node)][start_index]
                    if variable in assessed_variables:
                        continue
                    assessed_variables.add(variable)
                    variables_start_flags[variable] = True
        
                    heapq.heappush(searched_nodes, (node_iter, out_node))
                    node_iter += 1
                    
        # loop through each end_names and flag each variable
        variables_end_flags = [False] * len(variables)
        
        for end_index, end_name in enumerate(end_names):
            node_iter = 0
            searched_nodes = [(node_iter, end_name)]
            node_iter += 1
            assessed_variables = set()
            
            while len(searched_nodes) != 0:
                node_name = heapq.heappop(searched_nodes)[1]
                
                in_nodes = time_expanded_graph.nodes[node_name]["in_nodes"]
                for in_node in in_nodes:
                    variable = edge_to_variables[(in_node, node_name)][end_index]
                    if variable in assessed_variables:
                        continue
                    assessed_variables.add(variable)
                    variables_end_flags[variable] = True
        
                    heapq.heappush(searched_nodes, (node_iter, in_node))
                    node_iter += 1


        # Repopulate variable_to_edge as dict with only possible variables
        var_iter = 0
        convert_possible_var_to_var = []
        convert_var_to_possible_var = [None]*len(variables)
        variable_to_edge_dict = {}
        for i in range(len(start_names)):
            for j in range(len(time_expanded_graph_edges)):
                if variables_start_flags[var_iter] == True and variables_end_flags[var_iter] == True:
                    variable_to_edge_dict[var_iter] = variable_to_edge[var_iter]
                    convert_var_to_possible_var[var_iter] = len(convert_possible_var_to_var)
                    convert_possible_var_to_var.append(var_iter)
                var_iter += 1
        variable_to_edge = variable_to_edge_dict

        # Set impossible variables in edge_to_variables to None
        for i in range(len(time_expanded_graph_edges)):
            for j in range(len(edge_to_variables[time_expanded_graph_edges[i]])):
                if edge_to_variables[time_expanded_graph_edges[i]][j] not in variable_to_edge_dict:
                    edge_to_variables[time_expanded_graph_edges[i]][j] = None



    end_init = time.time()
    runtime = end_init - start
    runtime *= 1000
    msg = "init execution time: {time} ms"
    print(msg.format(time=runtime))
    
    lp_problem = LpProblem("swap_based_pathfinding", LpMinimize)
    
    weights = [(float)(time_expanded_graph.edges[x]["weight"]) for x in time_expanded_graph_edges]
    weights_dict = {}
    for edge in time_expanded_graph_edges:
        weights_dict[edge] = (float)(time_expanded_graph.edges[edge]["weight"])
    
    # Includes Max and Min capacity constraints (LpContinuous)
    if trim_impossible_variables == True:
        flows = []
        for x in variables:
            if convert_var_to_possible_var[x] != None:
                flows.append(LpVariable(str(convert_var_to_possible_var[x]), 0, 1, LpBinary))
    else:
        flows = [LpVariable(str(x), 0, 1, LpBinary) for x in variables]
    
    # The objective function is added first
    if trim_impossible_variables == True:
        objective_variables = []
        for i in range(len(variables)):
            if convert_var_to_possible_var[i] != None:
                objective_variables.append(flows[convert_var_to_possible_var[i]] * weights_dict[variable_to_edge[i]])
    else:
        objective_variables = [flows[i]*weights_dict[variable_to_edge[i]] for i in range(len(variables))]
        
    lp_problem += lpSum(objective_variables), "sum_of_path_segment_costs"
    
    # constraints
    
    # Source Constraint
    for start_index, start_name in enumerate(start_names):
        if trim_impossible_variables == True:
            # if any of these are not possible than solution to this program is infeasible
            for x in time_expanded_graph.nodes[start_name]["out_nodes"]:
                if edge_to_variables[(start_name, x)][start_index] == None:
                    # infeasible
                    print("Status: Infeasible (preprocessing step)")
                    end = time.time()
                    runtime = end - start
                    runtime *= 1000
                    msg = "execution time: {time} ms"
                    print(msg.format(time=runtime))
                    print()
                    return None, "Infeasible", 0
        number_of_out_nodes = len(time_expanded_graph.nodes[start_name]["out_nodes"])
        if trim_impossible_variables == True:
            lp_problem += lpSum([flows[convert_var_to_possible_var[edge_to_variables[(start_name, x)][start_index]]] for x in time_expanded_graph.nodes[start_name]["out_nodes"]]) >= number_of_out_nodes, start_name
        else:
            lp_problem += lpSum([flows[edge_to_variables[(start_name, x)][start_index]] for x in time_expanded_graph.nodes[start_name]["out_nodes"]]) >= number_of_out_nodes, start_name
    # Target Constraint
    for end_index, end_name in enumerate(end_names):
        if trim_impossible_variables == True:
            # if any of these are not possible than solution to this program is infeasible
            for x in time_expanded_graph.nodes[end_name]["in_nodes"]:
                if edge_to_variables[(x,end_name)][end_index] == None:
                    # infeasible
                    print("Status: Infeasible (preprocessing step)")
                    end = time.time()
                    runtime = end - start
                    runtime *= 1000
                    msg = "execution time: {time} ms"
                    print(msg.format(time=runtime))
                    print()
                    return None, "Infeasible", 0
        number_of_in_nodes = len(time_expanded_graph.nodes[end_name]["in_nodes"])
        if trim_impossible_variables == True:
            lp_problem += lpSum([flows[convert_var_to_possible_var[edge_to_variables[(x,end_name)][end_index]]] for x in time_expanded_graph.nodes[end_name]["in_nodes"]]) >= number_of_in_nodes, end_name
        else:
            lp_problem += lpSum([flows[edge_to_variables[(x,end_name)][end_index]] for x in time_expanded_graph.nodes[end_name]["in_nodes"]]) >= number_of_in_nodes, end_name

    # flow conservation constraints
    # for each node, get variable corresponding to in and out edges (+ in edge, - out edge)
    for start_index, start_name in enumerate(start_names):
        for node in time_expanded_graph.nodes:
            if node in start_names or node in end_names:
                continue
            in_edges = []
            out_edges = []

            for in_node in time_expanded_graph.nodes[node]['in_nodes']:
                in_edges.append((in_node, node))
            for out_node in time_expanded_graph.nodes[node]['out_nodes']:
                out_edges.append((node, out_node))
            
            if trim_impossible_variables == True:
                in_flow_variables = []
                for edge in in_edges:
                    if edge_to_variables[edge][start_index] != None:
                        in_flow_variables.append(flows[convert_var_to_possible_var[edge_to_variables[edge][start_index]]])
                out_flow_variables = []
                for edge in out_edges:
                    if edge_to_variables[edge][start_index] != None:
                        out_flow_variables.append(flows[convert_var_to_possible_var[edge_to_variables[edge][start_index]]])
                        
                if allow_out_flow_less_than_in == False:
                    lp_problem += lpSum([-x for x in in_flow_variables] + out_flow_variables)<=0, ("flow_conservation_lower_" + str(node) + "_" + str(start_index))
                lp_problem += lpSum(in_flow_variables + [-x for x in out_flow_variables])<=0, ("flow_conservation_upper_" + str(node) + "_" + str(start_index))
            else:
                if allow_out_flow_less_than_in == False:
                    lp_problem += lpSum([-flows[edge_to_variables[edge][start_index]] for edge in in_edges] + [flows[edge_to_variables[edge][start_index]] for edge in out_edges])<=0, ("flow_conservation_lower_" + str(node) + "_" + str(start_index))
                lp_problem += lpSum([flows[edge_to_variables[edge][start_index]] for edge in in_edges] + [-flows[edge_to_variables[edge][start_index]] for edge in out_edges])<=0, ("flow_conservation_upper_" + str(node) + "_" + str(start_index))
    
    # swap constraints
    # <= 1
    # TODO: Can trip these constraints. 
    for node in time_expanded_graph.nodes:
        # swap constraints don't affect source and target nodes
        if node in start_names or node in end_names:
            continue
            
        for out_node in time_expanded_graph.nodes[node]['out_nodes']:
            this_id = time_expanded_graph.nodes[node]['location_id']
            target_id = time_expanded_graph.nodes[out_node]['location_id']
            
            if this_id == target_id:
                continue
            
            involved_edges = []
            involved_edges.append((node, out_node))
            
            for in_node in time_expanded_graph.nodes[out_node]['in_nodes']:
                if in_node == node:
                    continue
                in_node_id = time_expanded_graph.nodes[in_node]['location_id']
                if in_node_id == target_id:
                    # this is the node we want to force into this_id location
                    for out_in_node in time_expanded_graph.nodes[in_node]['out_nodes']:
                        out_in_node_id = time_expanded_graph.nodes[out_in_node]['location_id']
                        if out_in_node_id == this_id:
                            # out_in_node is the initial node except one time step forward
                            continue
                                
                        involved_edges.append((in_node, out_in_node))
                        
                        
            if trim_impossible_variables == True:
                flow_list = []
                for edge in involved_edges:
                    for variable in edge_to_variables[edge]:
                        if variable != None:
                            flow_list.append(flows[convert_var_to_possible_var[variable]])
                lp_problem += lpSum(flow_list)<=1, ("swap_constraint_" + str((node, out_node)))
            else:
                flow_list = []
                for edge in involved_edges:
                    for variable in edge_to_variables[edge]:
                        flow_list.append(flows[variable])
                lp_problem += lpSum(flow_list)<=1, ("swap_constraint_" + str((node, out_node)))
                
    
    lp_problem.writeLP("SwapBasedPathfinding.lp")
    lp_problem.solve()
    
    print("Status:", LpStatus[lp_problem.status])
    print("Total Cost = ", value(lp_problem.objective))
    end = time.time()
    runtime = end - start
    runtime *= 1000
    msg = "execution time: {time} ms"
    print(msg.format(time=runtime))
    print()

    flow_results_by_group = None
    if str(LpStatus[lp_problem.status]) == "Optimal":
        flow_results_dict = {}
        for v in lp_problem.variables():
            if v.name != "__dummy":
                if trim_impossible_variables == True:
                    flow_results_dict[convert_possible_var_to_var[int(v.name)]] = v.varValue
                else:
                    flow_results_dict[int(v.name)] = v.varValue
        flow_results_by_group = []
        found_dummy = False
                
        var_iter = 0
        for i in range(len(start_names)):
            flow_results_by_group.append([])
            for j in range(len(time_expanded_graph_edges)):
                if var_iter not in flow_results_dict:
                    flow_results_by_group[i].append(0)
                else:
                    flow_results_by_group[i].append(flow_results_dict[var_iter])
                var_iter += 1
                
                
                
                
                
                
        if found_dummy == True:
            solution_edges = []
            for index, flow_results in enumerate(flow_results_by_group):
                solution_edges.append([])
                for i in range(len(flow_results)):
                    if flow_results[i] > 0.9:
                        solution_edges[index].append(time_expanded_graph_edges[i])
                        
            print("")
            print("DUMMY FOUND - solution_edges:")
            print(str(solution_edges))
            print("")
            assert(False)

    # still return the full flow results. except fill in the missing variables with 0
    return flow_results_by_group, LpStatus[lp_problem.status], value(lp_problem.objective)
    
def generate_linear_program_gurobi_multi(time_expanded_graph, start_names=["source"], end_names=["target"], time_limit = 1000., allow_out_flow_less_than_in = False, trim_impossible_variables = False):
    # return (c, G, h)
    # rows (constraints) and columns (variables / edge #)
    start = time.time()
    print("Converting to linear program and solving using pulp...")
    
    time_expanded_graph_edges = [e for e in time_expanded_graph.edges]
    number_of_variables = len(start_names) * len(time_expanded_graph_edges)

    variable_to_edge = {}
    var_iter = 0
    for i in range(len(start_names)):
        for j in range(len(time_expanded_graph_edges)):
            variable_to_edge[var_iter] = time_expanded_graph_edges[j]
            var_iter += 1
        
    edge_to_variables = {}
    variables = []
    var_iter = 0
    for i in range(len(start_names)):
        for j in range(len(time_expanded_graph_edges)):
            if time_expanded_graph_edges[j] not in edge_to_variables:
                edge_to_variables[time_expanded_graph_edges[j]] = []
            edge_to_variables[time_expanded_graph_edges[j]].append(var_iter)
            variables.append(var_iter)
            var_iter += 1

    if trim_impossible_variables == True:
        # loop through each start_names and flag each variable
        variables_start_flags = [False] * len(variables)
        for start_index, start_name in enumerate(start_names):
            node_iter = 0
            searched_nodes = [(node_iter, start_name)]
            node_iter += 1
            assessed_variables = set()
            
            while len(searched_nodes) != 0:
                node_name = heapq.heappop(searched_nodes)[1]
                
                out_nodes = time_expanded_graph.nodes[node_name]["out_nodes"]
                for out_node in out_nodes:
                    variable = edge_to_variables[(node_name, out_node)][start_index]
                    if variable in assessed_variables:
                        continue
                    assessed_variables.add(variable)
                    variables_start_flags[variable] = True
        
                    heapq.heappush(searched_nodes, (node_iter, out_node))
                    node_iter += 1
                    
        # loop through each end_names and flag each variable
        variables_end_flags = [False] * len(variables)
        
        for end_index, end_name in enumerate(end_names):
            node_iter = 0
            searched_nodes = [(node_iter, end_name)]
            node_iter += 1
            assessed_variables = set()
            
            while len(searched_nodes) != 0:
                node_name = heapq.heappop(searched_nodes)[1]
                
                in_nodes = time_expanded_graph.nodes[node_name]["in_nodes"]
                for in_node in in_nodes:
                    variable = edge_to_variables[(in_node, node_name)][end_index]
                    if variable in assessed_variables:
                        continue
                    assessed_variables.add(variable)
                    variables_end_flags[variable] = True
        
                    heapq.heappush(searched_nodes, (node_iter, in_node))
                    node_iter += 1


        # Repopulate variable_to_edge as dict with only possible variables
        var_iter = 0
        convert_possible_var_to_var = []
        convert_var_to_possible_var = [None]*len(variables)
        variable_to_edge_dict = {}
        for i in range(len(start_names)):
            for j in range(len(time_expanded_graph_edges)):
                if variables_start_flags[var_iter] == True and variables_end_flags[var_iter] == True:
                    variable_to_edge_dict[var_iter] = variable_to_edge[var_iter]
                    convert_var_to_possible_var[var_iter] = len(convert_possible_var_to_var)
                    convert_possible_var_to_var.append(var_iter)
                var_iter += 1
        variable_to_edge = variable_to_edge_dict

        # Set impossible variables in edge_to_variables to None
        for i in range(len(time_expanded_graph_edges)):
            for j in range(len(edge_to_variables[time_expanded_graph_edges[i]])):
                if edge_to_variables[time_expanded_graph_edges[i]][j] not in variable_to_edge_dict:
                    edge_to_variables[time_expanded_graph_edges[i]][j] = None



    end_init = time.time()
    runtime = end_init - start
    runtime *= 1000
    msg = "init execution time: {time} ms"
    print(msg.format(time=runtime))
    
    lp_problem = gp.Model("swap_based_pathfinding")
    lp_problem.Params.TimeLimit = time_limit
    
    weights = [(float)(time_expanded_graph.edges[x]["weight"]) for x in time_expanded_graph_edges]
    weights_dict = {}
    for edge in time_expanded_graph_edges:
        weights_dict[edge] = (float)(time_expanded_graph.edges[edge]["weight"])
    
    # Includes Max and Min capacity constraints (LpContinuous)
    if trim_impossible_variables == True:
        flows = []
        for x in variables:
            if convert_var_to_possible_var[x] != None:
                flows.append(lp_problem.addVar(vtype=GRB.BINARY, name=str(convert_var_to_possible_var[x])))
                #flows.append(LpVariable(str(convert_var_to_possible_var[x]), 0, 1, LpBinary))
    else:
        flows = [lp_problem.addVar(vtype=GRB.BINARY, name=str(x)) for x in variables]
    
    # The objective function is added first
    if trim_impossible_variables == True:
        objective_variables = []
        for i in range(len(variables)):
            if convert_var_to_possible_var[i] != None:
                objective_variables.append(flows[convert_var_to_possible_var[i]] * weights_dict[variable_to_edge[i]])
    else:
        objective_variables = [flows[i]*weights_dict[variable_to_edge[i]] for i in range(len(variables))]
        
    #lp_problem += lpSum(objective_variables), "sum_of_path_segment_costs"
    objective = gp.LinExpr()
    for obj_var in objective_variables:
        objective += obj_var

    lp_problem.setObjective(objective, GRB.MINIMIZE)
        
    # constraints
    
    # Source Constraint (TODO: Slight optimisation by setting source flows to 1 rather than a variable >= 1, exclude from optimisation)
    for start_index, start_name in enumerate(start_names):
        if trim_impossible_variables == True:
            # if any of these are not possible than solution to this program is infeasible
            for x in time_expanded_graph.nodes[start_name]["out_nodes"]:
                if edge_to_variables[(start_name, x)][start_index] == None:
                    # infeasible
                    print("Status: Infeasible (preprocessing step)")
                    end = time.time()
                    runtime = end - start
                    runtime *= 1000
                    msg = "execution time: {time} ms"
                    print(msg.format(time=runtime))
                    print()
                    return None, "Infeasible", 0
        number_of_out_nodes = len(time_expanded_graph.nodes[start_name]["out_nodes"])
        if trim_impossible_variables == True:
            sum_vars = gp.LinExpr()
            for constr_var in [flows[convert_var_to_possible_var[edge_to_variables[(start_name, x)][start_index]]] for x in time_expanded_graph.nodes[start_name]["out_nodes"]]:
                sum_vars += constr_var
            lp_problem.addConstr(sum_vars >= number_of_out_nodes, start_name)
            #lp_problem += lpSum([flows[convert_var_to_possible_var[edge_to_variables[(start_name, x)][start_index]]] for x in time_expanded_graph.nodes[start_name]["out_nodes"]]) >= number_of_out_nodes, start_name
        else:
            sum_vars = gp.LinExpr()
            for constr_var in [flows[edge_to_variables[(start_name, x)][start_index]] for x in time_expanded_graph.nodes[start_name]["out_nodes"]]:
                sum_vars += constr_var
            lp_problem.addConstr(sum_vars >= number_of_out_nodes, start_name)
            #lp_problem += lpSum([flows[edge_to_variables[(start_name, x)][start_index]] for x in time_expanded_graph.nodes[start_name]["out_nodes"]]) >= number_of_out_nodes, start_name
    
    # Target Constraint
    for end_index, end_name in enumerate(end_names):
        if trim_impossible_variables == True:
            # if any of these are not possible than solution to this program is infeasible
            for x in time_expanded_graph.nodes[end_name]["in_nodes"]:
                if edge_to_variables[(x,end_name)][end_index] == None:
                    # infeasible
                    print("Status: Infeasible (preprocessing step)")
                    end = time.time()
                    runtime = end - start
                    runtime *= 1000
                    msg = "execution time: {time} ms"
                    print(msg.format(time=runtime))
                    print()
                    return None, "Infeasible", 0
        number_of_in_nodes = len(time_expanded_graph.nodes[end_name]["in_nodes"])
        if trim_impossible_variables == True:
            sum_vars = gp.LinExpr()
            for constr_var in [flows[convert_var_to_possible_var[edge_to_variables[(x,end_name)][end_index]]] for x in time_expanded_graph.nodes[end_name]["in_nodes"]]:
                sum_vars += constr_var
            lp_problem.addConstr(sum_vars >= number_of_in_nodes, end_name)
            #lp_problem += lpSum([flows[convert_var_to_possible_var[edge_to_variables[(x,end_name)][end_index]]] for x in time_expanded_graph.nodes[end_name]["in_nodes"]]) >= number_of_in_nodes, end_name
        else:
            sum_vars = gp.LinExpr()
            for constr_var in [flows[edge_to_variables[(x,end_name)][end_index]] for x in time_expanded_graph.nodes[end_name]["in_nodes"]]:
                sum_vars += constr_var
            lp_problem.addConstr(sum_vars >= number_of_in_nodes, end_name)
            #lp_problem += lpSum([flows[edge_to_variables[(x,end_name)][end_index]] for x in time_expanded_graph.nodes[end_name]["in_nodes"]]) >= number_of_in_nodes, end_name

    # flow conservation constraints
    # for each node, get variable corresponding to in and out edges (+ in edge, - out edge)
    for start_index, start_name in enumerate(start_names):
        for node in time_expanded_graph.nodes:
            if node in start_names or node in end_names:
                continue
            in_edges = []
            out_edges = []

            for in_node in time_expanded_graph.nodes[node]['in_nodes']:
                in_edges.append((in_node, node))
            for out_node in time_expanded_graph.nodes[node]['out_nodes']:
                out_edges.append((node, out_node))
            
            if trim_impossible_variables == True:
                in_flow_variables = []
                for edge in in_edges:
                    if edge_to_variables[edge][start_index] != None:
                        in_flow_variables.append(flows[convert_var_to_possible_var[edge_to_variables[edge][start_index]]])
                out_flow_variables = []
                for edge in out_edges:
                    if edge_to_variables[edge][start_index] != None:
                        out_flow_variables.append(flows[convert_var_to_possible_var[edge_to_variables[edge][start_index]]])
                        
                if allow_out_flow_less_than_in == False:
                    sum_vars = gp.LinExpr()
                    for constr_var in [-x for x in in_flow_variables] + out_flow_variables:
                        sum_vars += constr_var
                    lp_problem.addConstr(sum_vars <= 0, "flow_conservation_lower_" + str(node) + "_" + str(start_index))
                    
                    #lp_problem += lpSum([-x for x in in_flow_variables] + out_flow_variables)<=0, ("flow_conservation_lower_" + str(node) + "_" + str(start_index))
                sum_vars = gp.LinExpr()
                for constr_var in in_flow_variables + [-x for x in out_flow_variables]:
                    sum_vars += constr_var
                lp_problem.addConstr(sum_vars <= 0, "flow_conservation_upper_" + str(node) + "_" + str(start_index))
                #lp_problem += lpSum(in_flow_variables + [-x for x in out_flow_variables])<=0, ("flow_conservation_upper_" + str(node) + "_" + str(start_index))
            else:
                if allow_out_flow_less_than_in == False:
                    sum_vars = gp.LinExpr()
                    for constr_var in [-flows[edge_to_variables[edge][start_index]] for edge in in_edges] + [flows[edge_to_variables[edge][start_index]] for edge in out_edges]:
                        sum_vars += constr_var
                    lp_problem.addConstr(sum_vars <= 0, "flow_conservation_lower_" + str(node) + "_" + str(start_index))
                    #lp_problem += lpSum([-flows[edge_to_variables[edge][start_index]] for edge in in_edges] + [flows[edge_to_variables[edge][start_index]] for edge in out_edges])<=0, ("flow_conservation_lower_" + str(node) + "_" + str(start_index))
                sum_vars = gp.LinExpr()
                for constr_var in [flows[edge_to_variables[edge][start_index]] for edge in in_edges] + [-flows[edge_to_variables[edge][start_index]] for edge in out_edges]:
                    sum_vars += constr_var
                lp_problem.addConstr(sum_vars <= 0, "flow_conservation_upper_" + str(node) + "_" + str(start_index))
                #lp_problem += lpSum([flows[edge_to_variables[edge][start_index]] for edge in in_edges] + [-flows[edge_to_variables[edge][start_index]] for edge in out_edges])<=0, ("flow_conservation_upper_" + str(node) + "_" + str(start_index))
    
    # swap constraints
    # <= 1
    # TODO: Can trip these constraints. 
    for node in time_expanded_graph.nodes:
        # swap constraints don't affect source and target nodes
        if node in start_names or node in end_names:
            continue
            
        for out_node in time_expanded_graph.nodes[node]['out_nodes']:
            this_id = time_expanded_graph.nodes[node]['location_id']
            target_id = time_expanded_graph.nodes[out_node]['location_id']
            
            if this_id == target_id:
                continue
            
            involved_edges = []
            involved_edges.append((node, out_node))
            
            for in_node in time_expanded_graph.nodes[out_node]['in_nodes']:
                if in_node == node:
                    continue
                in_node_id = time_expanded_graph.nodes[in_node]['location_id']
                if in_node_id == target_id:
                    # this is the node we want to force into this_id location
                    for out_in_node in time_expanded_graph.nodes[in_node]['out_nodes']:
                        out_in_node_id = time_expanded_graph.nodes[out_in_node]['location_id']
                        if out_in_node_id == this_id:
                            # out_in_node is the initial node except one time step forward
                            continue
                                
                        involved_edges.append((in_node, out_in_node))
                        
                        
            if trim_impossible_variables == True:
                flow_list = []
                for edge in involved_edges:
                    for variable in edge_to_variables[edge]:
                        if variable != None:
                            flow_list.append(flows[convert_var_to_possible_var[variable]])
                sum_vars = gp.LinExpr()
                for constr_var in flow_list:
                    sum_vars += constr_var
                lp_problem.addConstr(sum_vars <= 1, "swap_constraint_" + str((node, out_node)))
                #lp_problem += lpSum(flow_list)<=1, ("swap_constraint_" + str((node, out_node)))
            else:
                flow_list = []
                for edge in involved_edges:
                    for variable in edge_to_variables[edge]:
                        flow_list.append(flows[variable])
                sum_vars = gp.LinExpr()
                for constr_var in flow_list:
                    sum_vars += constr_var
                lp_problem.addConstr(sum_vars <= 1, "swap_constraint_" + str((node, out_node)))
                #lp_problem += lpSum(flow_list)<=1, ("swap_constraint_" + str((node, out_node)))
                
    
    #lp_problem.writeLP("SwapBasedPathfinding.lp")
    lp_problem.optimize()
    
    end = time.time()
    runtime = end - start
    runtime *= 1000
    msg = "execution time: {time} ms"
    
    total_cost = 0.
    
    flow_results_by_group = None
    status_str = ""
    if lp_problem.Status == GRB.OPTIMAL:
        total_cost = lp_problem.objVal
        status_str = "Optimal"
        flow_results_dict = {}
        for v in lp_problem.getVars():
            if trim_impossible_variables == True:
                flow_results_dict[convert_possible_var_to_var[int(v.varName)]] = v.x
            else:
                flow_results_dict[int(v.varName)] = v.x
        flow_results_by_group = []
        found_dummy = False
                
        var_iter = 0
        for i in range(len(start_names)):
            flow_results_by_group.append([])
            for j in range(len(time_expanded_graph_edges)):
                if var_iter not in flow_results_dict:
                    flow_results_by_group[i].append(0)
                else:
                    flow_results_by_group[i].append(flow_results_dict[var_iter])
                var_iter += 1

                
        if found_dummy == True:
            solution_edges = []
            for index, flow_results in enumerate(flow_results_by_group):
                solution_edges.append([])
                for i in range(len(flow_results)):
                    if flow_results[i] > 0.9:
                        solution_edges[index].append(time_expanded_graph_edges[i])
                        
            print("")
            print("DUMMY FOUND - solution_edges:")
            print(str(solution_edges))
            print("")
            assert(False)
    else:
        status_str = str(lp_problem.Status)

    print("Status:", status_str)
    print("Total Cost = ", total_cost)
    print(msg.format(time=runtime))
    print()
    
    # still return the full flow results. except fill in the missing variables with 0
    return flow_results_by_group, status_str, total_cost

def multiple_group_cbs(initial_graph, source_locations_list, target_locations_list, weights=None):
    class PathStateNode:
        def __init__(self, edge_constraints_group_list, graph_pos_group_list, solutions_group_list, depths_group_list):
            self.edge_constraints_group_list = edge_constraints_group_list
            self.graph_pos_group_list = graph_pos_group_list
            self.solutions_group_list = solutions_group_list
            self.depths_group_list = depths_group_list

        def __lt__(self, other):
            return self.depths_group_list[0] < other.depths_group_list[0]

        def __le__(self,other):
            return self.depths_group_list[0] <= other.depths_group_list[0]

    # validate inputs
    if len(source_locations_list) != len(target_locations_list):
        print("Error: len(source_locations_list) != len(target_locations_list)")
    for i in range(len(source_locations_list)):
        if len(source_locations_list[i]) != len(target_locations_list[i]):
            print("Error: len(source_locations_list[" + str(i) + "]) != len(target_locations_list[" + str(i) + "])")
    
    start_multiple_group_cbs_time = time.time()
    
    conflict_priority_queue = [] 
    # find initial result
    root = PathStateNode([],[],[],[])
    for group_id in range(len(source_locations_list)):
        found_optimal = False
        solution_depth = 0
        for max_depth in range(1, 2*len(initial_graph.nodes())):
            start_name="source_"+str(group_id)
            end_name="target_"+str(group_id)
            time_expanded_graph, positions = time_expand_swap(initial_graph, max_depth, source_locations_list[group_id], target_locations_list[group_id], weights=weights, start_name=start_name, end_name=end_name)
            flow_results, status = generate_linear_program_pulp(time_expanded_graph, start_name=start_name, end_name=end_name)
            if str(status) == "Optimal":
                found_optimal = True
                solution_depth = max_depth
                break
        if found_optimal == False:
            print("Failure! Could not find an optimal path lower than or equal to depth " + str(len(initial_graph.nodes())) + ".")
        else:
            print("Success! Found optimal with depth " + str(solution_depth) + ".")
        
        time_expanded_graph_edges = [e for e in time_expanded_graph.edges]
        solution_edges = []
        for i in range(len(flow_results)):
            if flow_results[i] > 0.9:
                solution_edges.append(time_expanded_graph_edges[i])
            
        root.edge_constraints_group_list.append([])
        root.graph_pos_group_list.append(positions)
        root.solutions_group_list.append(solution_edges)
        root.depths_group_list.append(solution_depth)
    
    
    
    # recalculate paths for depths smaller than this.
    largest_makespan = max(root.depths_group_list)
    for group_id in range(len(source_locations_list)):
        if root.depths_group_list[group_id] < largest_makespan:
            time_expanded_graph, positions = time_expand_swap(initial_graph, largest_makespan, source_locations_list[group_id], target_locations_list[group_id], weights=weights, start_name="source_"+str(group_id), end_name="target_"+str(group_id))
            flow_results, status = generate_linear_program_pulp(time_expanded_graph, start_name="source_"+str(group_id), end_name="target_"+str(group_id))
            
            time_expanded_graph_edges = [e for e in time_expanded_graph.edges]
            solution_edges = []
            for i in range(len(flow_results)):
                if flow_results[i] > 0.9:
                    solution_edges.append(time_expanded_graph_edges[i])
                
            root.graph_pos_group_list[group_id] = positions
            root.solutions_group_list[group_id] = solution_edges
            root.depths_group_list[group_id] = largest_makespan
        
        
        
        
    # make all of the group makespans equal to the highest among groups.
    # do this by inserting stationary (horizontal) edges at the end of the path.
    # actually this could cause issues for collisions beyond makespan.
                
    heapq.heappush(conflict_priority_queue, (largest_makespan, root))
    found_solution = False
    while len(conflict_priority_queue) is not 0:
        makespan_node_pair = heapq.heappop(conflict_priority_queue)
        largest_makespan = makespan_node_pair[0]
        node = makespan_node_pair[1]
        time_expanded_graph, positions = time_expand_swap_multi(initial_graph, largest_makespan, source_locations_list, target_locations_list, weights=weights)
        #nx.draw(time_expanded_graph, with_labels=True, font_weight='bold', pos=positions)
        #plt.show()

        # check for collisions
        resolved_conflict = False
        for group_i in range(len(node.solutions_group_list)):
            if resolved_conflict == True:
                break
            for solution_edge in node.solutions_group_list[group_i]:
                if resolved_conflict == True:
                    break
                
                # need to check if solution edge causes illegal movement for other groups.
                # first, determine what the illegal edges are.
                # then add them as constraints to all other groups

                from_node = solution_edge[0]
                target_node = solution_edge[1]
                target_id = time_expanded_graph.nodes[target_node]['location_id']
                from_id = time_expanded_graph.nodes[from_node]['location_id']
                
                illegal_edges = []
                
                for target_in_node in time_expanded_graph.nodes[target_node]['in_nodes']:
                    if target_in_node == from_node:
                        continue
                    illegal_edges.append((target_in_node, target_node))
                    
                    target_in_node_id = time_expanded_graph.nodes[target_in_node]['location_id']
                    if target_in_node_id == target_id:
                        # target node but one time step backwards
                        for out_node in time_expanded_graph.nodes[target_in_node]['out_nodes']:
                            out_node_id = time_expanded_graph.nodes[out_node]['location_id']
                            if out_node_id == from_id:
                                # out_node is the initial node except one time step forward
                                continue
                            illegal_edges.append((target_in_node, out_node))
                for out_node in time_expanded_graph.nodes[from_node]['out_nodes']:
                    out_node_id = time_expanded_graph.nodes[out_node]['location_id']
                    if out_node_id == from_id:
                        # out_node is the initial node except one time step forward
                        for in_node in time_expanded_graph.nodes[out_node]['in_nodes']:
                            in_node_id = time_expanded_graph.nodes[in_node]['location_id']
                            if in_node_id == target_id:
                                continue
                            illegal_edges.append((in_node, out_node))
                        break
                
                found_collision = False
                
                for group_j in range(len(node.solutions_group_list)):
                    if group_i == group_j:
                        continue
                    if resolved_conflict == True:
                        break
                        
                    for illegal_edge in illegal_edges:
                        if illegal_edge in node.solutions_group_list[group_j]:
                            # collision found
                            found_collision = True
                            break
                    if found_collision == False:
                        continue
                        
                    # create 2 new nodes, each removing the others solution edge.
                    continue_with_node_1 = True
                    if isinstance(solution_edge[0], str) or isinstance(solution_edge[1], str):
                        continue_with_node_1 = False
                        print("Can't remove edge: " + str(solution_edge))
                    
                    node1 = None
                    if continue_with_node_1 == True:
                        node1 = copy.deepcopy(node)
                        node1.edge_constraints_group_list[group_i].append(solution_edge)
                    
                    
                    continue_with_node_2 = True
                    for edge in illegal_edges:
                        if isinstance(edge[0], str) or isinstance(edge[1], str):
                            continue_with_node_2 = False
                            print("Can't remove edge: " + str(edge))
                            break
                    node2 = None
                    if continue_with_node_2 == True:
                        node2 = copy.deepcopy(node)
                        for edge in illegal_edges:
                            if edge not in node2.edge_constraints_group_list[group_j]:
                                node2.edge_constraints_group_list[group_j].append(edge)
                    
                    
                    # group_i is modified in node1
                    # group_j is modified in node2
                    
                    ##################
                    # NODE 1: find new solutions
                    ########
                    if node1 != None:
                        print("Calculating new node 1")
                        found_optimal = False
                        solution_depth = 0
                        for time_depth in range(largest_makespan, 2*len(initial_graph.nodes())):
                            start_name = "source_"+str(group_i)
                            end_name = "target_"+str(group_i)
                            node_1_time_expanded_graph, node_1_positions = time_expand_swap(initial_graph, time_depth, source_locations_list[group_i], target_locations_list[group_i], weights=weights, start_name=start_name, end_name=end_name)
                            for edge_constraint in node1.edge_constraints_group_list[group_i]:
                                #### NEED to also remove in and out edges for from and to nodes.
                                if edge_constraint[1] in node_1_time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"]:
                                    node_1_time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"].remove(edge_constraint[1])
                                if edge_constraint[0] in node_1_time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"]:
                                    node_1_time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"].remove(edge_constraint[0])
                                node_1_time_expanded_graph.remove_edge(*edge_constraint)

                            node_1_flow_results, status_1 = generate_linear_program_pulp(node_1_time_expanded_graph, start_name=start_name, end_name=end_name)
                            if str(status_1) == "Optimal":
                                solution_depth = time_depth
                                found_optimal = True
                                break

                        if found_optimal == True:
                            #print("Found solution for new node 1 using graph:")
                            #nx.draw(node_1_time_expanded_graph, pos=node_1_positions, with_labels=True, font_weight='bold')
                            #plt.show()
                            # update results for group_i
                            node_1_time_expanded_graph_edges = [e for e in node_1_time_expanded_graph.edges]
                            node_1_solution_edges = []
                            for i in range(len(node_1_flow_results)):
                                if node_1_flow_results[i] > 0.9:
                                    node_1_solution_edges.append(node_1_time_expanded_graph_edges[i])

                            node1.graph_pos_group_list[group_i] = node_1_positions
                            node1.solutions_group_list[group_i] = node_1_solution_edges
                            node1.depths_group_list[group_i] = solution_depth

                            # update results for other groups
                            # TODO: can make this more efficient by simply 
                            # extending the solutions with stationary edges.
                            if solution_depth > largest_makespan:
                                # recalculate all paths
                                for group_id in range(len(source_locations_list)):
                                    if group_id == group_i:
                                        continue
                                    node_1_time_expanded_graph, node_1_positions = time_expand_swap(initial_graph, solution_depth, source_locations_list[group_id], target_locations_list[group_id], weights=weights, start_name="source_"+str(group_id), end_name="target_"+str(group_id))
                                    for edge_constraint in node1.edge_constraints_group_list[group_id]:
                                        if edge_constraint[1] in node_1_time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"]:
                                            node_1_time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"].remove(edge_constraint[1])
                                        if edge_constraint[0] in node_1_time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"]:
                                            node_1_time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"].remove(edge_constraint[0])
                                        node_1_time_expanded_graph.remove_edge(*edge_constraint)
                                    node_1_flow_results, node_1_status = generate_linear_program_pulp(node_1_time_expanded_graph, start_name="source_"+str(group_id), end_name="target_"+str(group_id))

                                    node_1_time_expanded_graph_edges = [e for e in node_1_time_expanded_graph.edges]
                                    node_1_solution_edges = []
                                    for i in range(len(node_1_flow_results)):
                                        if node_1_flow_results[i] > 0.9:
                                            node_1_solution_edges.append(node_1_time_expanded_graph_edges[i])

                                    node1.graph_pos_group_list[group_id] = node_1_positions
                                    node1.solutions_group_list[group_id] = node_1_solution_edges
                                    node1.depths_group_list[group_id] = solution_depth
                            print("Adding node1 to queue")
                            heapq.heappush(conflict_priority_queue, (solution_depth, node1))

                    
                    ##################
                    # NODE 2: find new solutions
                    ########
                    if node2 != None:
                        print("Calculating new node 2")
                        found_optimal = False
                        solution_depth = 0
                        for time_depth in range(largest_makespan, 2*len(initial_graph.nodes())):
                            node_2_time_expanded_graph, node_2_positions = time_expand_swap(initial_graph, time_depth, source_locations_list[group_j], target_locations_list[group_j], weights=weights, start_name="source_"+str(group_j), end_name="target_"+str(group_j))
                            for edge_constraint in node2.edge_constraints_group_list[group_j]:
                                if edge_constraint[1] in node_2_time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"]:
                                    node_2_time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"].remove(edge_constraint[1])
                                if edge_constraint[0] in node_2_time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"]:
                                    node_2_time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"].remove(edge_constraint[0])
                                node_2_time_expanded_graph.remove_edge(*edge_constraint)

                            node_2_flow_results, status_2 = generate_linear_program_pulp(node_2_time_expanded_graph, start_name="source_"+str(group_j), end_name="target_"+str(group_j))
                            if str(status_2) == "Optimal":
                                solution_depth = time_depth
                                found_optimal = True
                                break

                        if found_optimal == True:
                            #print("Found solution for new node 2 using graph:")
                            #nx.draw(node_2_time_expanded_graph, pos=node_2_positions, with_labels=True, font_weight='bold')
                            #plt.show()
                            # update results for group_i
                            node_2_time_expanded_graph_edges = [e for e in node_2_time_expanded_graph.edges]
                            node_2_solution_edges = []
                            for i in range(len(node_2_flow_results)):
                                if node_2_flow_results[i] > 0.9:
                                    node_2_solution_edges.append(node_2_time_expanded_graph_edges[i])

                            node2.graph_pos_group_list[group_j] = node_2_positions
                            node2.solutions_group_list[group_j] = node_2_solution_edges
                            node2.depths_group_list[group_j] = solution_depth

                            # update results for other groups
                            # TODO: can make this more efficient by simply 
                            # extending the solutions with stationary edges.
                            if solution_depth > largest_makespan:
                                # recalculate all paths
                                for group_id in range(len(source_locations_list)):
                                    if group_id == group_j:
                                        continue
                                    node_2_time_expanded_graph, node_2_positions = time_expand_swap(initial_graph, solution_depth, source_locations_list[group_id], target_locations_list[group_id], weights=weights, start_name="source_"+str(group_id), end_name="target_"+str(group_id))
                                    for edge_constraint in node2.edge_constraints_group_list[group_id]:
                                        if edge_constraint[1] in node_2_time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"]:
                                            node_2_time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"].remove(edge_constraint[1])
                                        if edge_constraint[0] in node_2_time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"]:
                                            node_2_time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"].remove(edge_constraint[0])
                                        node_2_time_expanded_graph.remove_edge(*edge_constraint)
                                    node_2_flow_results, node_2_status = generate_linear_program_pulp(node_2_time_expanded_graph, start_name="source_"+str(group_id), end_name="target_"+str(group_id))

                                    node_2_time_expanded_graph_edges = [e for e in node_2_time_expanded_graph.edges]
                                    node_2_solution_edges = []
                                    for i in range(len(node_2_flow_results)):
                                        if node_2_flow_results[i] > 0.9:
                                            node_2_solution_edges.append(node_2_time_expanded_graph_edges[i])

                                    node2.graph_pos_group_list[group_id] = node_2_positions
                                    node2.solutions_group_list[group_id] = node_2_solution_edges
                                    node2.depths_group_list[group_id] = solution_depth
                            
                            print("Adding node2 to queue")
                            print("solution_depth:", str(solution_depth))
                            heapq.heappush(conflict_priority_queue, (solution_depth, node2))

                    resolved_conflict = True
                    
        if resolved_conflict == False:
            print("Found conflict free solution!")
            found_solution = True
            
            end_multiple_group_cbs_time = time.time()
            runtime = end_multiple_group_cbs_time - start_multiple_group_cbs_time
            runtime *= 1000
            msg = "total multiple_group_cbs execution time: {time} ms"
            print(msg.format(time=runtime))
            
            # we have found a solution
            makespan = max(node.depths_group_list)
            time_expanded_graph_multi, pos = time_expand_swap_multi(initial_graph, makespan, source_locations_list, target_locations_list, weights=weights)
            
            time_expanded_graph_multi_edges = [e for e in time_expanded_graph_multi.edges]
            solution_edges = []
            widths = []
            colours = []
            
            for i in range(len(time_expanded_graph_multi_edges)):
                widths.append(1.)
                colours.append(0.)
                
            nx.draw_networkx_nodes(time_expanded_graph, pos,
                                node_color='#A0CBE2', # blue
                                node_size=400,
                                with_labels=True)
            
            nx.draw_networkx_edges(time_expanded_graph_multi, pos, with_labels=True, alpha=0.2)
            
            for group_i in range(len(node.solutions_group_list)):
                print("Solution (group " + str(group_i) + "): " + str(node.solutions_group_list[group_i]))
                cmap = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))
                widths = [3.5] * len(node.solutions_group_list[group_i])
                colours = [1.] * len(node.solutions_group_list[group_i])
                nx.draw_networkx_edges(time_expanded_graph_multi, pos, edgelist=node.solutions_group_list[group_i], with_labels=True, width=widths, edge_color=colours, alpha=0.5, edge_cmap=cmap)


            #nx.draw_networkx_edges(time_expanded_graph, pos, with_labels=True, width=widths, edge_color=colours, alpha=0.5, edge_cmap=plt.cm.Blues)
            nx.draw_networkx_labels(time_expanded_graph_multi, pos, font_size=16)
            
            plt.show(block=False)
            for group_i in range(len(node.solutions_group_list)):
                print("time_expanded_graph (group " + str(group_i) + "): ")
                time_expanded_graph, positions = time_expand_swap(initial_graph, node.depths_group_list[group_i], source_locations_list[group_i], target_locations_list[group_i], weights=weights, start_name="source_"+str(group_i), end_name="target_"+str(group_i))
                for edge_constraint in node.edge_constraints_group_list[group_i]:
                    if edge_constraint[1] in time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"]:
                        time_expanded_graph.nodes[edge_constraint[0]]["out_nodes"].remove(edge_constraint[1])
                    if edge_constraint[0] in time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"]:
                        time_expanded_graph.nodes[edge_constraint[1]]["in_nodes"].remove(edge_constraint[0])
                    time_expanded_graph.remove_edge(*edge_constraint)
                nx.draw(time_expanded_graph, pos=positions, with_labels=True, font_weight='bold')
                plt.show(block=False)
                plt.clf()
            break

            # need to time expand graph including all group's sources and targets.
            # then make edges coloured according to each group solution.
    if found_solution == False:
        print("")
        print("Failed to find a solution.")
      

def multiple_group_commodity(initial_graph, source_locations_list, target_locations_list, weights=None, time_limit=1000., output_solution_graph = True, use_single_group_first = False, allow_out_flow_less_than_in = False, trim_impossible_variables = False, problem_name = None, use_gurobi = False, approximate_min_time_depth = False):
    if weights == None:
        weights = {}
        edge_iter = 1
        for edge in initial_graph.edges():
            weights[edge] = 1.0
            edge_iter += 1

    #nx.draw(initial_graph, with_labels=True, font_weight='bold')
    
    start_full = True
    if use_single_group_first == True:
        source_same_group = [[t for x in source_locations_list for t in x]]
        target_same_group = [[t for x in target_locations_list for t in x]]
        if len(source_locations_list) == 1:
            start_full = True
            print("Starting multi group computation.")
        else:
            start_full = False
            print("Using single group to first find min time depth.")
    
    start_multiple_group_commodity_time = time.time()
    
    start_locations = source_locations_list
    end_locations = target_locations_list
    solution_depth = 0
    starting_depth = 1
    if approximate_min_time_depth == True:
        starting_depth = approximate_time_depth_lower_bound(initial_graph, source_locations_list, target_locations_list)
    time_out = False
    for depth in range(starting_depth, 2*len(initial_graph.nodes())):
        print("time_depth:", depth-1)
        if start_full == False:
            start_names = []
            end_names = []
            for i in range(len(source_same_group)):
                start_names.append("source_"+str(i))
                end_names.append("target_"+str(i))
            time_expanded_graph_multi, pos = time_expand_swap_multi(initial_graph, depth, source_same_group, target_same_group, weights=weights)
            current_time = time.time()
            runtime = current_time - start_multiple_group_commodity_time
            if runtime >= time_limit:
                time_out = True
                break
            if use_gurobi == True:
                flow_results_by_group, status, optimal_cost = generate_linear_program_gurobi_multi(time_expanded_graph_multi, start_names, end_names, time_limit = time_limit-runtime, allow_out_flow_less_than_in=allow_out_flow_less_than_in, trim_impossible_variables=trim_impossible_variables)
            else:
                flow_results_by_group, status, optimal_cost = generate_linear_program_pulp_multi(time_expanded_graph_multi, start_names, end_names, time_limit = time_limit-runtime, allow_out_flow_less_than_in=allow_out_flow_less_than_in, trim_impossible_variables=trim_impossible_variables)
            if str(status) == "Optimal":
                start_full = True
                print("Starting multi group computation (time depth: " + str(depth-1) + ")")
        
        if start_full == True:
            start_names = []
            end_names = []
            for i in range(len(start_locations)):
                start_names.append("source_"+str(i))
                end_names.append("target_"+str(i))
            time_expanded_graph_multi, pos = time_expand_swap_multi(initial_graph, depth, start_locations, end_locations, weights=weights)
            current_time = time.time()
            runtime = current_time - start_multiple_group_commodity_time
            if runtime >= time_limit:
                time_out = True
                break
            if use_gurobi == True:
                flow_results_by_group, status, optimal_cost = generate_linear_program_gurobi_multi(time_expanded_graph_multi, start_names, end_names, time_limit = time_limit-runtime, allow_out_flow_less_than_in=allow_out_flow_less_than_in, trim_impossible_variables=trim_impossible_variables)
            else:
                flow_results_by_group, status, optimal_cost = generate_linear_program_pulp_multi(time_expanded_graph_multi, start_names, end_names, time_limit = time_limit-runtime, allow_out_flow_less_than_in=allow_out_flow_less_than_in, trim_impossible_variables=trim_impossible_variables)
                
            if str(status) == "Optimal":
                found_optimal = True
                solution_depth = depth
                break
                
    if time_out == True:
        print("Warning: Computation timed out (1000 sec limit - " + str(runtime) + " sec)")
        return 0, solution_depth-1, runtime, time_out
    if found_optimal == False:
        print("Failure! Could not find an optimal path lower than or equal to depth " + str(len(initial_graph.nodes())) + ".")
    else:
        print("Success! Found optimal with time depth " + str(solution_depth-1) + ".")
        
        end_multiple_group_commodity_time = time.time()
        runtime = end_multiple_group_commodity_time - start_multiple_group_commodity_time
        msg = "total multiple_group_commodity execution time: {time} ms"
        print(msg.format(time=runtime))
        if output_solution_graph == True:
            time_expanded_graph_edges = [e for e in time_expanded_graph_multi.edges]
            
            solution_edges = []
            for index, flow_results in enumerate(flow_results_by_group):
                solution_edges.append([])
                for i in range(len(flow_results)):
                    if flow_results[i] > 0.9:
                        solution_edges[index].append(time_expanded_graph_edges[i])
            
            plt.figure(3, figsize=(solution_depth + 3, 3*len(initial_graph.nodes())/4))
            
            nx.draw_networkx_nodes(time_expanded_graph_multi, pos,
                    node_color='#A0CBE2', # blue
                    node_size=400)    
            nx.draw_networkx_labels(time_expanded_graph_multi, pos)    
            
            nx.draw_networkx_edges(time_expanded_graph_multi, pos, alpha=0.2)
            nx.draw_networkx_edge_labels(time_expanded_graph_multi, pos)
            
            for group_i in range(len(solution_edges)):
                print("Solution (group " + str(group_i) + "): " + str(solution_edges[group_i]))
                cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))
                widths = [3.5] * len(solution_edges[group_i])
                colours = [1.] * len(solution_edges[group_i])
                nx.draw_networkx_edges(time_expanded_graph_multi, pos, edgelist=solution_edges[group_i], edge_color=colours, width=widths, alpha=0.8, edge_cmap=cmap)
                nx.draw_networkx_edge_labels(time_expanded_graph_multi, pos)
            
            nx.draw_networkx_labels(time_expanded_graph_multi, pos, font_size=16)
            #plt.show(block=False)
            
            if problem_name == None:
                problem_name = "solution-path"
            else:
                problem_name = "solution-path-" + problem_name
            plt.savefig(problem_name + "-" + str(int(round(random.random()*1000000))) + ".png", format="png")
            plt.savefig(problem_name + ".svg", format="svg")
            
            plt.clf()
        
    return optimal_cost, solution_depth-1, runtime, time_out
    
def compare_optimisation_runtimes():
    # loop over a bunch of agent counts and run 10 times each. Use 1000 sec runtime limit.
    # then order lists in results to be increasing runtime
    max_agent_count = 53
    
    pos = None
    initial_graph = relabel.convert_node_labels_to_integers(nx.grid_2d_graph(8, 8))
    #initial_graph, pos = create_ibmq_poughkeepsie_hardware_graph()
    #initial_graph, pos = create_ibmq_rochester_hardware_graph()
    #initial_graph, pos = create_ibmq_melbourne_hardware_graph()
    #initial_graph, pos = create_rigetti_acorn_hardware_graph()
    #initial_graph, pos = create_ibmq_paris_hardware_graph()
    #initial_graph = relabel.convert_node_labels_to_integers(nx.barbell_graph(4, 2))
    #initial_graph = relabel.convert_node_labels_to_integers(nx.barbell_graph(3, 0))

    weights = {}
    for edge in initial_graph.edges():
        weights[edge] = 1.0 # + random.random() * 25.0
        
    if pos != None:
        nx.draw(initial_graph, pos=pos, 
                    node_color='#A0CBE2', # blue
                    node_size=400,
                    with_labels=True)
    else:
        nx.draw(initial_graph, 
                    node_color='#A0CBE2', # blue
                    node_size=400,
                    with_labels=True)
                    
    #plt.show(block=False)
    plt.savefig("initial-graph.png", format="png")
    plt.savefig("initial-graph.svg", format="svg")
    plt.clf()


    number_of_iterations = 1
    #max_number_of_agents = 
    times = []
    number_of_qubits = 53
    individual_qubits = True
    single_qubit_group = False
    costs_normal = []
    costs_optimised = []

    times = []
    agent_count = 5
    dead_node_count = 0
    # uniformly sample from log with mean and range
    error_log10_std = 0.5
    error_mean = 0.001
    max_agent_count = 64
    
    for i in range(number_of_iterations):
        random.seed(i+23)
        np.random.seed(i+23)
        weights = {}
        for edge in initial_graph.edges():
            weights[edge] = -1.0 + random.random() * 25.0
            
        if individual_qubits == True:
            split_points = list(range(1, number_of_qubits))
        else:
            if single_qubit_group == True:
                split_points = []
            else:
                split_points = np.random.choice(number_of_qubits - 2, random.randint(2, number_of_qubits - 1) - 1, replace=False) + 1
                split_points.sort()
        
        start_nodes = [x for x in random.sample(list(range(len(initial_graph.nodes()))), number_of_qubits)]
        start_nodes = np.split(start_nodes, split_points)
        
        end_nodes = [x for x in random.sample(list(range(len(initial_graph.nodes()))), number_of_qubits)]
        end_nodes = np.split(end_nodes, split_points)
        print("start nodes:", start_nodes)
        print("end nodes:", end_nodes)
        
        optimal_cost, runtime = multiple_group_commodity(initial_graph, start_nodes, end_nodes, time_limit = 1000, problem_name = "normal", weights=weights, use_single_group_first = False, allow_out_flow_less_than_in = True, trim_impossible_variables = True, use_gurobi = True, approximate_min_time_depth = False)
        costs_normal.append(optimal_cost)
        times.append(runtime)
    
    normal_average_time = statistics.mean(times)

    from scipy.stats import truncnorm
    def get_truncated_normal(mean=0, sd=1, low=0, upp=1):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
            
    for i in range(number_of_iterations):
        random.seed(i+23)
        np.random.seed(i+23)
        weights = {}
        for edge in initial_graph.edges():
            X = np.random.normal(math.log(error_mean, 10), error_log10_std)
            weights[edge] = -3*math.log(1 - math.pow(10, X))
        
        
        if individual_qubits == True:
            split_points = list(range(1, agent_count))
        else:
            if single_qubit_group == True:
                split_points = []
            else:
                split_points = np.random.choice(agent_count - 2, random.randint(2, agent_count - 1) - 1, replace=False) + 1
                split_points.sort()
        
        start_nodes = [x for x in random.sample(list(range(len(initial_graph.nodes()))), agent_count)]
        start_nodes = np.split(start_nodes, split_points)
        
        end_nodes = [x for x in random.sample(list(range(len(initial_graph.nodes()))), agent_count)]
        end_nodes = np.split(end_nodes, split_points)
        print("start nodes:", start_nodes)
        print("end nodes:", end_nodes)
        
        optimal_cost, runtime = multiple_group_commodity(initial_graph, start_nodes, end_nodes, weights=weights, time_limit = 1000, problem_name = "optimised", use_single_group_first = False, allow_out_flow_less_than_in = True, trim_impossible_variables = True, use_gurobi = True, approximate_min_time_depth = True)
        
        print("total error: " + str(1.0 - 1.0/math.exp(optimal_cost)))
        errors = []
        for weight in weights.values():
            errors.append(1.0 - 1.0/math.exp(weight))
        print("weights:", str(errors))
        costs_optimised.append(optimal_cost)
        times.append(runtime)

    optimise_average_time = statistics.mean(times)
    
    msg = "NORMAL - Average time over " + str(number_of_iterations) + " iterations (multiple_group_commodity execution): {time} ms"
    print(msg.format(time=normal_average_time))
    print("Optimal costs:")
    print(str(costs_normal))
    
    msg = "OPTIMISED - Average time over " + str(number_of_iterations) + " iterations (multiple_group_commodity execution): {time} ms"
    print(msg.format(time=optimise_average_time))
    print("Optimal costs:")
    print(str(costs_optimised))


###############################################################
####
#######################################
####
###############################################################

def main():

    # Start at min dist.
    results = {}
    results["experiment_layout"] = "ibmq_melbourne" #"grid_8x8"
    results["experiment_variables"] = ["error_log10_stds"] # which parameters change
    results["runtimes"] = []
    results["total_costs"] = []
    results["total_errors"] = []
    results["depths"] = []
    results["node_counts"] = []
    results["agent_counts"] = []
    results["split_counts"] = []
    results["dead_node_counts"] = []
    results["error_means"] = []
    results["error_log10_stds"] = []
    
    
    pos = None
    #initial_graph = relabel.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5))
    #initial_graph = relabel.convert_node_labels_to_integers(nx.grid_2d_graph(8, 8))
    #initial_graph, pos = create_ibmq_poughkeepsie_hardware_graph()
    #initial_graph, pos = create_ibmq_rochester_hardware_graph()
    initial_graph, pos = create_ibmq_melbourne_hardware_graph()
    #initial_graph, pos = create_rigetti_acorn_hardware_graph()
    #initial_graph, pos = create_ibmq_paris_hardware_graph()
    #initial_graph = relabel.convert_node_labels_to_integers(nx.barbell_graph(4, 2))
    #initial_graph = relabel.convert_node_labels_to_integers(nx.barbell_graph(3, 0))

    weights = {}
    for edge in initial_graph.edges():
        weights[edge] = 1.0 # + random.random() * 25.0
        
    if pos != None:
        nx.draw(initial_graph, pos=pos, 
                    node_color='#A0CBE2', # blue
                    node_size=400,
                    with_labels=True)
    else:
        nx.draw(initial_graph, 
                    node_color='#A0CBE2', # blue
                    node_size=400,
                    with_labels=True)
                
    #plt.show(block=False)
    plt.savefig("initial-graph.png", format="png")
    plt.savefig("initial-graph.svg", format="svg")
    plt.clf()

    results_directory = "results-ibmq_melbourne-increasing-error_stds"
    
    max_agent_count = 12
    max_split_count = 24
    
    number_of_iterations = 10
    individual_qubits = False
    single_qubit_group = False

    dead_node_count = 0
    error_log10_std = 0.5 # normal sample from log with mean and std
    error_mean = 0.001
    
    instance_iter = 0 #26 * number_of_iterations
    split_count = 8 # None
    number_of_agents = max_agent_count
    #for number_of_agents in range(1, max_agent_count+1):
    #for split_count in range(0, max_split_count+1):
    for std_iter in range(0, 1):
        error_log10_std = std_iter / 100.0
        print("")
        print("Agent Count:", str(number_of_agents))
        print("error_log10_std:", str(error_log10_std))
        #print("Split Count:", str(split_count))
        for i in range(number_of_iterations):
            print("")
            print("Agent Count:", str(number_of_agents))
            print("error_log10_std:", str(error_log10_std))
            #print("Split Count:", str(split_count))
            print("Iteration:", str(i))
            instance_iter += 1
            #random.seed(instance_iter)
            #np.random.seed(instance_iter)
            weights = {}
            for edge in initial_graph.edges():
                cont = True
                while cont == True:
                    if error_log10_std != 0:
                        X = get_truncated_normal(math.log(error_mean, 10), error_log10_std, -10, 0).rvs()
                    else:
                        X = math.log(error_mean, 10)
                    #X = np.random.normal(math.log(error_mean, 10), error_log10_std)
                    
                    if X < 0:
                        cont = False
                if X == 0:
                    print("X is 0")
                    
                weights[edge] = -3*math.log(1 - math.pow(10, X))
            
            print("")
            print("weights:", str(weights))
            
            
            if individual_qubits == True:
                split_points = list(range(1, number_of_agents))
            else:
                if single_qubit_group == True or number_of_agents == 1:
                    split_points = []
                else:
                    if split_count != None:
                        split_points = np.random.choice(number_of_agents - 1, split_count, replace=False) + 1
                        split_points.sort()
                    else:
                        if number_of_agents == 2:
                            if bool(random.getrandbits(1)) == True:
                                split_points = [1]
                            else:
                                split_points = []
                        else:
                            split_points = np.random.choice(number_of_agents - 2, random.randint(2, number_of_agents - 1) - 1, replace=False) + 1
                            split_points.sort()
            
            start_nodes = [x for x in random.sample(list(range(len(initial_graph.nodes()))), number_of_agents)]
            start_nodes = np.split(start_nodes, split_points)
            
            end_nodes = [x for x in random.sample(list(range(len(initial_graph.nodes()))), number_of_agents)]
            end_nodes = np.split(end_nodes, split_points)
            print("start nodes:", start_nodes)
            print("end nodes:", end_nodes)
            
            optimal_cost, depth, runtime, time_out = multiple_group_commodity(initial_graph, start_nodes, end_nodes, weights=weights, time_limit = 1000, output_solution_graph = True, problem_name = "optimised", use_single_group_first = False, allow_out_flow_less_than_in = True, trim_impossible_variables = True, use_gurobi = True, approximate_min_time_depth = True)
            if time_out == False:
                results["total_costs"].append(optimal_cost)
                results["total_errors"].append(1.0 - 1.0/math.exp(optimal_cost))
                results["depths"].append(depth)
                results["runtimes"].append(runtime*1000)
                results["node_counts"].append(15)
                results["agent_counts"].append(number_of_agents)
                results["split_counts"].append(len(split_points))
                results["dead_node_counts"].append(dead_node_count)
                results["error_log10_stds"].append(error_log10_std)
                results["error_means"].append(error_mean)
            
                print("total error: " + str(1.0 - 1.0/math.exp(optimal_cost)))
                errors = []
                for weight in weights.values():
                    errors.append(1.0 - 1.0/math.exp(weight))
                print("weights:", str(errors))
                
                save_dict_as_json(results, results_directory)

    ## order dictionary in order of runtime.
    #sorted_indices = np.argsort(results["runtimes"])
    #sorted_results = {}
    #sorted_results["experiment_layout"] = "grid_8x8"
    #sorted_results["experiment_variables"] = ["agent_counts"] # which parameters change
    #sorted_results["runtimes"] = []
    #sorted_results["total_costs"] = []
    #sorted_results["total_errors"] = []
    #sorted_results["node_counts"] = []
    #sorted_results["agent_counts"] = []
    #sorted_results["split_counts"] = []
    #sorted_results["dead_node_counts"] = []
    #sorted_results["error_means"] = []
    #sorted_results["error_log10_stds"] = []
    
    #for index in sorted_indices:
    #    sorted_results["total_costs"].append(results["total_costs"][index])
    #    sorted_results["total_errors"].append(results["total_errors"][index])
    #    sorted_results["runtimes"].append(results["runtimes"][index])
    #    sorted_results["node_counts"].append(results["node_counts"][index])
    #    sorted_results["agent_counts"].append(results["agent_counts"][index])
    #    sorted_results["split_counts"].append(results["split_counts"][index])
    #    sorted_results["dead_node_counts"].append(results["dead_node_counts"][index])
    #    sorted_results["error_log10_stds"].append(results["error_log10_stds"][index])
    #    sorted_results["error_means"].append(results["error_means"][index])
    #    
    #    
    #save_dict_as_json(sorted_results, results_directory + "-sorted")

if __name__ == '__main__':
    main()