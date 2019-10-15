# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:56:51 2019

@author: btt1

Signal Tampering Problem

"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pulp as plp
import time

def extract_decision_variables(network_data):

    intersections = []; intersection_variables = []
    for i in range(len(data)):
        if network_data.iloc[i]['type'] == 'intflow':
            if network_data.iloc[i]['start'].isalpha():
                intersections.append(network_data.iloc[i]['start'])
                intersection_variables.append(network_data.iloc[i]['name'])
            elif network_data.iloc[i]['end'].isalpha():
                intersections.append(network_data.iloc[i]['end'])
                intersection_variables.append(network_data.iloc[i]['name'])
    intersections = np.unique(np.array(intersections))
    intersection_variables = np.unique(np.array(intersection_variables))

    ends = []; end_flow_variables = []
    for i in range(len(network_data)):
        if network_data.iloc[i]['type'] == 'end':
            end = network_data.iloc[i]['start'].split('_')[0]
            ends.append(end)
            end_flow_variables.append('y'+end+'S')
    ends = np.unique(np.array(ends))
    end_flow_variables = np.unique(np.array(end_flow_variables))
    
    start_nodes = []; end_nodes = []
    for i in range(len(network_data)):
        if network_data.iloc[i]['type'] == 'start':
            node = network_data.iloc[i]['start'].split('_')[0]
            start_nodes.append(node)
        elif network_data.iloc[i]['type'] == 'end':
            node = network_data.iloc[i]['start'].split('_')[0]
            end_nodes.append(node)
    start_nodes = np.unique(np.array(start_nodes))
    end_nodes = np.unique(np.array(end_nodes))

    return intersections, intersection_variables, ends, end_flow_variables, start_nodes, end_nodes

def create_graph_singletimestep(t, data):
    
    g = nx.DiGraph(timestep=t)
    for i in range(len(data)):
        row = data.iloc[i]
        start = str(row['start'])+'_{}'.format(t); end = str(row['end'])+'_{}'.format(t)
        name = row['name'] + '_' + str(t)
        g.add_edge(start, end, edge_type=row['type'], edge_name=name)
    
    return g

def create_edge_betweengraphs(ts, G):
    
    nodes = np.array(list(G.nodes()))
    for t in range(ts-1):
        condition = [i.split('_')[-1]==str(t) and i.split('_')[-2]=='d' for i in nodes]
        start_nodes = np.extract(condition, nodes)
        for each in start_nodes:
            start = each
            end = each.split('_')[0]+'_s_'+str(t+1)
            name = 'x' + start.split('_')[0] + '_' + str(t+1)
            G.add_edge(start, end, edge_type='occ', edge_name=name)
    
    return G

def create_supergraph(ts, data, start_nodes, end_nodes):

    print("\n----- Super graph -----\n")
    directed_graphs = [];
    print("\tCreating individual timestep graphs...")
    for t in range(ts):
        g = create_graph_singletimestep(t, data)
        directed_graphs.append(g)
    G = nx.DiGraph(name='SuperGraph')
    print("\tCreating Supergraph...")
    for g in directed_graphs:
        G = nx.union(G, g)
    G = create_edge_betweengraphs(ts, G); 
    G.add_node('S')
    for i_start, start_cell in enumerate(start_nodes):
        source_node = 'R'+str(int(i_start)+1)
        start_node = start_cell + '_d'
        G.add_node(source_node)
        for t in range(ts):
            start = start_node + '_' + str(t)
            name_1 = 'y' + source_node + str(int(i_start)) + start_node.split('_')[0] + '_' + str(t)
            G.add_edge(source_node, start, edge_type='flow', edge_name=name_1)
    for end_cell in end_nodes:
        end_node = end_cell + '_s'
        for t in range(ts):
            end = end_node + '_' + str(t)
            name_2 = 'y' + end_node.split('_')[0] + 'S' + '_' + str(t)
            G.add_edge(end, 'S', edge_type='flow', edge_name=name_2)
    
    print("\tSupergraph created!")
    print("\t", nx.info(G), "\n")
    
    return G, directed_graphs

def create_opt_formulation_constants(G, cost, demand_source, demand_sink, slack_bound, occupancy_bound, flow_bound, edge_list, node_list):
    
    A = np.array(nx.incidence_matrix(G, oriented=True).todense())
    cost_vector = np.array([float(i.split("_")[-1]) + cost if i[0]=='x' else 0.0 for i in edge_list[:,2]])
    demand_vector = np.array([-demand_source if 'R' in i else demand_sink if i=='S' else 0.0 for i in node_list])
    bound_vector = np.array([occupancy_bound if i[0]=='x' else slack_bound if i[0]=='s' else flow_bound for i in edge_list[:,2]])
    
    return A, cost_vector, demand_vector, bound_vector
    
def solve_optimal_assignment(A, d, u, c, edge_list):
    
    s = time.time()
    prob = None
    print("\n----- Optimal Assignment Problem -----\n")
    print("\tCreating new problem instance...")
    prob = plp.LpProblem("Opt_assignment_problem", plp.LpMinimize)
    print("\tAdding super-graph variables...")
    flows = {i:plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=u[i], name=str(edge_list[:,2][i])) for i in range(A.shape[1])}
    print("\tAdding constraints..."); percent_complete = 0
    for j in range(A.shape[0]):
#        prob += plp.LpAffineExpression([(flows[i],A[j,i]) for i in range(A.shape[1])]) == d[j]
        prob += plp.lpSum(A[j,i]*flows[i] for i in range(A.shape[1])) == d[j]
        if (j/A.shape[0])*100 > percent_complete:
            print("\t\t{} % of constraints added".format(percent_complete))
            percent_complete += 10
    e1 = time.time()
    print("\tConstraints added. Total time took: ", int((e1-s)/60), "mins")
    objective = plp.lpSum([c[i]*flows[i] for i in range(A.shape[1])])
    prob.setObjective(objective)
    prob.writeLP("Opt_assignment_problem.lp")
    print("\tSolving the optimal assignment problem...")
    prob.solve(solver=plp.GUROBI_CMD())
    print("\tSolution status: ", plp.LpStatus[prob.status])
    print("\tObjective function value: ", plp.value(prob.objective))
    solution = pd.DataFrame(columns=['Variable','OptimalValue'])
    for i,v in enumerate(prob.variables()):
        solution.loc[i] = [v.name, v.varValue]
    solution.to_csv("./Optimal_solution.csv"); print("\tSolutions saved.\n"); e2 = time.time()
    print("\tTotal time took for solving the optimal assignment: ", int((e2-s)/60), "mins")
    
    return prob, flows, solution

def extract_intersection_flows(F, intersection_variables, sort=True):
    
    i = 0
    intersection_flows = pd.DataFrame(columns=["Intersection","Var_id","Variables","Timesteps","OptimalValue"])
    for var_id, var in F.items():
        if var.name.split("_")[0] in intersection_variables:
            intersection_flows.loc[i] = [var.name.split("_")[0], float(var_id), var.name, int(var.name.split("_")[-1]), var.varValue]
            i += 1
    if sort:
        intersection_flows.sort_values("Var_id", ascending=True, inplace=True)
    
    return intersection_flows


def extract_end_flows(F, end_flow_variables):
    
    i = 0
    end_flows = pd.DataFrame(columns=["Ends","Endpoint","Var_id","Variables","Timesteps","OptimalValue"])
    for var_id, var in F.items():
        if var.name.split("_")[0] in end_flow_variables:
            e = ""
            for j in var.name.split("_")[0]:
                if j.isdigit():
                    e += j
            end_flows.loc[i] = [e, var.name.split("_")[0], float(var_id), var.name, int(var.name.split("_")[-1]), var.varValue]
            i += 1
    end_flows.sort_values("Timesteps", ascending=True, inplace=True)
    
    return end_flows

def extract_z1_variable_index(F, end_flow_variables):
    
    end_flows_index = []
    for var_id, var in F.items():
        if var.name.split("_")[0] in end_flow_variables:
            end_flows_index.append(float(var_id))
    
    return np.array(end_flows_index)

def extract_z2_surrogate_variable_index(F):
    
    int_flows_index = []
    for var_id, var in F.items():
        if str(var).split("_")[0] == "pp":
            int_flows_index.append(float(var_id))
        elif str(var).split("_")[0] == "pm":
            int_flows_index.append(float(var_id))
        else:
            int_flows_index.append(-1)
#        if var.name.split("_")[0] in intersection_variables:
#            int_flows_index.append(float(var_id))
    
    return np.array(int_flows_index)

def extract_z2_variable_index(F, intersection_variables):
    
    int_flows_index = []
    for var_id, var in F.items():
        if var.name.split("_")[0] in intersection_variables:
            int_flows_index.append(float(var_id))
    
    return np.array(int_flows_index)

def optimal_assignment_results(t, f, intersection_variables, end_flow_variables):
    
    opt_end_flows = extract_end_flows(f, end_flow_variables)
    opt_int_flows = extract_intersection_flows(f, intersection_variables)
    Po = np.array(opt_int_flows[opt_int_flows["Timesteps"] < t]["OptimalValue"].tolist())
    Cum_Qo = 0
    for end_var in end_flow_variables:
        y_opt = np.array(opt_end_flows[opt_end_flows['Endpoint']==end_var]['OptimalValue'].tolist())
        y_cum_opt = np.cumsum(y_opt)
        Cum_Qo += np.sum(y_cum_opt)
    
    return Po, Cum_Qo

def plot_intersection_flows(int_flows, intersection_variables, assign_problem):
    
    figures = []
    for intr in intersections:
        fig = plt.figure(figsize=(8,5)); leg = []
        plt.title("{} flow at intersection: {}".format(assign_problem, intr))
        for int_var in intersection_variables:
            if intr in int_var:
                y = int_flows[int_flows['Intersection']==int_var]['OptimalValue'].tolist()
                x = int_flows[int_flows['Intersection']==int_var]['Timesteps'].tolist()
                leg.append(int_var)
                plt.plot(x, y)
        plt.xlabel("Time steps", fontsize=12); plt.ylabel("Flows", fontsize=12)
        plt.legend(leg); plt.grid(); plt.show()
        figures.append(fig)
        plt.close()
        
    return figures

def plot_end_flows(end_flows, end_flow_variables, assign_problem):
    
    figures = []
    for endpt in ends:
        fig = plt.figure(figsize=(8,5)); leg = []
        plt.title("{} end flows from the position: {}".format(assign_problem, endpt), fontsize=13)
        for end_var in end_flow_variables:
            y = opt_end_flows[opt_end_flows['Endpoint']==end_var]['OptimalValue'].tolist()
            x = opt_end_flows[opt_end_flows['Endpoint']==end_var]['Timesteps'].tolist()
            plt.plot(x, y)
            leg.append(end_var)
        plt.xlabel("Time steps", fontsize=12); plt.ylabel("Flows", fontsize=12)
        plt.legend(leg); plt.grid(); plt.show()
        figures.append(fig)
        plt.close()
        
    return figures

def create_adv_problem_variables(A, u, edge_list, int_variables_pertimestep, l_int):
    
    print("\n----- Adversarial Signal Tampering ------\n")
    print("\tAdding variables to the adversarial problem...")

    adv_flows = {i:plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=u[i], name=str(edge_list[:,2][i])) for i in range(A.shape[1])}
    Pp = {len(adv_flows)+i:plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=None, name='pp_'+str(int_variables_pertimestep[i])) for i in range(l_int)}
    new_flows = {**adv_flows, **Pp}
    Pm = {len(new_flows)+i:plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=None, name='pm_'+str(int_variables_pertimestep[i])) for i in range(l_int)}
    new_flows.update(Pm)
    S1 = {len(new_flows)+i:plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=None, name='s1_'+str(int_variables_pertimestep[i])) for i in range(l_int)}
    new_flows.update(S1)
    S2 = {len(new_flows)+i:plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=None, name='s2_'+str(int_variables_pertimestep[i])) for i in range(l_int)}
    new_flows.update(S2)
    S3 = {len(new_flows)+i:plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=None, name='s3_'+str(int_variables_pertimestep[i])) for i in range(l_int)}
    new_flows.update(S3)
    S4 = {len(new_flows)+i:plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=None, name='s4_'+str(int_variables_pertimestep[i])) for i in range(l_int)}
    new_flows.update(S4)
    F = new_flows

    return F
    
def create_adv_formulation_constants(adv_f, F, A, d, Po, l_int, end_variables_pertimestep):
    
    print("\tAdding formulation constants...")

    flow_variables = []
    for v_id, v in adv_f.items():
        flow_variables.append(str(v))
    flow_variables = np.array(flow_variables)
    
    If = np.zeros((int(l_int), len(adv_f)))
    for row in range(If.shape[0]):
        var = str(int_variables_pertimestep[row])
        If[row] = np.where(flow_variables == var, 1, 0)
    
    m = int(A.shape[0] + 5*l_int); n = int(A.shape[1] + 6*l_int)
    I = np.identity(int(l_int)); M = np.zeros((m,n))
    
    M[:A.shape[0], :A.shape[1]] = A
    M[A.shape[0]:A.shape[0]+If.shape[0], :If.shape[1]] = If
    M[A.shape[0]:A.shape[0]+If.shape[0], If.shape[1]:If.shape[1]+I.shape[1]] = I
    M[A.shape[0]:A.shape[0]+If.shape[0], If.shape[1]+I.shape[1]:If.shape[1]+2*I.shape[1]] = -I
    M[A.shape[0]+If.shape[0]:A.shape[0]+If.shape[0]+I.shape[0], If.shape[1]:If.shape[1]+I.shape[1]] = I
    M[A.shape[0]+If.shape[0]:A.shape[0]+If.shape[0]+I.shape[0], If.shape[1]+2*I.shape[1]:If.shape[1]+3*I.shape[1]] = -I
    M[A.shape[0]+If.shape[0]+I.shape[0]:A.shape[0]+If.shape[0]+2*I.shape[0], If.shape[1]+I.shape[1]:If.shape[1]+2*I.shape[1]] = I
    M[A.shape[0]+If.shape[0]+I.shape[0]:A.shape[0]+If.shape[0]+2*I.shape[0], If.shape[1]+3*I.shape[1]:If.shape[1]+4*I.shape[1]] = -I
    M[A.shape[0]+If.shape[0]+2*I.shape[0]:A.shape[0]+If.shape[0]+3*I.shape[0], :If.shape[1]] = If
    M[A.shape[0]+If.shape[0]+2*I.shape[0]:A.shape[0]+If.shape[0]+3*I.shape[0], If.shape[1]+4*I.shape[1]:If.shape[1]+5*I.shape[1]] = I
    M[A.shape[0]+2*If.shape[0]+2*I.shape[0]:A.shape[0]+3*If.shape[0]+2*I.shape[0], :If.shape[1]] = -If
    M[A.shape[0]+2*If.shape[0]+2*I.shape[0]:A.shape[0]+3*If.shape[0]+2*I.shape[0], If.shape[1]+5*I.shape[1]:If.shape[1]+6*I.shape[1]] = I
    
    D = np.zeros(m)
    D[:d.shape[0]] = d
    D[d.shape[0]:d.shape[0]+I.shape[0]] = Po
    D[d.shape[0]+3*I.shape[0]:d.shape[0]+4*I.shape[0]] = 1 + Po
    D[d.shape[0]+4*I.shape[0]:d.shape[0]+5*I.shape[0]] = 1 - Po
    
    end_flow_indicators = []
    for var_id, var in F.items():
        if str(var) in end_variables_pertimestep:
            end_flow_indicators.append(1)
        else:
            end_flow_indicators.append(0)
    cost_vector1 = np.array(end_flow_indicators)
    
    int_flow_indicators = []
    for var_id, var in F.items():
        if str(var).split("_")[0] == "pp":
            int_flow_indicators.append(1)
        elif str(var).split("_")[0] == "pm":
            int_flow_indicators.append(1)
        else:
            int_flow_indicators.append(0)
    cost_vector2 = np.array(int_flow_indicators)
    
    return M, D, cost_vector1, cost_vector2

def objective_function_1(Cum_Qo, cost_vector1, F, M):

    Qa = []
    for i in range(M.shape[1]):
        try:
           e = float(str(cost_vector1[i]*F[i]))
           if e != 0.0:
               Qa.append(cost_vector1[i]*F[i])
        except ValueError:
            Qa.append(cost_vector1[i]*F[i])
    Cum_Qa = []; j = 0
    while j < len(Qa):
        Cum_Qa.append(plp.lpSum(Qa[:j+1]))
        j += 1
    objective_fn1 = Cum_Qo - plp.lpSum(Cum_Qa)
    
    return -objective_fn1

def objective_function_2(cost_vector2, F, M):
    
    objective_fn2 = plp.lpSum([cost_vector2[i]*F[i] for i in range(M.shape[1])])
    
    return objective_fn2

def objective_function_1_value(Cum_Qo, F, i_ends):
    
    y_adv = []
    for i in i_ends:
        y_adv_i = F[i].varValue
        y_adv.append(y_adv_i)
    y_cum_adv = np.cumsum(y_adv)
    Cum_Qa = np.sum(y_cum_adv)
    obj1_val = round(Cum_Qo - Cum_Qa)
    print("\t\tObjective function 1 value: ", obj1_val)
    
    return -obj1_val

def objective_function_2_value(F, cost_vector2, i_ints):
    
    solution_list = []
    for i in i_ints:
        if i != -1:
            solution_list.append(F[i].varValue)
        else:
            solution_list.append(0)
    solution_array = np.array(solution_list)
    obj2_val = round(np.dot(cost_vector2, solution_array))
    print("\t\tObjective function 2 value: ", obj2_val)
    
    return obj2_val

def calculate_total_num_sig_changes(Po, i_ints_realvar, F):
    
    num_sig_changes = []
    for i in i_ints_realvar:
        num_changes = F[i].varValue
        num_sig_changes.append(num_changes)
    changes = np.count_nonzero(np.array(num_sig_changes) - Po)
    print("\t\tNumber of signal changes: ", changes)
    
    return changes

def weighted_obj_funcs(w1, w2, Cum_Qo, cost_vector1, cost_vector2, F, M):
    
    mod_objective_fn = w1*objective_function_1(Cum_Qo, cost_vector1, F, M) + w2*objective_function_2(cost_vector2, F, M)
    
    return mod_objective_fn

def create_base_problem(M, F, D):
    
    base_problem = None
    base_problem = plp.LpProblem("Adv_assignment_problem_base") 
    print("\tCreating adversarial assignment base problem...")
    s = time.time(); print("\tAdding adversarial problem constraints...")
    percent_complete = 0
    for j in range(M.shape[0]):
        base_problem += plp.lpSum(M[j,i]*F[i] for i in range(M.shape[1])) == D[j]
        if (j/M.shape[0])*100 > percent_complete:
            print("\t\t{} % of constraints added".format(percent_complete))
            percent_complete += 10
    e = time.time();
    print("\tConstraints added. Total time took = {} mins".format(round((e-s)/60)))
    
    return base_problem

def modify_problem(base_problem, k, opt_sense, constraint_lhs=None, constraint_rhs=None):
    
    problem_instance = None
    problem_instance = base_problem.copy()
    if constraint_lhs is not None:
        problem_instance += constraint_lhs == constraint_rhs
    
    return problem_instance

def solve_problem(base_problem, opt_sense, k, objective_fn, F, constraint_lhs=None, constraint_rhs=None):
    
    problem_instance = modify_problem(base_problem, opt_sense,  k, constraint_lhs, constraint_rhs)
    problem_instance.setObjective(objective_fn)
    problem_instance.sense = opt_sense
    problem_instance.writeLP("Adv_problem_instance_{}.lp".format(0)) # ideally k
    problem_instance.solve(solver=plp.GUROBI_CMD())
    if plp.LpStatus[problem_instance.status] != "Optimal":
        raise("\tNo optimal solution found.\n Optimization ceased!")
    else:
        z = plp.value(problem_instance.objective)
    solution_df = pd.DataFrame(columns=['Variable','OptimalValue']);
#    for i,v in F.items():
#        solution_df.loc[i] = [v.name, v.varValue]
    
    return z, solution_df

def solve_extreme_objectivefunctions(base_problem, cost_vector1, cost_vector2, Cum_Qo, Po, end_flow_variables, intersection_variables, M, F, D):
    
    k = 1
    objs = {}; solns = {}
#    base_problem = create_base_problem(M, F, D)
    i_ends = extract_z1_variable_index(F, end_flow_variables)
    i_ints = extract_z2_surrogate_variable_index(F)
    i_ints_realvar = extract_z2_variable_index(F, intersection_variables)
    print("\tSolving extreme objective solutions...")
    
    objective_fn1 = objective_function_1(Cum_Qo, cost_vector1, F, M)
    z, soln_df = solve_problem(base_problem, plp.LpMinimize, k, objective_fn1, F, constraint_lhs=None, constraint_rhs=None)
    obj1_val = objective_function_1_value(Cum_Qo, F, i_ends)
    obj2_val = objective_function_2_value(F, cost_vector2, i_ints)
    
    const_lhs = objective_function_1(Cum_Qo, cost_vector1, F, M); const_rhs = obj1_val
    objective_fn2 = objective_function_2(cost_vector2, F, M)
    z, soln_df = solve_problem(base_problem, plp.LpMinimize, k, objective_fn2, F, constraint_lhs=const_lhs, constraint_rhs=const_rhs)
    obj1_val = objective_function_1_value(Cum_Qo, F, i_ends)
    obj2_val = objective_function_2_value(F, cost_vector2, i_ints)
    num_sig_changes = calculate_total_num_sig_changes(Po, i_ints_realvar, F)
    objs[str(k)] = np.array([obj1_val, obj2_val]); solns[str(k)] = np.array([num_sig_changes, -obj1_val])
    print("\tStep {}: New Solution! Objective function {}".format(k, objs[str(k)]))
    
    k += 1
    
    objective_fn2 = objective_function_2(cost_vector2, F, M)
    z, soln_df = solve_problem(base_problem, plp.LpMinimize, k, objective_fn2, F, constraint_lhs=None, constraint_rhs=None)
    obj1_val = objective_function_1_value(Cum_Qo, F, i_ends)
    obj2_val = objective_function_2_value(F, cost_vector2, i_ints)
    
    const_lhs = objective_function_2(cost_vector2, F, M); const_rhs = obj2_val
    objective_fn1 = objective_function_1(Cum_Qo, cost_vector1, F, M)
    z, soln_df = solve_problem(base_problem, plp.LpMinimize, k, objective_fn1, F, constraint_lhs=const_lhs, constraint_rhs=const_rhs)
    obj1_val = objective_function_1_value(Cum_Qo, F, i_ends)
    obj2_val = objective_function_2_value(F, cost_vector2, i_ints)
    num_sig_changes = calculate_total_num_sig_changes(Po, i_ints_realvar, F)
    objs[str(k)] = np.array([obj1_val, obj2_val]); solns[str(k)] = np.array([num_sig_changes, -obj1_val])
    print("\tStep {}: New Solution! Objective function {}".format(k, objs[str(k)]))
    
    return base_problem, objs, solns, k

def step0(base_problem, M, F, D, cost_vector1, cost_vector2, Cum_Qo, Po, end_flow_variables, intersection_variables):
    
    L = []; E = [];
    base_problem, objs, solns, k = solve_extreme_objectivefunctions(base_problem, cost_vector1, cost_vector2, Cum_Qo, Po, end_flow_variables, intersection_variables, M, F, D)
    if np.array_equal(objs[str(k)], objs[str(k-1)]):
        return objs, solns, L, E, k
    else:
        L.append([str(k-1), str(k)])
    
    return base_problem, objs, solns, L, E, k
    
def step1(base_problem, M, F, D, cost_vector1, cost_vector2, Cum_Qo, Po, end_flow_variables, intersection_variables, objs, solns, L, E, k):
    
    
    i_ends = extract_z1_variable_index(F, end_flow_variables)
    i_ints = extract_z2_surrogate_variable_index(F)
    i_ints_realvar = extract_z2_variable_index(F, intersection_variables)
    while len(L) != 0:
        [r,s] = L[0]; k += 1
        w1 = np.abs(objs[s][1] - objs[r][1])
        w2 = np.abs(objs[s][0] - objs[r][0])        
        wt_objective_fn = weighted_obj_funcs(w1, w2, Cum_Qo, cost_vector1, cost_vector2, F, M)
        z, soln_df = solve_problem(base_problem, plp.LpMinimize, k, wt_objective_fn, F, constraint_lhs=None, constraint_rhs=None)
        
        const_lhs = wt_objective_fn; const_rhs = z
        objective_fn1 = objective_function_1(Cum_Qo, cost_vector1, F, M)
        z, soln_df = solve_problem(base_problem, plp.LpMinimize, k, objective_fn1, F, constraint_lhs=const_lhs, constraint_rhs=const_rhs)
        wt_obj1_val = objective_function_1_value(Cum_Qo, F, i_ends)
        wt_obj2_val = objective_function_2_value(F, cost_vector2, i_ints)
        wt_obj_array = np.append(np.array(np.round(wt_obj1_val)), np.array(np.round(wt_obj2_val)))
        num_sig_changes = calculate_total_num_sig_changes(Po, i_ints_realvar, F)

        if np.array_equal(wt_obj_array, objs[r]) or np.array_equal(wt_obj_array, objs[s]):
            E.append([r,s])
            L.remove([r,s])
            print("\tStep {}: Found same solution !!!".format(k))
        else:
            L.remove([r,s])
            L.append([r,str(k)]); L.append([str(k),s])
            objs[str(k)] = wt_obj_array; solns[str(k)] = np.array([num_sig_changes, -np.round(wt_obj1_val)])
            print("\tStep {}: New Solution! Objective function {}".format(k, objs[str(k)]))
            
    print("\tSearch space is empty!")
    print("\tPareto optimization search finished.\n")
    
    return objs, solns, L, E, k

def pareto_solution_algorithm(base_problem, M, F, D, cost_vector1, cost_vector2, Cum_Qo, Po, end_flow_variables, intersection_variables):
    
    base_problem, objs, solns, L, E, k = step0(base_problem, M, F, D, cost_vector1, cost_vector2, Cum_Qo, Po, end_flow_variables, intersection_variables)
    objs, solns, L, E, k = step1(base_problem, M, F, D, cost_vector1, cost_vector2, Cum_Qo, Po, end_flow_variables, intersection_variables, objs, solns, L, E, k)
    
    return objs, solns

def pareto_solution(objs):

    objectives = np.zeros((len(objs), 2)); j = 0
    for i,v in objs.items():
        objectives[j,:] = np.abs(objs[i]); j += 1

    return objectives

def plot_cum_endflow(F, ends, end_flow_variables):
    
    figures = []
    for end_point in ends:
        adv_end_flows = extract_end_flows(F, end_flow_variables)
        adv_end_flows = adv_end_flows[adv_end_flows['Ends'] == end_point]
        opt_end_flows = extract_end_flows(F, end_flow_variables)
        opt_end_flows = opt_end_flows[opt_end_flows['Ends'] == end_point]
        fig = plt.figure(figsize=(8,6));
        plt.title("Flows from the cell {}".format(end_point), fontsize=14); plt.grid()
        for end_var in end_flow_variables:
            y1 = opt_end_flows[opt_end_flows['Endpoint']==end_var]['OptimalValue'].tolist()
            y2 = adv_end_flows[adv_end_flows['Endpoint']==end_var]['OptimalValue'].tolist()
            x = adv_end_flows[adv_end_flows['Endpoint']==end_var]['Timesteps'].tolist()
            plt.plot(x, np.cumsum(y1))
            plt.plot(x, np.cumsum(y2))
        plt.xlabel("Time steps", fontsize=12); plt.xticks(fontsize=12)
        plt.ylabel("Flows", fontsize=12); plt.yticks(fontsize=12)
        plt.legend(['Optimal flows', 'Adversarial flows'], fontsize=11)
#        plt.savefig('End_flow_{}'.format(end_point))
        figures.append(fig)
        plt.close()
    
    return figures

# --------------------------------------------------------------------------- #
# -------------------------- Optimal Assignment ----------------------------- #

expt_no = int(input('Experiment number = '))
ts = int(input('Enter number of timesteps = ')); #ts = 80
cost = 1.0
number_demands = int(input('Enter number of source nodes = '))
demand_source = float(int(input('Enter demand at each source node = '))); #demand_source = 5.0
demand_sink = float(int(input('Enter demand at the sink node = '))); #demand_sink = 20.0
if number_demands*demand_source != demand_sink:
    raise Exception('FeasibilityError: Aggregated demand at source nodes not matching with demand at the sink node ')
slack_bound = 5.0
occupancy_bound = 5.0
flow_bound = 1.0
t = ts

cwd = os.getcwd()
os.chdir(cwd)
data_filename = input('Enter the network edges filename: ')
data = pd.read_excel(data_filename + '.xlsx')
intersections, intersection_variables, ends, end_flow_variables, start_nodes, end_nodes = extract_decision_variables(data)
G, dir_gs = create_supergraph(ts, data, start_nodes, end_nodes)
edge_list = np.array(list(G.edges(data='edge_name')))
node_list = np.array(list(G.nodes()))

A, c, d, u = create_opt_formulation_constants(G, cost, demand_source, demand_sink, slack_bound, occupancy_bound, flow_bound, edge_list, node_list)
opt_problem, f, opt_solution = solve_optimal_assignment(A, d, u, c, edge_list)
opt_int_flows = extract_intersection_flows(f, intersection_variables)
opt_end_flows = extract_end_flows(f, end_flow_variables)
Po, Cum_Qo = optimal_assignment_results(t, f, intersection_variables, end_flow_variables)
int_flow_figures = plot_intersection_flows(opt_int_flows, intersection_variables, 'Optimal')
end_flow_figures = plot_end_flows(opt_end_flows, end_flow_variables, 'Optimal')
#Qo = np.array(opt_end_flows[opt_end_flows["Timesteps"] < t]["OptimalValue"].tolist())
#opt_policy = signal_control_policy(opt_int_flows, t)
#opt_endflow_vector = end_flow_vector(opt_end_flows, t)


# --------------------------------------------------------------------------- #
# -------------------------- adversarial assignment ------------------------- #
int_variables_pertimestep = opt_int_flows[opt_int_flows["Timesteps"] < t]["Variables"].tolist()
end_variables_pertimestep = opt_end_flows[opt_end_flows["Timesteps"] < t]["Variables"].tolist()
#Po = np.array(opt_int_flows[opt_int_flows["Timesteps"] < t]["OptimalValue"].tolist())
l_int = len(int_variables_pertimestep)
l_end = len(end_variables_pertimestep)

F = create_adv_problem_variables(A, u, edge_list, int_variables_pertimestep, l_int)
M, D, cost_vector1, cost_vector2 = create_adv_formulation_constants(f, F, A, d, Po, l_int, end_variables_pertimestep)
adversarial_base_problem = create_base_problem(M, F, D)
objs, solns = pareto_solution_algorithm(adversarial_base_problem, M, F, D, cost_vector1, cost_vector2, Cum_Qo, Po, end_flow_variables, intersection_variables)
objectives = pareto_solution(objs)
pareto_frontier_values = np.zeros((len(objs), 2)); i = 0
for values in solns.values():
    pareto_frontier_values[i,:] = values; i += 1

# -------------------------------------------------------------------------- #
# --------------------------- Saving offline -------------------------------- #
opt_assignment_results = {'Po':Po, 'Cum_Qo':Cum_Qo}
opt_assignment_figures = {'Intersection_flows':int_flow_figures, 'End_flows':end_flow_figures}
np.save('Optimal_assignment_results_{}.npy'.format(expt_no), [opt_assignment_results, opt_assignment_figures])
np.save('Adversarial_assignment_results_{}.npy'.format(expt_no), [objs, solns, pareto_frontier_values])

plt.figure(figsize=(10,8))
plt.title("Optimal Pareto Frontier", fontsize=14)
plt.scatter(pareto_frontier_values[:,0], pareto_frontier_values[:,1])
plt.grid()
plt.xlabel("Number of Signal changes", fontsize=12)
plt.ylabel("Difference in the number of cumulative flows", fontsize=12)
plt.savefig("./Optimal_Pareto_Frontier_{}.png".format(expt_no))
# --------------------------------------------------------------------------- #