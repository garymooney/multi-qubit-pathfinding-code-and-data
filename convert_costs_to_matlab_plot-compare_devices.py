import json
import os
import sys
import numpy as np
import statistics

def load_dict_from_json(filename):
    try:
        fh = open(filename, 'r')
    except FileNotFoundError:
        print("could not find:", filename)
        return None
    print("found:", filename)
        
    with open(filename) as f:
        return json.load(f)

output_filename = 'matlab_plot_results_costs_compare_devices.m'
if os.path.exists(output_filename):
    os.remove(output_filename)
        
def write_line_to_file(text = '', output_filename=output_filename):
    try: 
        with open(output_filename, 'a') as file:
            file.write(text + "\n")

    except IOError:
        with open(output_filename, 'w') as file:
            file.write(text + "\n")
        
def generate_variable_string(variable_name, values):
    variable_string = str(variable_name) + " = ["
    add_space = False
    for value in values:
        if add_space == True:
            variable_string += " "
        else:
            add_space = True
        variable_string += str(value)
    variable_string += "];"
    return variable_string
     
results_list = []
results_list.append(load_dict_from_json("results-grid-8x8-increasing-independent-agents.txt"))
results_list.append(load_dict_from_json("results-ibmq_rochester-increasing-individual-agents.txt"))
results_list.append(load_dict_from_json("results-ibmq_paris-increasing-individual-agents.txt"))
results_list.append(load_dict_from_json("results-rigetti_acorn-increasing-individual-agents.txt"))
results_list.append(load_dict_from_json("results-ibmq_poughkeepsie-increasing-individual-agents.txt"))
results_list.append(load_dict_from_json("results-ibmq_melbourne-increasing-individual-agents.txt"))

results_sorted_list = []

for results in results_list:
    sorted_indices = np.argsort(results["runtimes"])
    results_sorted = {}
    results_sorted["experiment_layout"] = results["experiment_layout"]
    results_sorted["experiment_variables"] = results["experiment_variables"]
    results_sorted["runtimes"] = []
    results_sorted["total_costs"] = []
    results_sorted["total_errors"] = []
    results_sorted["depths"] = []
    results_sorted["node_counts"] = []
    results_sorted["agent_counts"] = []
    results_sorted["split_counts"] = []
    results_sorted["dead_node_counts"] = []
    results_sorted["error_means"] = []
    results_sorted["error_log10_stds"] = []
    
    for index in sorted_indices:
        results_sorted["runtimes"].append(results["runtimes"][index])
        results_sorted["total_costs"].append(results["total_costs"][index])
        results_sorted["total_errors"].append(results["total_errors"][index])
        results_sorted["node_counts"].append(results["node_counts"][index])
        results_sorted["agent_counts"].append(results["agent_counts"][index])
        results_sorted["split_counts"].append(results["split_counts"][index])
        results_sorted["dead_node_counts"].append(results["dead_node_counts"][index])
        results_sorted["error_means"].append(results["error_means"][index])
        results_sorted["error_log10_stds"].append(results["error_log10_stds"][index])
    
    results_sorted_list.append(results_sorted)
    
grid_independent_benchmark_results_sorted = results_sorted_list[0]
ibmq_rochester_independent_benchmark_results_sorted = results_sorted_list[1]
ibmq_paris_independent_benchmark_results_sorted = results_sorted_list[2]
rigetti_acorn_independent_benchmark_results_sorted = results_sorted_list[3]
ibmq_poughkeepsie_independent_benchmark_results_sorted = results_sorted_list[4]
ibmq_melbourne_independent_benchmark_results_sorted = results_sorted_list[5]

results_sorted_total_error_list = []

for results in results_list:
    sorted_indices = np.argsort(results["total_errors"])
    results_sorted = {}
    results_sorted["experiment_layout"] = results["experiment_layout"]
    results_sorted["experiment_variables"] = results["experiment_variables"]
    results_sorted["runtimes"] = []
    results_sorted["total_costs"] = []
    results_sorted["total_errors"] = []
    results_sorted["node_counts"] = []
    results_sorted["agent_counts"] = []
    results_sorted["split_counts"] = []
    results_sorted["dead_node_counts"] = []
    results_sorted["error_means"] = []
    results_sorted["error_log10_stds"] = []
    
    for index in sorted_indices:
        results_sorted["runtimes"].append(results["runtimes"][index])
        results_sorted["total_costs"].append(results["total_costs"][index])
        results_sorted["total_errors"].append(results["total_errors"][index])
        results_sorted["node_counts"].append(results["node_counts"][index])
        results_sorted["agent_counts"].append(results["agent_counts"][index])
        results_sorted["split_counts"].append(results["split_counts"][index])
        results_sorted["dead_node_counts"].append(results["dead_node_counts"][index])
        results_sorted["error_means"].append(results["error_means"][index])
        results_sorted["error_log10_stds"].append(results["error_log10_stds"][index])
    
    results_sorted_total_error_list.append(results_sorted)
    
grid_total_error_results_sorted = results_sorted_total_error_list[0]
ibmq_rochester_total_error_results_sorted = results_sorted_total_error_list[1]
ibmq_paris_total_error_results_sorted = results_sorted_total_error_list[2]
rigetti_acorn_total_error_results_sorted = results_sorted_total_error_list[3]
ibmq_poughkeepsie_total_error_results_sorted = results_sorted_total_error_list[4]
ibmq_melbourne_total_error_results_sorted = results_sorted_total_error_list[5]
    
results_sorted_depths_list = []

for results in results_list:
    sorted_indices = np.argsort(results["depths"])
    results_sorted = {}
    results_sorted["experiment_layout"] = results["experiment_layout"]
    results_sorted["experiment_variables"] = results["experiment_variables"]
    results_sorted["runtimes"] = []
    results_sorted["total_costs"] = []
    results_sorted["total_errors"] = []
    results_sorted["depths"] = []
    results_sorted["node_counts"] = []
    results_sorted["agent_counts"] = []
    results_sorted["split_counts"] = []
    results_sorted["dead_node_counts"] = []
    results_sorted["error_means"] = []
    results_sorted["error_log10_stds"] = []
    
    for index in sorted_indices:
        results_sorted["runtimes"].append(results["runtimes"][index])
        results_sorted["total_costs"].append(results["total_costs"][index])
        results_sorted["total_errors"].append(results["total_errors"][index])
        results_sorted["depths"].append(results["depths"][index])
        results_sorted["node_counts"].append(results["node_counts"][index])
        results_sorted["agent_counts"].append(results["agent_counts"][index])
        results_sorted["split_counts"].append(results["split_counts"][index])
        results_sorted["dead_node_counts"].append(results["dead_node_counts"][index])
        results_sorted["error_means"].append(results["error_means"][index])
        results_sorted["error_log10_stds"].append(results["error_log10_stds"][index])
        
    results_sorted_depths_list.append(results_sorted)
        
grid_depths_results_sorted = results_sorted_depths_list[0]
ibmq_rochester_depths_results_sorted = results_sorted_depths_list[1]
ibmq_paris_depths_results_sorted = results_sorted_depths_list[2]
rigetti_acorn_depths_results_sorted = results_sorted_depths_list[3]
ibmq_poughkeepsie_depths_results_sorted = results_sorted_depths_list[4]
ibmq_melbourne_depths_results_sorted = results_sorted_depths_list[5]

results_agent_counts = []
results_average_runtimes = []
results_average_total_errors = []
results_average_depths = []

for results in results_list:
    average_runtimes = []
    average_total_errors = []
    average_depths = []
    agent_counts_set = set()
    for agent_count in results["agent_counts"]:
        agent_counts_set.add(agent_count)
    agent_counts = sorted(list(agent_counts_set))

    for agent_count in agent_counts:
        runtimes = []
        total_errors = []
        depths = []
        for i in range(len(results["runtimes"])):
            if results["agent_counts"][i] == agent_count:
                runtimes.append(results["runtimes"][i])
                total_errors.append(results["total_errors"][i])
                depths.append(results["depths"][i])
                
        average_runtimes.append(statistics.mean(runtimes))
        average_total_errors.append(statistics.mean(total_errors))
        average_depths.append(statistics.mean(depths))
    
    results_agent_counts.append(agent_counts)
    results_average_runtimes.append(average_runtimes)
    results_average_total_errors.append(average_total_errors)
    results_average_depths.append(average_depths)
  
grid_agent_counts = results_agent_counts[0]
ibmq_rochester_agent_counts = results_agent_counts[1]
ibmq_paris_agent_counts = results_agent_counts[2]
rigetti_acorn_agent_counts = results_agent_counts[3]
ibmq_poughkeepsie_agent_counts = results_agent_counts[4]
ibmq_melbourne_agent_counts = results_agent_counts[5]
  
grid_average_runtimes = results_average_runtimes[0]
ibmq_rochester_average_runtimes = results_average_runtimes[1]
ibmq_paris_average_runtimes = results_average_runtimes[2]
rigetti_acorn_average_runtimes = results_average_runtimes[3]
ibmq_poughkeepsie_average_runtimes = results_average_runtimes[4]
ibmq_melbourne_average_runtimes = results_average_runtimes[5]

grid_average_total_errors = results_average_total_errors[0]
ibmq_rochester_average_total_errors = results_average_total_errors[1]
ibmq_paris_average_total_errors = results_average_total_errors[2]
rigetti_acorn_average_total_errors = results_average_total_errors[3]
ibmq_poughkeepsie_average_total_errors = results_average_total_errors[4]
ibmq_melbourne_average_total_errors = results_average_total_errors[5]

grid_average_depths = results_average_depths[0]
ibmq_rochester_average_depths = results_average_depths[1]
ibmq_paris_average_depths = results_average_depths[2]
rigetti_acorn_average_depths = results_average_depths[3]
ibmq_poughkeepsie_average_depths = results_average_depths[4]
ibmq_melbourne_average_depths = results_average_depths[5]
    
x_instances_independent_string = generate_variable_string("x_instances_independent", [i+1 for i in range(len(grid_independent_benchmark_results_sorted["runtimes"]))])
y_runtimes_independent_string = generate_variable_string("y_runtimes_independent", grid_independent_benchmark_results_sorted["runtimes"])
x_instances_ibmq_rochester_string = generate_variable_string("x_instances_ibmq_rochester", [i+1 for i in range(len(ibmq_rochester_independent_benchmark_results_sorted["runtimes"]))])
y_runtimes_ibmq_rochester_string = generate_variable_string("y_runtimes_ibmq_rochester", ibmq_rochester_independent_benchmark_results_sorted["runtimes"])
x_instances_ibmq_paris_string = generate_variable_string("x_instances_ibmq_paris", [i+1 for i in range(len(ibmq_paris_independent_benchmark_results_sorted["runtimes"]))])
y_runtimes_ibmq_paris_string = generate_variable_string("y_runtimes_ibmq_paris", ibmq_paris_independent_benchmark_results_sorted["runtimes"])
x_instances_rigetti_acorn_string = generate_variable_string("x_instances_rigetti_acorn", [i+1 for i in range(len(rigetti_acorn_independent_benchmark_results_sorted["runtimes"]))])
y_runtimes_rigetti_acorn_string = generate_variable_string("y_runtimes_rigetti_acorn", rigetti_acorn_independent_benchmark_results_sorted["runtimes"])
x_instances_ibmq_poughkeepsie_string = generate_variable_string("x_instances_ibmq_poughkeepsie", [i+1 for i in range(len(ibmq_poughkeepsie_independent_benchmark_results_sorted["runtimes"]))])
y_runtimes_ibmq_poughkeepsie_string = generate_variable_string("y_runtimes_ibmq_poughkeepsie", ibmq_poughkeepsie_independent_benchmark_results_sorted["runtimes"])
x_instances_ibmq_melbourne_string = generate_variable_string("x_instances_ibmq_melbourne", [i+1 for i in range(len(ibmq_melbourne_independent_benchmark_results_sorted["runtimes"]))])
y_runtimes_ibmq_melbourne_string = generate_variable_string("y_runtimes_ibmq_melbourne", ibmq_melbourne_independent_benchmark_results_sorted["runtimes"])

x_instances_total_error_grid_string = generate_variable_string("x_instances_total_error_grid", [i+1 for i in range(len(grid_total_error_results_sorted["total_errors"]))])
y_total_error_grid_string = generate_variable_string("y_total_error_grid", grid_total_error_results_sorted["total_errors"])
x_instances_total_error_ibmq_rochester_string = generate_variable_string("x_instances_total_error_ibmq_rochester", [i+1 for i in range(len(ibmq_rochester_total_error_results_sorted["total_errors"]))])
y_total_error_ibmq_rochester_string = generate_variable_string("y_total_error_ibmq_rochester", ibmq_rochester_total_error_results_sorted["total_errors"])
x_instances_total_error_ibmq_paris_string = generate_variable_string("x_instances_total_error_ibmq_paris", [i+1 for i in range(len(ibmq_paris_total_error_results_sorted["total_errors"]))])
y_total_error_ibmq_paris_string = generate_variable_string("y_total_error_ibmq_paris", ibmq_paris_total_error_results_sorted["total_errors"])
x_instances_total_error_rigetti_acorn_string = generate_variable_string("x_instances_total_error_rigetti_acorn", [i+1 for i in range(len(rigetti_acorn_total_error_results_sorted["total_errors"]))])
y_total_error_rigetti_acorn_string = generate_variable_string("y_total_error_rigetti_acorn", rigetti_acorn_total_error_results_sorted["total_errors"])
x_instances_total_error_ibmq_poughkeepsie_string = generate_variable_string("x_instances_total_error_ibmq_poughkeepsie", [i+1 for i in range(len(ibmq_poughkeepsie_total_error_results_sorted["total_errors"]))])
y_total_error_ibmq_poughkeepsie_string = generate_variable_string("y_total_error_ibmq_poughkeepsie", ibmq_poughkeepsie_total_error_results_sorted["total_errors"])
x_instances_total_error_ibmq_melbourne_string = generate_variable_string("x_instances_total_error_ibmq_melbourne", [i+1 for i in range(len(ibmq_melbourne_total_error_results_sorted["total_errors"]))])
y_total_error_ibmq_melbourne_string = generate_variable_string("y_total_error_ibmq_melbourne", ibmq_melbourne_total_error_results_sorted["total_errors"])



x_instances_depths_grid_string = generate_variable_string("x_instances_depths_grid", [i+1 for i in range(len(grid_depths_results_sorted["depths"]))])
y_depths_grid_string = generate_variable_string("y_depths_grid", grid_depths_results_sorted["depths"])
x_instances_depths_ibmq_rochester_string = generate_variable_string("x_instances_depths_ibmq_rochester", [i+1 for i in range(len(ibmq_rochester_depths_results_sorted["depths"]))])
y_depths_ibmq_rochester_string = generate_variable_string("y_depths_ibmq_rochester", ibmq_rochester_depths_results_sorted["depths"])
x_instances_depths_ibmq_paris_string = generate_variable_string("x_instances_depths_ibmq_paris", [i+1 for i in range(len(ibmq_paris_depths_results_sorted["depths"]))])
y_depths_ibmq_paris_string = generate_variable_string("y_depths_ibmq_paris", ibmq_paris_depths_results_sorted["depths"])
x_instances_depths_rigetti_acorn_string = generate_variable_string("x_instances_depths_rigetti_acorn", [i+1 for i in range(len(rigetti_acorn_depths_results_sorted["depths"]))])
y_depths_rigetti_acorn_string = generate_variable_string("y_depths_rigetti_acorn", rigetti_acorn_depths_results_sorted["depths"])
x_instances_depths_ibmq_poughkeepsie_string = generate_variable_string("x_instances_depths_ibmq_poughkeepsie", [i+1 for i in range(len(ibmq_poughkeepsie_depths_results_sorted["depths"]))])
y_depths_ibmq_poughkeepsie_string = generate_variable_string("y_depths_ibmq_poughkeepsie", ibmq_poughkeepsie_depths_results_sorted["depths"])
x_instances_depths_ibmq_melbourne_string = generate_variable_string("x_instances_depths_ibmq_melbourne", [i+1 for i in range(len(ibmq_melbourne_depths_results_sorted["depths"]))])
y_depths_ibmq_melbourne_string = generate_variable_string("y_depths_ibmq_melbourne", ibmq_melbourne_depths_results_sorted["depths"])



x_grid_agent_counts_string = generate_variable_string("x_grid_agent_counts", grid_agent_counts)
x_ibmq_rochester_agent_counts_string = generate_variable_string("x_ibmq_rochester_agent_counts", ibmq_rochester_agent_counts)
x_ibmq_paris_agent_counts_string = generate_variable_string("x_ibmq_paris_agent_counts", ibmq_paris_agent_counts)
x_rigetti_acorn_agent_counts_string = generate_variable_string("x_rigetti_acorn_agent_counts", rigetti_acorn_agent_counts)
x_ibmq_poughkeepsie_agent_counts_string = generate_variable_string("x_ibmq_poughkeepsie_agent_counts", ibmq_poughkeepsie_agent_counts)
x_ibmq_melbourne_agent_counts_string = generate_variable_string("x_ibmq_melbourne_agent_counts", ibmq_melbourne_agent_counts)

y_grid_average_runtimes_string = generate_variable_string("y_grid_average_runtimes", grid_average_runtimes)
y_ibmq_rochester_average_runtimes_string = generate_variable_string("y_ibmq_rochester_average_runtimes", ibmq_rochester_average_runtimes)
y_ibmq_paris_average_runtimes_string = generate_variable_string("y_ibmq_paris_average_runtimes", ibmq_paris_average_runtimes)
y_rigetti_acorn_average_runtimes_string = generate_variable_string("y_rigetti_acorn_average_runtimes", rigetti_acorn_average_runtimes)
y_ibmq_poughkeepsie_average_runtimes_string = generate_variable_string("y_ibmq_poughkeepsie_average_runtimes", ibmq_poughkeepsie_average_runtimes)
y_ibmq_melbourne_average_runtimes_string = generate_variable_string("y_ibmq_melbourne_average_runtimes", ibmq_melbourne_average_runtimes)

y_grid_average_total_errors_string = generate_variable_string("y_grid_average_total_errors", grid_average_total_errors)
y_ibmq_rochester_average_total_errors_string = generate_variable_string("y_ibmq_rochester_average_total_errors", ibmq_rochester_average_total_errors)
y_ibmq_paris_average_total_errors_string = generate_variable_string("y_ibmq_paris_average_total_errors", ibmq_paris_average_total_errors)
y_rigetti_acorn_average_total_errors_string = generate_variable_string("y_rigetti_acorn_average_total_errors", rigetti_acorn_average_total_errors)
y_ibmq_poughkeepsie_average_total_errors_string = generate_variable_string("y_ibmq_poughkeepsie_average_total_errors", ibmq_poughkeepsie_average_total_errors)
y_ibmq_melbourne_average_total_errors_string = generate_variable_string("y_ibmq_melbourne_average_total_errors", ibmq_melbourne_average_total_errors)

y_grid_average_depths_string = generate_variable_string("y_grid_average_depths", grid_average_depths)
y_ibmq_rochester_average_depths_string = generate_variable_string("y_ibmq_rochester_average_depths", ibmq_rochester_average_depths)
y_ibmq_paris_average_depths_string = generate_variable_string("y_ibmq_paris_average_depths", ibmq_paris_average_depths)
y_rigetti_acorn_average_depths_string = generate_variable_string("y_rigetti_acorn_average_depths", rigetti_acorn_average_depths)
y_ibmq_poughkeepsie_average_depths_string = generate_variable_string("y_ibmq_poughkeepsie_average_depths", ibmq_poughkeepsie_average_depths)
y_ibmq_melbourne_average_depths_string = generate_variable_string("y_ibmq_melbourne_average_depths", ibmq_melbourne_average_depths)


write_line_to_file("")
write_line_to_file(x_instances_independent_string)
write_line_to_file(x_instances_ibmq_rochester_string)
write_line_to_file(x_instances_ibmq_paris_string)
write_line_to_file(x_instances_rigetti_acorn_string)
write_line_to_file(x_instances_ibmq_poughkeepsie_string)
write_line_to_file(x_instances_ibmq_melbourne_string)
write_line_to_file(y_runtimes_independent_string)
write_line_to_file(y_runtimes_ibmq_rochester_string)
write_line_to_file(y_runtimes_ibmq_paris_string)
write_line_to_file(y_runtimes_rigetti_acorn_string)
write_line_to_file(y_runtimes_ibmq_poughkeepsie_string)
write_line_to_file(y_runtimes_ibmq_melbourne_string)


write_line_to_file(x_instances_total_error_grid_string)
write_line_to_file(y_total_error_grid_string)
write_line_to_file(x_instances_total_error_ibmq_rochester_string)
write_line_to_file(y_total_error_ibmq_rochester_string)
write_line_to_file(x_instances_total_error_ibmq_paris_string)
write_line_to_file(y_total_error_ibmq_paris_string)
write_line_to_file(x_instances_total_error_rigetti_acorn_string)
write_line_to_file(y_total_error_rigetti_acorn_string)
write_line_to_file(x_instances_total_error_ibmq_poughkeepsie_string)
write_line_to_file(y_total_error_ibmq_poughkeepsie_string)
write_line_to_file(x_instances_total_error_ibmq_melbourne_string)
write_line_to_file(y_total_error_ibmq_melbourne_string)

write_line_to_file(x_instances_depths_grid_string)
write_line_to_file(y_depths_grid_string)
write_line_to_file(x_instances_depths_ibmq_rochester_string)
write_line_to_file(y_depths_ibmq_rochester_string)
write_line_to_file(x_instances_depths_ibmq_paris_string)
write_line_to_file(y_depths_ibmq_paris_string)
write_line_to_file(x_instances_depths_rigetti_acorn_string)
write_line_to_file(y_depths_rigetti_acorn_string)
write_line_to_file(x_instances_depths_ibmq_poughkeepsie_string)
write_line_to_file(y_depths_ibmq_poughkeepsie_string)
write_line_to_file(x_instances_depths_ibmq_melbourne_string)
write_line_to_file(y_depths_ibmq_melbourne_string)



write_line_to_file(x_grid_agent_counts_string)
write_line_to_file(x_ibmq_rochester_agent_counts_string)
write_line_to_file(x_ibmq_paris_agent_counts_string)
write_line_to_file(x_rigetti_acorn_agent_counts_string)
write_line_to_file(x_ibmq_poughkeepsie_agent_counts_string)
write_line_to_file(x_ibmq_melbourne_agent_counts_string)

write_line_to_file(y_grid_average_runtimes_string)
write_line_to_file(y_ibmq_rochester_average_runtimes_string)
write_line_to_file(y_ibmq_paris_average_runtimes_string)
write_line_to_file(y_rigetti_acorn_average_runtimes_string)
write_line_to_file(y_ibmq_poughkeepsie_average_runtimes_string)
write_line_to_file(y_ibmq_melbourne_average_runtimes_string)

write_line_to_file(y_grid_average_total_errors_string)
write_line_to_file(y_ibmq_rochester_average_total_errors_string)
write_line_to_file(y_ibmq_paris_average_total_errors_string)
write_line_to_file(y_rigetti_acorn_average_total_errors_string)
write_line_to_file(y_ibmq_poughkeepsie_average_total_errors_string)
write_line_to_file(y_ibmq_melbourne_average_total_errors_string)

write_line_to_file(y_grid_average_depths_string)
write_line_to_file(y_ibmq_rochester_average_depths_string)
write_line_to_file(y_ibmq_paris_average_depths_string)
write_line_to_file(y_rigetti_acorn_average_depths_string)
write_line_to_file(y_ibmq_poughkeepsie_average_depths_string)
write_line_to_file(y_ibmq_melbourne_average_depths_string)


write_line_to_file("y_runtimes_independent = y_runtimes_independent./1000;")
write_line_to_file("y_runtimes_ibmq_rochester = y_runtimes_ibmq_rochester./1000;")
write_line_to_file("y_runtimes_ibmq_paris = y_runtimes_ibmq_paris./1000;")
write_line_to_file("y_runtimes_rigetti_acorn = y_runtimes_rigetti_acorn./1000;")
write_line_to_file("y_runtimes_ibmq_poughkeepsie = y_runtimes_ibmq_poughkeepsie./1000;")
write_line_to_file("y_runtimes_ibmq_melbourne = y_runtimes_ibmq_melbourne./1000;")

write_line_to_file("y_grid_average_runtimes = y_grid_average_runtimes./1000;")
write_line_to_file("y_ibmq_rochester_average_runtimes = y_ibmq_rochester_average_runtimes./1000;")
write_line_to_file("y_ibmq_paris_average_runtimes = y_ibmq_paris_average_runtimes./1000;")
write_line_to_file("y_rigetti_acorn_average_runtimes = y_rigetti_acorn_average_runtimes./1000;")
write_line_to_file("y_ibmq_poughkeepsie_average_runtimes = y_ibmq_poughkeepsie_average_runtimes./1000;")
write_line_to_file("y_ibmq_melbourne_average_runtimes = y_ibmq_melbourne_average_runtimes./1000;")

# Runtime by instances
write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_instances_independent, y_runtimes_independent, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_ibmq_rochester, y_runtimes_ibmq_rochester, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_ibmq_paris, y_runtimes_ibmq_paris, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_rigetti_acorn, y_runtimes_rigetti_acorn, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_ibmq_poughkeepsie, y_runtimes_ibmq_poughkeepsie, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_ibmq_melbourne, y_runtimes_ibmq_melbourne, 'LineWidth', 2);")

#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
write_line_to_file("set(gca,'YScale','log');")
write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'});")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Runtimes');")
write_line_to_file("xlabel('Instance');")
write_line_to_file("ylabel('Runtime (seconds)');")
write_line_to_file("xlim([0 " + str(len(ibmq_rochester_independent_benchmark_results_sorted["runtimes"])+1) + "]);")

# Average runtime by agent counts
write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_grid_agent_counts, y_grid_average_runtimes, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_rochester_agent_counts, y_ibmq_rochester_average_runtimes, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_paris_agent_counts, y_ibmq_paris_average_runtimes, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_rigetti_acorn_agent_counts, y_rigetti_acorn_average_runtimes, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_poughkeepsie_agent_counts, y_ibmq_poughkeepsie_average_runtimes, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_melbourne_agent_counts, y_ibmq_melbourne_average_runtimes, 'LineWidth', 2);")


#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
write_line_to_file("set(gca,'YScale','log');")
write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'});")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Runtimes');")
write_line_to_file("xlabel('Agent count');")
write_line_to_file("ylabel('Average runtime (seconds)');")
write_line_to_file("xlim([0 " + str(ibmq_rochester_agent_counts[len(ibmq_rochester_agent_counts)-1]+1) + "]);")

# Total error by instance
write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_instances_total_error_grid, y_total_error_grid, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_total_error_ibmq_rochester, y_total_error_ibmq_rochester, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_total_error_ibmq_paris, y_total_error_ibmq_paris, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_total_error_rigetti_acorn, y_total_error_rigetti_acorn, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_total_error_ibmq_poughkeepsie, y_total_error_ibmq_poughkeepsie, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_total_error_ibmq_melbourne, y_total_error_ibmq_melbourne, 'LineWidth', 2);")

#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
#write_line_to_file("set(gca,'YScale','log');")
#write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
#write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'})")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Cost of implementation');")
write_line_to_file("xlabel('Instance');")
write_line_to_file("ylabel('Total error');")
write_line_to_file("xlim([0 " + str(len(ibmq_rochester_total_error_results_sorted["total_errors"])+1) + "]);")
write_line_to_file("ylim([0 1]);")

# Depths by instance
write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_instances_depths_grid, y_depths_grid, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_depths_ibmq_rochester, y_depths_ibmq_rochester, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_depths_ibmq_paris, y_depths_ibmq_paris, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_depths_rigetti_acorn, y_depths_rigetti_acorn, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_depths_ibmq_poughkeepsie, y_depths_ibmq_poughkeepsie, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_depths_ibmq_melbourne, y_depths_ibmq_melbourne, 'LineWidth', 2);")

write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Minimum number of parallel swaps');")
write_line_to_file("xlabel('Instance');")
write_line_to_file("ylabel('Swap Depth');")
write_line_to_file("xlim([0 " + str(len(ibmq_rochester_depths_results_sorted["depths"])+1) + "]);")

# Average total error by agent counts
write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_grid_agent_counts, y_grid_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_rochester_agent_counts, y_ibmq_rochester_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_paris_agent_counts, y_ibmq_paris_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_rigetti_acorn_agent_counts, y_rigetti_acorn_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_poughkeepsie_agent_counts, y_ibmq_poughkeepsie_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_melbourne_agent_counts, y_ibmq_melbourne_average_total_errors, 'LineWidth', 2);")

#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
#write_line_to_file("set(gca,'YScale','log');")
#write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
#write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'})")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Cost of implementation');")
write_line_to_file("xlabel('Agent Count');")
write_line_to_file("ylabel('Average total error');")
write_line_to_file("xlim([0 " + str(ibmq_rochester_agent_counts[len(ibmq_rochester_agent_counts)-1]+1) + "]);")
write_line_to_file("ylim([0 1]);")


# Average depths by agent counts
write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_grid_agent_counts, y_grid_average_depths, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_rochester_agent_counts, y_ibmq_rochester_average_depths, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_paris_agent_counts, y_ibmq_paris_average_depths, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_rigetti_acorn_agent_counts, y_rigetti_acorn_average_depths, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_poughkeepsie_agent_counts, y_ibmq_poughkeepsie_average_depths, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_melbourne_agent_counts, y_ibmq_melbourne_average_depths, 'LineWidth', 2);")

write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Minimum number of parallel swaps');")
write_line_to_file("xlabel('Agent Count');")
write_line_to_file("ylabel('Average swap Depth');")
#write_line_to_file("xlim([0 " + str(ibmq_rochester_agent_counts[len(ibmq_rochester_agent_counts)-1]+1) + "]);")