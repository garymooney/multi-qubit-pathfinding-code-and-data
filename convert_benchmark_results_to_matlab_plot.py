import json
import os
import sys
import numpy as np

def load_dict_from_json(filename):
    try:
        fh = open(filename, 'r')
    except FileNotFoundError:
        print("could not find:", filename)
        return None
    print("found:", filename)
        
    with open(filename) as f:
        return json.load(f)

output_filename = 'matlab_plot_results_benchmark.m'
if os.path.exists(output_filename):
    os.remove(output_filename)
        
def write_line_to_file(text = '', output_filename=output_filename):
    try: 
        with open(output_filename, 'a') as file:
            file.write(text + "\n")

    except IOError:
        with open(output_filename, 'w') as file:
            file.write(text + "\n")
        
     
results_list = []
results_list.append(load_dict_from_json("results-grid-8x8-increasing-independent-agents.txt"))
results_list.append(load_dict_from_json("results-grid-8x8-increasing-mixed-agents.txt"))
results_list.append(load_dict_from_json("results-grid-8x8-increasing-single_group-agents.txt"))

results_sorted_list = []

for results in results_list:
    sorted_indices = np.argsort(results["runtimes"])
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
    
    results_sorted_list.append(results_sorted)
        
grid_independent_benchmark_results_sorted = results_sorted_list[0]
grid_mixed_benchmark_results_sorted = results_sorted_list[1]
grid_group_benchmark_results_sorted = results_sorted_list[2]


x_instances_independent_string = "x_instances_independent = ["
add_space = False
for i in range(len(grid_independent_benchmark_results_sorted["runtimes"])):
    if add_space == True:
        x_instances_independent_string += " "
    else:
        add_space = True
    x_instances_independent_string += str(i+1)
x_instances_independent_string += "];"

y_runtimes_independent_string = "y_runtimes_independent = ["
add_space = False
for runtime in grid_independent_benchmark_results_sorted["runtimes"]:
    if add_space == True:
        y_runtimes_independent_string += " "
    else:
        add_space = True
    y_runtimes_independent_string += str(runtime)
y_runtimes_independent_string += "];"

x_instances_mixed_string = "x_instances_mixed = ["
add_space = False
for i in range(len(grid_mixed_benchmark_results_sorted["runtimes"])):
    if add_space == True:
        x_instances_mixed_string += " "
    else:
        add_space = True
    x_instances_mixed_string += str(i+1)
x_instances_mixed_string += "];"

y_runtimes_mixed_string = "y_runtimes_mixed = ["
add_space = False
for runtime in grid_mixed_benchmark_results_sorted["runtimes"]:
    if add_space == True:
        y_runtimes_mixed_string += " "
    else:
        add_space = True
    y_runtimes_mixed_string += str(runtime)
y_runtimes_mixed_string += "];"

x_instances_group_string = "x_instances_group = ["
add_space = False
for i in range(len(grid_group_benchmark_results_sorted["runtimes"])):
    if add_space == True:
        x_instances_group_string += " "
    else:
        add_space = True
    x_instances_group_string += str(i+1)
x_instances_group_string += "];"

y_runtimes_group_string = "y_runtimes_group = ["
add_space = False
for runtime in grid_group_benchmark_results_sorted["runtimes"]:
    if add_space == True:
        y_runtimes_group_string += " "
    else:
        add_space = True
    y_runtimes_group_string += str(runtime)
y_runtimes_group_string += "];"

write_line_to_file("")
write_line_to_file(x_instances_independent_string)
write_line_to_file(x_instances_mixed_string)
write_line_to_file(x_instances_group_string)
write_line_to_file(y_runtimes_independent_string)
write_line_to_file(y_runtimes_mixed_string)
write_line_to_file(y_runtimes_group_string)

write_line_to_file("y_runtimes_independent = y_runtimes_independent./1000")
write_line_to_file("y_runtimes_mixed = y_runtimes_mixed./1000")
write_line_to_file("y_runtimes_group = y_runtimes_group./1000")

write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_instances_independent, y_runtimes_independent);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_mixed, y_runtimes_mixed);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_instances_group, y_runtimes_group);")

#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
write_line_to_file("set(gca,'YScale','log');")
write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'})")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
write_line_to_file("legend('independent', 'mixed', 'single group', 'Location', 'Best');")
write_line_to_file("title('Runtime Grid 8x8');")
write_line_to_file("xlabel('Instance');")
write_line_to_file("ylabel('Runtime (seconds)');")
write_line_to_file("xlim([0 " + str(len(grid_independent_benchmark_results["runtimes"])+1) + "])")